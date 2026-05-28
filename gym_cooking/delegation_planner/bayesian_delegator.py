import copy
from collections import namedtuple
from itertools import combinations, permutations, product

import navigation_planner.utils as nav_utils
import numpy as np
import scipy as sp
from delegation_planner.delegator import Delegator
from delegation_planner.utils import NEG_INF_LOG_VAL, SubtaskAllocDistribution
from navigation_planner.utils import (
    get_single_actions,
)

SubtaskAllocation = namedtuple("SubtaskAllocation", "subtask subtask_agent_names")


class BayesianDelegator(Delegator):
    def __init__(
        self,
        agent_name,
        all_agent_names,
        model_type,
        planner,
        none_action_prob,
        comm_funcs,
    ):
        """Initializing Bayesian Delegator for agent_name.

        Args:
            agent_name: Str of agent's name.
            all_agent_names: List of str agent names.
            model_type: Str of model type. Must be either "bd"=Bayesian Delegation,
                "fb"=Fixed Beliefs, "up"=Uniform Priors, "dc"=Divide & Conquer,
                "greedy"=Greedy.
            planner: Navigation Planner object, belonging to agent.
            none_action_prob: Float of probability for taking (0, 0) in a None subtask.
        """
        self.name = "Bayesian Delegator"
        self.agent_name = agent_name
        self.all_agent_names = all_agent_names
        self.probs = None
        self.model_type = model_type
        self.priors = "uniform" if model_type == "up" else "spatial"
        self.planner = planner
        self.none_action_prob = none_action_prob
        self.comm_funcs = comm_funcs

    def should_reset_priors(
        self, obs, belief, subtask_to_wrapper_dict, incomplete_subtasks
    ):
        """Returns whether priors should be reset.

        Priors should be reset when 1) They haven't yet been set or
        2) If the possible subtask allocations to infer over have changed.

        Args:
            obs: Copy of the environment object. Current observation
                of environment.
            incomplete_subtasks: List of subtasks. Subtasks have not
                yet been completed according to agent.py.

        Return:
            Boolean of whether or not the subtask allocations have changed.
        """
        if self.probs is None:
            return True

        # Get currently available subtasks.
        self.incomplete_subtasks = incomplete_subtasks
        self.subtask_to_wrapper_dict = subtask_to_wrapper_dict
        probs = self.get_subtask_alloc_probs()
        probs = self.prune_subtask_allocs(
            observation=obs, belief=belief, subtask_alloc_probs=probs
        )

        # Compare previously available subtasks with currently available subtasks.
        cur_subtask_allocs_str_set = set(
            [str(ta) for ta in self.probs.enumerate_subtask_allocs()]
        )
        new_subtask_allocs_str_set = set(
            [str(ta) for ta in probs.enumerate_subtask_allocs()]
        )

        is_reseting_priors = not (
            cur_subtask_allocs_str_set == new_subtask_allocs_str_set
        )

        return is_reseting_priors

    def get_subtask_alloc_probs(self):
        """Return the appropriate belief distribution (determined by model type) over
        subtask allocations (combinations of all_agent_names and incomplete_subtasks)."""
        if self.model_type == "greedy":
            probs = self.add_greedy_subtasks()
        elif self.model_type == "dc":
            probs = self.add_dc_subtasks()
        else:
            probs = self.add_subtasks()
        return probs

    def subtask_alloc_is_doable(self, env, belief, subtask, subtask_agent_names):
        """Return whether subtask allocation (subtask x subtask_agent_names) is doable
        in the current environment state."""

        # Doing nothing is always possible.
        if subtask is None:
            return True

        start_obj, goal_obj = nav_utils.get_subtask_obj(subtask)
        action_obj = nav_utils.get_subtask_action_obj(subtask)

        distance = env.get_bound_for_subtask_given_objs(
            belief=belief,
            subtask=subtask,
            subtask_agent_names=subtask_agent_names,
            start_obj=start_obj,
            goal_obj=goal_obj,
            action_obj=action_obj,
            _type="lower",
            use_holding_penalty=False,
        )

        # Subtask allocation is doable if there's a possibility that the required distance is less than the world perimeter.
        return distance < env.world.perimeter

    def get_bound_for_subtask_alloc(
        self, obs, belief, subtask, subtask_agent_names, ta_probs
    ):
        """Return the value lower bound for a subtask allocation
        (subtask x subtask_agent_names)."""
        if subtask is None:
            print("Subtask is none...")
            return 0

        _ = self.planner.get_next_action(
            env=obs,
            belief=belief,
            subtask=subtask,
            subtask_agent_names=subtask_agent_names,
            ta_probs=ta_probs,
        )

        value_u = self.planner.v_u[
            (
                (
                    self.planner.cur_state.get_repr(),
                    self.planner.cur_belief.get_repr(),
                ),
                (subtask, subtask_agent_names),
            )
        ]

        value_l = self.planner.v_l[
            (
                (
                    self.planner.cur_state.get_repr(),
                    self.planner.cur_belief.get_repr(),
                ),
                (subtask, subtask_agent_names),
            )
        ]

        bounded_v = value_u

        print(
            f"[{self.agent_name}.get_bound_for_subtask_alloc] Got {bounded_v} for {subtask} with {subtask_agent_names}."
        )

        return bounded_v

    def prune_subtask_allocs(self, observation, belief, subtask_alloc_probs):
        """Removing subtask allocs from subtask_alloc_probs that are
        infeasible or where multiple agents are doing None together.
        """

        for subtask_alloc in subtask_alloc_probs.enumerate_subtask_allocs():
            for t in subtask_alloc:
                # Remove unreachable/undoable subtask subtask_allocations.
                if not self.subtask_alloc_is_doable(
                    env=observation,
                    belief=belief,
                    subtask=t.subtask,
                    subtask_agent_names=t.subtask_agent_names,
                ):
                    subtask_alloc_probs.delete(subtask_alloc)
                    break
                # Remove joint Nones (cannot be collaborating on doing nothing).
                if t.subtask is None and len(t.subtask_agent_names) > 1:
                    subtask_alloc_probs.delete(subtask_alloc)
                    break

            # Remove all Nones (at least 1 agent must be doing something).
            if (
                all([t.subtask is None for t in subtask_alloc])
                and len(subtask_alloc) > 1
            ):
                subtask_alloc_probs.delete(subtask_alloc)

        return subtask_alloc_probs

    def set_priors(
        self, obs, belief, incomplete_subtasks, subtask_to_wrapper_dict, priors_type
    ):
        """Setting the prior probabilities for subtask allocations."""
        print("{} setting priors".format(self.agent_name))
        self.incomplete_subtasks = incomplete_subtasks
        self.subtask_to_wrapper_dict = subtask_to_wrapper_dict

        probs = self.get_subtask_alloc_probs()
        probs = self.prune_subtask_allocs(
            observation=obs, belief=belief, subtask_alloc_probs=probs
        )
        probs.normalize()

        if priors_type == "spatial":
            self.probs = self.get_spatial_priors(obs, belief, probs)
        elif priors_type == "uniform":
            # Do nothing because probs already initialized to be uniform.
            self.probs = probs

        self.ensure_at_least_one_subtask()
        self.probs.normalize()

    def get_spatial_priors(self, obs, belief, some_probs):
        """Setting prior probabilities w.r.t spatial metrics."""

        # Weight inversely by distance.
        for subtask_alloc in some_probs.enumerate_subtask_allocs():
            print(
                f"[{self.agent_name}-get_spatial_priors] Getting spatial prior for {subtask_alloc}."
            )

            total_weight = 0
            assigned_agent_cnt = 0
            for t in subtask_alloc:
                if t.subtask is not None:
                    assigned_agent_cnt += 1
                    # Calculate prior with this agent's planner.
                    total_weight += 1.0 / (
                        float(
                            self.get_bound_for_subtask_alloc(
                                obs=copy.copy(obs),
                                belief=copy.copy(belief),
                                subtask=t.subtask,
                                subtask_agent_names=t.subtask_agent_names,
                                ta_probs=copy.copy(some_probs),
                            )
                        )
                    )

            log_p = np.log(total_weight)

            for agent_name, comm in obs.comms.items():
                logit_p = self.comm_funcs.get_logits(agent_name, comm, subtask_alloc)
                print(
                    f'[{self.agent_name}.get_spatial_priors] Task allocation {subtask_alloc} has logit log-probability of {logit_p} for "{comm}".'
                )
                log_p += logit_p

            some_probs.update(
                subtask_alloc=subtask_alloc,
                factor=log_p,
            )
        return some_probs

    def prob_nav_actions(
        self,
        obs_tm1,
        b_tm1,
        actions_tm1,
        subtask,
        subtask_agent_names,
        agent_name,
        beta,
    ):
        """Return probabability that subtask_agents performed subtask, given
        previous observations (obs_tm1) and actions (actions_tm1).

        Args:
            obs_tm1: Copy of environment object. Represents environment at t-1.
            actions_tm1: Dictionary of agent actions. Maps agent str names to tuple actions.
            subtask: Subtask object to perform inference for.
            subtask_agent_names: Tuple of agent str names, of agents who perform subtask.
                subtask and subtask_agent_names make up subtask allocation.
            beta: Beta float value for softmax function.
            no_level_1: Bool, whether to turn off level-k planning.
        Returns:
            A float probability update of whether agents in subtask_agent_names are
            performing subtask.
        """
        print(
            "[BayesianDelgation.prob_nav_actions] Calculating probs for subtask {} by {} as {}".format(
                str(subtask), " & ".join(subtask_agent_names), agent_name
            )
        )
        assert len(subtask_agent_names) == 1 or len(subtask_agent_names) == 2

        # Perform inference over None subtasks.
        if subtask is None:
            assert len(subtask_agent_names) != 2, "Two agents are doing None."
            sim_agent = list(
                filter(lambda a: a.name == agent_name, obs_tm1.sim_agents)
            )[0]
            # Get the number of possible actions at obs_tm1 available to agent.
            num_actions = len(get_single_actions(env=obs_tm1, agent=sim_agent)) - 1
            action_prob = (1.0 - self.none_action_prob) / (
                num_actions
            )  # exclude (0, 0)
            diffs = [self.none_action_prob] + [action_prob] * num_actions
            softmax_diffs = sp.special.softmax(beta * np.asarray(diffs))
            # Probability agents did nothing for None subtask.
            if actions_tm1[subtask_agent_names[0]] == (0, 0):
                return softmax_diffs[0]
            # Probability agents moved for None subtask.
            else:
                return softmax_diffs[1]

        # Perform inference over all non-None subtasks.
        # Calculate Q_{subtask}(obs_tm1, action) for all actions.
        agent_name = obs_tm1.sim_agents[0].name
        action = actions_tm1[agent_name]

        state = obs_tm1
        belief = b_tm1
        self.planner.set_settings(
            env=obs_tm1,
            belief=b_tm1,
            ta_probs=copy.copy(self.probs),
            subtask=subtask,
            subtask_agent_names=subtask_agent_names,
        )
        old_q = self.planner.Q(
            state=state, belief=belief, action=action, value_f=self.planner.v_l
        )

        # Collect actions the agents could have taken in obs_tm1.
        valid_nav_actions = self.planner.get_actions(state=obs_tm1)

        # Check action taken is in the list of actions available to agents in obs_tm1.
        assert action in valid_nav_actions, (
            "valid_nav_actions: {}\nlocs: {}\naction: {}".format(
                valid_nav_actions,
                list(filter(lambda a: a.location, state.sim_agents)),
                action,
            )
        )

        # Calculating the softmax Q_{subtask} for each action.
        qdiffs = [
            old_q
            - self.planner.Q(
                state=state, belief=belief, action=nav_action, value_f=self.planner.v_l
            )
            for nav_action in valid_nav_actions
        ]
        softmax_diffs = sp.special.softmax(beta * np.asarray(qdiffs))

        # Taking the softmax of the action actually taken.
        return softmax_diffs[valid_nav_actions.index(action)]

    def get_other_subtask_allocations(
        self, remaining_agents, remaining_subtasks, base_subtask_alloc
    ):
        """Return a list of subtask allocations to be added onto `subtask_allocs`.

        Each combination should be built off of the `base_subtask_alloc`.
        Add subtasks for all other agents and all other recipe subtasks NOT in
        the ignore set.

        e.g. base_subtask_combo=[
            SubtaskAllocation(subtask=(Chop(T)),
            subtask_agent_names(agent-1, agent-2))]
        To be added on: [
            SubtaskAllocation(subtask=(Chop(L)),
            subtask_agent_names(agent-3,))]
        Note the different subtask and the different agent.
        """
        other_subtask_allocs = []
        if not remaining_agents:
            return [base_subtask_alloc]

        # This case is hit if we have more agents than subtasks.
        if not remaining_subtasks:
            for agent in remaining_agents:
                new_subtask_alloc = base_subtask_alloc + [
                    SubtaskAllocation(subtask=None, subtask_agent_names=tuple(agent))
                ]
                other_subtask_allocs.append(new_subtask_alloc)
            return other_subtask_allocs

        # Otherwise assign remaining agents to remaining subtasks.
        # If only 1 agent left, assign to all remaining subtasks.
        if len(remaining_agents) == 1:
            for t in remaining_subtasks:
                new_subtask_alloc = base_subtask_alloc + [
                    SubtaskAllocation(
                        subtask=t, subtask_agent_names=tuple(remaining_agents)
                    )
                ]
                other_subtask_allocs.append(new_subtask_alloc)
            return other_subtask_allocs
        # If >1 agent remaining, create cooperative and divide & conquer
        # subtask allocations.
        else:
            # Cooperative subtasks (same subtask assigned to remaining agents).
            for t in remaining_subtasks:
                new_subtask_alloc = base_subtask_alloc + [
                    SubtaskAllocation(
                        subtask=t, subtask_agent_names=tuple(remaining_agents)
                    )
                ]
                other_subtask_allocs.append(new_subtask_alloc)
            # Divide and Conquer subtasks (different subtask assigned to remaining agents).
            if len(remaining_subtasks) > 1:
                for ts in product(remaining_subtasks, repeat=2):
                    if ts[0] == ts[1] and self.subtask_to_wrapper_dict[ts[0]].cnt == 1:
                        continue

                    new_subtask_alloc = base_subtask_alloc + [
                        SubtaskAllocation(
                            subtask=ts[0], subtask_agent_names=(remaining_agents[0],)
                        ),
                        SubtaskAllocation(
                            subtask=ts[1], subtask_agent_names=(remaining_agents[1],)
                        ),
                    ]
                    other_subtask_allocs.append(new_subtask_alloc)
            return other_subtask_allocs

    def add_subtasks(self):
        """Return the entire distribution of subtask allocations."""
        subtask_allocs = []

        subtasks = list(self.incomplete_subtasks)
        # Just one agent: Assign itself to all subtasks.
        if len(self.all_agent_names) == 1:
            for t in subtasks:
                subtask_alloc = [
                    SubtaskAllocation(
                        subtask=t, subtask_agent_names=tuple(self.all_agent_names)
                    )
                ]

                subtask_allocs.append(subtask_alloc)
        else:
            for first_agents in combinations(self.all_agent_names, 2):
                # Temporarily add Nones, to allow agents to be allocated no subtask.
                # Later, we filter out allocations where all agents are assigned to None.
                subtasks_temp = subtasks + [
                    None for _ in range(len(self.all_agent_names) - 1)
                ]
                # Cooperative subtasks (same subtask assigned to agents).
                for t in subtasks_temp:
                    subtask_alloc = [
                        SubtaskAllocation(
                            subtask=t, subtask_agent_names=tuple(first_agents)
                        )
                    ]
                    remaining_agents = sorted(
                        list(set(self.all_agent_names) - set(first_agents))
                    )
                    remaining_subtasks = list(set(subtasks_temp) - set([t]))
                    subtask_allocs += self.get_other_subtask_allocations(
                        remaining_agents=remaining_agents,
                        remaining_subtasks=remaining_subtasks,
                        base_subtask_alloc=subtask_alloc,
                    )
                # Divide and Conquer subtasks (different subtask assigned to remaining agents).
                if len(subtasks_temp) > 1:
                    for ts in product(subtasks_temp, repeat=2):
                        if (
                            ts[0] is not None
                            and ts[0] == ts[1]
                            and self.subtask_to_wrapper_dict[ts[0]].cnt == 1
                        ):
                            # Continue if there's only 1 of ts that needs to be accomplished.
                            continue

                        subtask_alloc = [
                            SubtaskAllocation(
                                subtask=ts[0],
                                subtask_agent_names=(first_agents[0],),
                            ),
                            SubtaskAllocation(
                                subtask=ts[1],
                                subtask_agent_names=(first_agents[1],),
                            ),
                        ]
                        remaining_agents = sorted(
                            list(set(self.all_agent_names) - set(first_agents))
                        )
                        remaining_subtasks = list(set(subtasks_temp) - set(ts))
                        subtask_allocs += self.get_other_subtask_allocations(
                            remaining_agents=remaining_agents,
                            remaining_subtasks=remaining_subtasks,
                            base_subtask_alloc=subtask_alloc,
                        )

        return SubtaskAllocDistribution(subtask_allocs)

    def add_greedy_subtasks(self):
        """Return the entire distribution of greedy subtask allocations.
        i.e. subtasks performed only by agent with self.agent_name."""
        subtask_allocs = []

        subtasks = self.incomplete_subtasks
        # At least 1 agent must be doing something.
        if None not in subtasks:
            subtasks += [None]

        # Assign this agent to all subtasks. No joint subtasks because this function
        # only considers greedy subtask allocations.
        for subtask in subtasks:
            subtask_alloc = [
                SubtaskAllocation(
                    subtask=subtask, subtask_agent_names=(self.agent_name,)
                )
            ]
            subtask_allocs.append(subtask_alloc)
        return SubtaskAllocDistribution(subtask_allocs)

    def add_dc_subtasks(self):
        """Return the entire distribution of divide & conquer subtask allocations.
        i.e. no subtask is shared between two agents.

        If there are no subtasks, just make an empty distribution and return."""
        subtask_allocs = []

        subtasks = list(self.incomplete_subtasks) + [
            None for _ in range(len(self.all_agent_names) - 1)
        ]
        for p in permutations(subtasks, len(self.all_agent_names)):
            subtask_alloc = [
                SubtaskAllocation(
                    subtask=p[i], subtask_agent_names=(self.all_agent_names[i],)
                )
                for i in range(len(self.all_agent_names))
            ]
            subtask_allocs.append(subtask_alloc)
        return SubtaskAllocDistribution(subtask_allocs)

    def select_subtask(self, agent_name):
        """Return subtask and subtask_agent_names for agent with agent_name
        with max. probability."""

        max_subtask_alloc = self.probs.get_max()
        if max_subtask_alloc is not None:
            for t in max_subtask_alloc:
                if agent_name in t.subtask_agent_names:
                    return t.subtask, t.subtask_agent_names, max_subtask_alloc
        return (None, agent_name, None)

    def ensure_at_least_one_subtask(self):
        # Make sure each agent has None task by itself.
        if self.model_type == "greedy" or self.model_type == "dc":
            if not self.probs.probs:
                subtask_allocs = [
                    [
                        SubtaskAllocation(
                            subtask=None, subtask_agent_names=(self.agent_name,)
                        )
                    ]
                ]
                self.probs = SubtaskAllocDistribution(subtask_allocs)

    def bayes_update(self, obs_tm1, b_tm1, actions_tm1, comm_info, beta):
        """Apply Bayesian update based on previous observation (obs_tms1)
        and most recent actions taken (actions_tm1). Beta is used to determine
        how rational agents act."""
        # First, remove unreachable/undoable subtask agent subtask_allocs.
        for subtask_alloc in self.probs.enumerate_subtask_allocs():
            for t in subtask_alloc:
                if not self.subtask_alloc_is_doable(
                    env=obs_tm1,
                    belief=b_tm1,
                    subtask=t.subtask,
                    subtask_agent_names=t.subtask_agent_names,
                ):
                    self.probs.delete(subtask_alloc)
                    break

        self.ensure_at_least_one_subtask()

        if self.model_type == "fb":
            return

        ta_set = self.probs.enumerate_subtask_allocs()
        for task_alloc in ta_set:
            update = 0.0
            for t in task_alloc:
                for agent_name in t.subtask_agent_names:
                    if agent_name in actions_tm1:
                        p = self.prob_nav_actions(
                            obs_tm1=copy.copy(obs_tm1),
                            b_tm1=copy.copy(b_tm1),
                            actions_tm1=actions_tm1,
                            subtask=t.subtask,
                            subtask_agent_names=t.subtask_agent_names,
                            agent_name=agent_name,
                            beta=beta,
                        )  # P(a_t | s_t, ta)
                    else:
                        # If the action isn't visible we marginalize
                        # out the action probability.
                        print(
                            f"[BayesianDelgation.bayes_update] Marginalizing {agent_name} action due to no action in obs."
                        )
                        p = 1.0

                    if p != 0:
                        update += np.log(p)
                    else:
                        update += NEG_INF_LOG_VAL

            if comm_info is not None:
                for agent_name, (_, _, comm) in comm_info.items():
                    logit_p = self.comm_funcs.get_logits(agent_name, comm, task_alloc)
                    print(
                        f'[{self.agent_name}.bayes_update] Task allocation {task_alloc} has logit log-probability of {logit_p} for "{comm}".'
                    )
                    update += logit_p
            self.probs.update(subtask_alloc=task_alloc, factor=update)
            print("UPDATING: subtask_alloc {} by {}".format(task_alloc, update))
        self.probs.normalize()
