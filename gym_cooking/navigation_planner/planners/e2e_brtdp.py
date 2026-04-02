# Recipe planning
import copy
import random
import time
from collections import defaultdict
from enum import Enum
from functools import lru_cache
from itertools import product

import navigation_planner.utils as nav_utils
import numpy as np
import scipy as sp
from navigation_planner.utils import MinPriorityQueue as mpq
from recipe_planner.utils import *
from utils.core import *
from utils.interact import interact

# Other core modules
from utils.world import World


def argmin(vector):
    e_x = np.array(vector) == min(vector)
    return np.where(np.random.multinomial(1, e_x / e_x.sum()))[0][0]


def argmax(vector):
    e_x = np.array(vector) == max(vector)
    return np.where(np.random.multinomial(1, e_x / e_x.sum()))[0][0]


class E2E_BRTDP:
    """Bounded Real Time Dynamic Programming (BRTDP) algorithm.

    For more details on this algorithm, please refer to the original
    paper: http://www.cs.cmu.edu/~ggordon/mcmahan-likhachev-gordon.brtdp.pdf
    """

    def __init__(self, alpha, tau, cap, main_cap):
        """
        Initializes BRTDP algorithm with its hyper-parameters.
        Rf. BRTDP paper for how these hyper-parameters are used in their
        algorithm.

        http://www.cs.cmu.edu/~ggordon/mcmahan-likhachev-gordon.brtdp.pdf

        Args:
            alpha: BRTDP convergence criteria.
            tau: BRTDP normalization constant.
            cap: BRTDP cap on sample trial rollouts.
            main_cap: BRTDP main cap on its main loop.
        """
        self.alpha = alpha
        self.tau = tau
        self.cap = cap
        self.main_cap = main_cap

        print("Planner Settings:")
        print(f"\t Main Cap: {self.main_cap}")
        print(f"\t Cap: {self.cap}")

        self.v_l = {}
        self.v_u = {}
        self.repr_to_env_dict = dict()
        self.start = None
        self.start_belief = None
        self.pq = mpq()
        self.actions = World.NAV_ACTIONS
        self.is_joint = False
        self.cur_object_count = 0
        self.is_subtask_complete = lambda h: False
        self.removed_object = None
        self.goal_obj = None

        # Setting up costs for value function.
        self.time_cost = 1.0
        self.action_cost = 0.1

    def __copy__(self):
        copy_ = E2E_BRTDP(
            alpha=self.alpha, tau=self.tau, cap=self.cap, main_cap=self.main_cap
        )
        copy_.__dict__ = self.__dict__.copy()
        return copy_

    def T(self, state, belief, action):
        subtask_agents = self.get_agents(env_state=state)

        sim_state = copy.copy(state)
        if len(subtask_agents) > 1:
            agent_1, agent_2 = subtask_agents
            sim_agent_1 = sim_agent = list(
                filter(lambda a: a.name == agent_1.name, sim_state.sim_agents)
            )[0]
            sim_agent_2 = sim_agent = list(
                filter(lambda a: a.name == agent_2.name, sim_state.sim_agents)
            )[0]
            sim_agent_1.action = action[0]
            sim_agent_2.action = action[1]

            if sim_agent_1.action is not None:
                interact(agent=sim_agent_1, world=sim_state.world)

            if sim_agent_2.action is not None:
                interact(agent=sim_agent_2, world=sim_state.world)

        else:
            agent = subtask_agents[0]
            sim_agent = list(
                filter(lambda a: a.name == agent.name, sim_state.sim_agents)
            )[0]

            sim_agent.action = action

            if sim_agent.action is not None:
                interact(agent=sim_agent, world=sim_state.world)

        self.repr_init(env_state=sim_state, expected_belief=belief)
        self.value_init(env_state=sim_state, belief_state=belief)
        return (sim_state, belief)

    def get_actions(self, state_repr):
        """Returns list of possible actions from current state."""
        if self.subtask is None:
            return [(0, 0)]
        # Convert repr into an environment object.
        state = self.repr_to_env_dict[state_repr]

        subtask_agents = self.get_agents(env_state=state)
        output_actions = []

        # Return single-agent actions.
        if not self.is_joint:
            agent = subtask_agents[0]

            if agent.location is not None:
                output_actions = nav_utils.get_single_actions(env=state, agent=agent)
            else:
                output_actions = [None]
        # Return joint-agent actions.
        else:
            agent_1, agent_2 = subtask_agents

            null_actions = [None]

            if agent_1.location is None:
                output_actions = list(
                    product(
                        null_actions,
                        nav_utils.get_single_actions(env=state, agent=agent_2),
                    )
                )
            else:
                output_actions = list(
                    product(
                        nav_utils.get_single_actions(env=state, agent=agent_1),
                        null_actions,
                    )
                )

        return output_actions

    def runSampleTrial(self):
        """runSampleTrial from BRTDP paper."""
        start_time = time.time()
        x, x_belief = self.start, self.start_belief
        traj = nav_utils.Stack()

        # Terminating if this takes too long e.g. path is infeasible.
        counter = 0
        start_repr = self.start.get_repr()
        start_belief_tuple = self.start_belief.to_tuple()
        diff = (
            self.v_u[((start_repr, start_belief_tuple), self.subtask)]
            - self.v_l[((start_repr, start_belief_tuple), self.subtask)]
        )
        print("DIFF AT START: {}".format(diff))

        while True:
            counter += 1
            if counter > self.cap:
                break
            traj.push((x, x_belief))

            # Get repr of current environment state.
            x_repr, x_belief_tuple = x.get_repr(), x_belief.to_tuple()

            # Get available actions from this state.
            actions = self.get_actions(state_repr=x_repr)

            # We pick actions based on expected state.
            new_upper = min(
                [
                    self.Q(
                        state=x,
                        belief=x_belief,
                        action=a,
                        value_f=self.v_u,
                    )
                    for a in actions
                ]
            )
            self.v_u[((x_repr, x_belief_tuple), self.subtask)] = new_upper

            action_index = argmin(
                [
                    self.Q(
                        state=x,
                        belief=x_belief,
                        action=a,
                        value_f=self.v_u,
                    )
                    for a in actions
                ]
            )
            a = actions[action_index]

            new_lower = self.Q(
                state=x,
                belief=x_belief,
                action=a,
                value_f=self.v_l,
            )
            self.v_l[((x_repr, x_belief_tuple), self.subtask)] = new_lower

            b = self.get_expected_diff(x, x_belief, a)
            B = sum(b.values())
            diff = (
                self.v_u[((start_repr, start_belief_tuple), self.subtask)]
                - self.v_l[((start_repr, start_belief_tuple), self.subtask)]
            ) / self.tau
            if B <= diff:
                break

            x = self.repr_to_env_dict[list(b.keys())[0]]

            # Track this new state in repr dict and value function
            # if it's new.
            self.repr_init(env_state=x, expected_belief=x_belief)
            self.value_init(
                env_state=x,
                belief_state=x_belief,
            )
        print(
            "RUN SAMPLE EXPLORED {} STATES, took {}".format(
                len(traj), time.time() - start_time
            )
        )
        while not (traj.empty()):
            x, x_belief = traj.pop()
            x_repr, x_belief_tuple = x.get_repr(), x_belief.to_tuple()
            actions = self.get_actions(state_repr=x_repr)
            self.v_u[(x_repr, self.subtask)] = min(
                [
                    self.Q(state=x, belief=x_belief, action=a, value_f=self.v_u)
                    for a in actions
                ]
            )

            self.v_l[(x_repr, self.subtask)] = min(
                [
                    self.Q(
                        state=x,
                        belief=x_belief,
                        action=a,
                        value_f=self.v_l,
                    )
                    for a in actions
                ]
            )

    def main(self):
        """Main loop function for BRTDP."""
        main_counter = 0
        start_repr = self.start.get_repr()
        start_belief_tuple = self.start_belief.to_tuple()

        upper = self.v_u[((start_repr, start_belief_tuple), self.subtask)]
        lower = self.v_l[((start_repr, start_belief_tuple), self.subtask)]
        diff = upper - lower

        # Run until convergence or until you max out on iteration
        while (diff > self.alpha) and (main_counter < self.main_cap):
            print("\nstarting main loop #", main_counter)

            new_upper = self.v_u[((start_repr, start_belief_tuple), self.subtask)]
            new_lower = self.v_l[((start_repr, start_belief_tuple), self.subtask)]
            new_diff = new_upper - new_lower
            if new_diff > diff + 0.01:
                self.start.update_display()
                self.start.display()
                self.start.print_agents()
                print("old: upper {}, lower {}".format(upper, lower))
                print("new: upper {}, lower {}".format(new_upper, new_lower))
            diff = new_diff
            upper = new_upper
            lower = new_lower
            main_counter += 1
            print("diff = {}, self.alpha = {}".format(diff, self.alpha))
            self.runSampleTrial()

    def _configure_subtask_information(self, subtask, subtask_agent_names):
        """Tracking information about subtask allocation."""
        # Subtask allocation
        self.subtask = subtask
        self.subtask_agent_names = subtask_agent_names

        assert len(subtask_agent_names) <= 2, (
            "Cannot have more than 2 agents! Hm... {}".format(subtask_agents)
        )
        self.is_joint = len(subtask_agent_names) == 2

        # Relevant objects for subtask allocation.
        self.start_obj, self.goal_obj = nav_utils.get_subtask_obj(subtask)
        self.subtask_action_obj = nav_utils.get_subtask_action_obj(subtask)

    def _define_goal_state(self, env, subtask):
        """Defining a goal state (termination condition on state) for subtask."""
        if subtask is None:
            self.is_goal_state = lambda h, b: True
            return

        if self.is_joint:
            self.is_goal_state = None
            agent = [a for a in env.sim_agents if a.location is not None][0]
            if (
                isinstance(subtask, recipe.Chop)
                or isinstance(subtask, recipe.Cook)
                or isinstance(subtask, recipe.Deliver)
            ):
                action_obj = nav_utils.get_subtask_action_obj(subtask)
                action_obj_locs = env.world.get_all_object_locs(action_obj)

                start_world_locs = env.world.get_all_object_locs(self.start_obj)
                if not len(action_obj_locs) and len(start_world_locs):
                    self.objs_in_shared = agent.objs_shared_cnt.get(
                        self.start_obj.full_name, 0
                    )

                    self.has_more_obj_in_shared = lambda x: int(x) > self.objs_in_shared

                    self.is_goal_state = lambda e, b: self.has_more_obj_in_shared(
                        [a for a in e.sim_agents if a.location is not None][
                            0
                        ].objs_shared_cnt.get(self.start_obj.full_name, 0)
                    )
            elif isinstance(subtask, recipe.Merge):
                so_1_locs = env.world.get_all_object_locs(self.start_obj[0])
                so_2_locs = env.world.get_all_object_locs(self.start_obj[1])

                if len(so_1_locs) and not len(so_2_locs):
                    self.objs_in_shared = agent.objs_shared_cnt.get(
                        self.start_obj[0].full_name, 0
                    )
                    self.has_more_obj_in_shared = lambda x: int(x) > self.objs_in_shared
                    self.is_goal_state = lambda e, b: self.has_more_obj_in_shared(
                        [a for a in e.sim_agents if a.location is not None][
                            0
                        ].objs_shared_cnt.get(self.start_obj[0].full_name, 0)
                    )
                elif not len(so_1_locs) and len(so_2_locs):
                    self.objs_in_shared = agent.objs_shared_cnt.get(
                        self.start_obj[1].full_name, 0
                    )
                    self.has_more_obj_in_shared = lambda x: int(x) > self.objs_in_shared
                    self.is_goal_state = lambda e, b: self.has_more_obj_in_shared(
                        [a for a in e.sim_agents if a.location is not None][
                            0
                        ].objs_shared_cnt.get(self.start_obj[0].full_name, 0)
                    )

        if not self.is_joint or self.is_goal_state is None:
            # Termination condition is when desired object is at a Deliver location.
            if isinstance(subtask, recipe.Deliver):
                self.cur_obj_count = len(
                    list(
                        filter(
                            lambda o: (
                                o
                                in set(
                                    env.world.get_all_object_locs(
                                        self.subtask_action_obj
                                    )
                                )
                            ),
                            env.world.get_object_locs(self.goal_obj, is_held=False),
                        )
                    )
                )
                self.has_more_obj = lambda x: int(x) > self.cur_obj_count
                self.is_goal_state = lambda e, b: self.has_more_obj(
                    len(
                        list(
                            filter(
                                lambda o: (
                                    o
                                    in set(
                                        env.world.get_all_object_locs(
                                            self.subtask_action_obj
                                        )
                                    )
                                ),
                                e.world.get_object_locs(self.goal_obj, is_held=False),
                            )
                        )
                    )
                )

            else:
                # Get current count of desired objects.
                self.cur_obj_count = len(env.world.get_all_object_locs(self.goal_obj))
                # Goal state is reached when the number of desired objects has increased.
                self.has_more_obj = lambda x: int(x) > self.cur_obj_count
                self.is_goal_state = lambda e, b: self.has_more_obj(
                    len(e.world.get_all_object_locs(self.goal_obj))
                )

    def set_settings(self, env, beliefs, subtask, subtask_agent_names):
        """Configure planner."""
        # Configuring subtask related information.
        self._configure_subtask_information(
            subtask=subtask, subtask_agent_names=subtask_agent_names
        )

        # Defining what the goal is for this planner.
        self._define_goal_state(env=env, subtask=subtask)

        # Make sure termination counter has been reset.
        self.counter = 0
        self.num_explorations = 0
        self.stop = False

        # Set start state.
        self.start = copy.copy(env)
        self.start_belief = copy.copy(beliefs)
        self.repr_init(env_state=env, expected_belief=beliefs)
        self.value_init(env_state=env, belief_state=beliefs)

    def get_agents(self, env_state):
        """Return subtask agent for this planner given state."""
        subtask_agents = list(
            filter(lambda a: a.name in self.subtask_agent_names, env_state.sim_agents)
        )

        return subtask_agents

    def repr_init(self, env_state, expected_belief):
        """Initialize repr for environment state."""

        es_repr = env_state.get_repr()
        belief_tuple = expected_belief.to_tuple()

        if es_repr not in self.repr_to_env_dict:
            self.repr_to_env_dict[es_repr] = copy.copy(env_state)

        return (es_repr, belief_tuple)

    def value_init(self, env_state, belief_state, task_alloc_probs):
        subtask = self.subtask

        es_repr = env_state.get_repr()
        belief_tuple = belief_state.to_tuple()
        task_alloc_tuple = task_alloc_probs.to_tuple()
        if ((es_repr, belief_tuple, task_alloc_tuple), subtask) in self.v_l and (
            (es_repr, belief_tuple, task_alloc_tuple),
            subtask,
        ) in self.v_u:
            return

        # Goal state has value 0.
        if self.is_goal_state(env_state, belief_state):
            self.v_l[((es_repr, belief_tuple, task_alloc_tuple), subtask)] = 0.0
            self.v_u[((es_repr, belief_tuple, task_alloc_tuple), subtask)] = 0.0
            return

        # Determine lower bound on this environment state.
        lower = env_state.get_bound_for_subtask_given_objs(
            belief=belief_state,
            task_alloc_probs=task_alloc_probs,
            subtask=subtask,
            subtask_agent_names=self.subtask_agent_names,
            start_obj=self.start_obj,
            goal_obj=self.goal_obj,
            action_obj=self.subtask_action_obj,
            _type="lower",
        )

        upper = env_state.get_bound_for_subtask_given_objs(
            belief=belief_state,
            task_alloc_probs=task_alloc_probs,
            subtask=subtask,
            subtask_agent_names=self.subtask_agent_names,
            start_obj=self.start_obj,
            goal_obj=self.goal_obj,
            action_obj=self.subtask_action_obj,
            _type="upper",
        )

        lower = lower * (self.time_cost + self.action_cost)
        upper = upper * (self.time_cost + self.action_cost)

        # By BRTDP assumption, this should never be negative.
        assert lower > 0, "lower: {}, {}, {}".format(
            lower, env_state.display(), env_state.print_agents()
        )

        self.v_l[((es_repr, belief_tuple, task_alloc_tuple), subtask)] = lower
        self.v_u[((es_repr, belief_tuple, task_alloc_tuple), subtask)] = upper

    def Q(self, state, belief, action, value_f):
        """Get Q value using value_f of (state, belief, action)."""
        cost = self.cost(state, belief, action)

        _ = self.repr_init(env_state=state, expected_belief=belief)
        self.value_init(env_state=state, belief_state=belief)

        # Get next state.
        next_state, next_belief = self.T(state=state, belief=belief, action=action)

        # Initialize new state if it's new.
        ns_repr, belief_tuple = self.repr_init(
            env_state=next_state, expected_belief=next_belief
        )
        self.value_init(env_state=next_state, belief_state=next_belief)

        expected_value = 1.0 * value_f[((ns_repr, belief_tuple), self.subtask)]
        return float(cost + expected_value)

    def V(self, state, belief, _type):
        """Get V*(x) = min_{a \in A} Q_{v*}(x, a)."""

        # Initialize state if it's new.
        s_repr, b_tuple = self.repr_init(env_state=state, expected_belief=belief)

        # Check if this is the desired goal state.
        if self.is_goal_state(state, belief):
            return 0

        # Use lower bound on value function.
        if _type == "lower":
            return min(
                [
                    self.Q(state=state, belief=belief, action=action, value_f=self.v_l)
                    for action in self.get_actions(state_repr=s_repr)
                ]
            )
        # Use upper bound on value function.
        elif _type == "upper":
            return min(
                [
                    self.Q(state=state, belief=belief, action=action, value_f=self.v_u)
                    for action in self.get_actions(state_repr=s_repr)
                ]
            )
        else:
            raise ValueError(
                "Don't recognize the value state function type: {}".format(_type)
            )

    def cost(self, state, belief, action):
        """Return cost of taking action in this state."""
        if action is None:
            return 0

        cost = self.time_cost
        if isinstance(action[0], int):
            action = tuple([action])
        for a in action:
            if a != (0, 0):
                cost += self.action_cost
        return cost

    def get_expected_diff(self, start_state, belief, action):
        # Get next state.
        s_, b_ = self.T(state=start_state, belief=belief, action=action)

        # Initialize state if it's new.
        s_repr, b_tuple = self.repr_init(env_state=s_, expected_belief=b_)
        self.value_init(env_state=s_, belief_state=b_)

        # Get expected diff.
        b = {
            s_repr: 1.0
            * (
                self.v_u[((s_repr, b_tuple), self.subtask)]
                - self.v_l[((s_repr, b_tuple), self.subtask)]
            )
        }
        return b

    def reset_value_caches(self, subtask):
        for state, action in list(self.v_l.keys()):
            if action == subtask:
                del self.v_l[(state, action)]
                del self.v_u[(state, action)]

    def get_next_action(
        self, env, belief, subtask, subtask_agent_names, task_alloc_probs
    ):
        """Return next action."""
        print("-------------[e2e]-----------")
        print(f"Subtask: {subtask}")
        self.removed_object = None
        start_time = time.time()

        # Configure planner settings.
        self.set_settings(
            env=env,
            beliefs=belief,
            subtask=subtask,
            subtask_agent_names=subtask_agent_names,
        )

        cur_state = copy.copy(env)
        cur_belief = copy.copy(belief)

        # BRTDP main loop.
        actions = self.get_actions(state_repr=cur_state.get_repr())
        action_index = argmin(
            [
                self.Q(
                    state=cur_state,
                    belief=cur_belief,
                    action=a,
                    value_f=self.v_l,
                )
                for a in actions
            ]
        )
        a = actions[action_index]
        B = sum(self.get_expected_diff(cur_state, belief, a).values())

        diff = (
            self.v_u[((cur_state.get_repr(), belief.to_tuple()), self.subtask)]
            - self.v_l[((cur_state.get_repr(), belief.to_tuple()), self.subtask)]
        ) / self.tau
        self.cur_state = cur_state
        if B > diff:
            print("exploring, B: {}, diff: {}".format(B, diff))
            self.main()

        # Determine best action after BRTDP.
        if self.is_goal_state(cur_state, belief):
            print("already at goal state...")
            print(f"Time: {time.time() - start_time}")
            return None
        else:
            actions = self.get_actions(cur_state.get_repr())
            qvals = []

            for a in actions:
                q_u = self.Q(
                    state=cur_state,
                    belief=cur_belief,
                    action=a,
                    value_f=self.v_u,
                )
                q_l = self.Q(
                    state=cur_state,
                    belief=cur_belief,
                    action=a,
                    value_f=self.v_u,
                )

                qvals.append(q_u)

            print([x for x in zip(actions, qvals)])
            print(
                "upper is",
                self.v_u[((cur_state.get_repr(), belief.to_tuple()), self.subtask)],
            )
            print(
                "lower is",
                self.v_l[((cur_state.get_repr(), belief.to_tuple()), self.subtask)],
            )

            action_index = argmin(np.array(qvals))
            a = actions[action_index]

            if self.is_joint:
                if a[0] is None:
                    a = a[1]
                else:
                    a = a[0]

            print("chose action:", a)
            print("cost:", self.cost(cur_state, belief, a))
            print(f"Time: {time.time() - start_time}")
            return a
