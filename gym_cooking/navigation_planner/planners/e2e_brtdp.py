# Recipe planning
import copy
import random
import time
from collections import defaultdict
from enum import Enum
from functools import lru_cache
from itertools import product

# Navigation planning
import navigation_planner.utils as nav_utils
import numpy as np
import scipy as sp
from navigation_planner.utils import COMM_ACTION
from navigation_planner.utils import MinPriorityQueue as mpq
from recipe_planner.utils import *
from utils.core import *
from utils.interact import interact

# Other core modules
from utils.world import World


class PlannerLevel(Enum):
    LEVEL1 = 1
    LEVEL0 = 0


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

    def __init__(self, alpha, tau, cap, main_cap, can_communicate):
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
        self.can_communicate = can_communicate

        print("Planner Settings:")
        print(f"\t Main Cap: {self.main_cap}")
        print(f"\t Cap: {self.cap}")

        self.v_l = {}
        self.v_u = {}
        self.repr_to_env_dict = dict()
        self.start = None
        self.start_belief = None
        self.start_task_alloc = None
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
        self.comm_cost = 10.0

    def __copy__(self):
        copy_ = E2E_BRTDP(
            alpha=self.alpha,
            tau=self.tau,
            cap=self.cap,
            main_cap=self.main_cap,
            can_communicate=self.can_communicate,
        )
        copy_.__dict__ = self.__dict__.copy()
        return copy_

    def T(self, state, belief, task_alloc_probs, action):
        subtask_agents = self.get_agents(env_state=state)

        sim_state = copy.copy(state)
        if len(subtask_agents) > 1:
            if COMM_ACTION in action:
                task_alloc_probs = task_alloc_probs.get_comm_dist()

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
            if action == COMM_ACTION:
                task_alloc_probs = task_alloc_probs.get_comm_dist()

            agent = subtask_agents[0]
            sim_agent = list(
                filter(lambda a: a.name == agent.name, sim_state.sim_agents)
            )[0]

            sim_agent.action = action

            if sim_agent.action is not None:
                interact(agent=sim_agent, world=sim_state.world)

        self.repr_init(
            env_state=sim_state, expected_belief=belief, task_alloc=task_alloc_probs
        )
        self.value_init(
            env_state=sim_state, belief_state=belief, task_alloc_probs=task_alloc_probs
        )
        return (sim_state, belief, task_alloc_probs)

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
                output_actions = nav_utils.get_single_actions(
                    env=state, agent=agent, can_communicate=self.can_communicate
                )
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
                        nav_utils.get_single_actions(
                            env=state,
                            agent=agent_2,
                            can_communicate=self.can_communicate,
                        ),
                    )
                )
            else:
                output_actions = list(
                    product(
                        nav_utils.get_single_actions(
                            env=state,
                            agent=agent_1,
                            can_communicate=self.can_communicate,
                        ),
                        null_actions,
                    )
                )

        return output_actions

    def runSampleTrial(self):
        """runSampleTrial from BRTDP paper."""
        start_time = time.time()
        x, x_belief, x_task_alloc_probs = (
            self.start,
            self.start_belief,
            self.start_task_alloc_probs,
        )
        traj = nav_utils.Stack()

        # Terminating if this takes too long e.g. path is infeasible.
        counter = 0
        start_repr = self.start.get_repr()
        start_belief_tuple = self.start_belief.to_tuple()
        start_task_tuple = self.start_task_alloc_probs.to_tuple()
        diff = (
            self.v_u[
                ((start_repr, start_belief_tuple, start_task_tuple), self.subtask_alloc)
            ]
            - self.v_l[
                ((start_repr, start_belief_tuple, start_task_tuple), self.subtask_alloc)
            ]
        )
        print("DIFF AT START: {}".format(diff))

        while True:
            counter += 1
            if counter > self.cap:
                break
            traj.push((x, x_belief, x_task_alloc_probs))

            # Get repr of current environment state.
            x_repr, x_belief_tuple, x_task_alloc_tuple = (
                x.get_repr(),
                x_belief.to_tuple(),
                x_task_alloc_probs.to_tuple(),
            )

            # Get available actions from this state.
            actions = self.get_actions(state_repr=x_repr)

            # We pick actions based on expected state.
            new_upper = min(
                [
                    self.Q(state=modified_state, action=a, value_f=self.v_u)
                    for a in actions
                ]
            )
            self.v_u[
                (
                    (x_repr, x_belief_tuple, x_task_alloc_tuple),
                    tuple(self.subtask_alloc),
                )
            ] = new_upper

            action_index = argmin(
                [
                    self.Q(state=modified_state, action=a, value_f=self.v_l)
                    for a in actions
                ]
            )
            a = actions[action_index]

            new_lower = self.Q(state=modified_state, action=a, value_f=self.v_l)
            self.v_l[(modified_state_repr, self.subtask)] = new_lower

            b = self.get_expected_diff(x, x_belief, x_task_alloc_probs, a)
            B = sum(b.values())
            diff = (
                self.v_u[(start_repr, self.subtask)]
                - self.v_l[(start_repr, self.subtask)]
            ) / self.tau
            if B <= diff:
                break

            x = self.repr_to_env_dict[list(b.keys())[0]]

            # Track this new state in repr dict and value function
            # if it's new.
            self.repr_init(env_state=x)
            self.value_init(env_state=x)

        print(
            "RUN SAMPLE EXPLORED {} STATES, took {}".format(
                len(traj), time.time() - start_time
            )
        )
        while not (traj.empty()):
            x, x_belief, x_task_alloc_probs = traj.pop()
            x_repr, x_belief_tuple, x_task_alloc_tuple = (
                x.get_repr(),
                x_belief.to_tuple(),
                x_task_alloc_probs.to_tuple(),
            )
            actions = self.get_actions(state_repr=x_repr)
            self.v_u[
                (
                    (x_repr, x_belief_tuple, x_task_alloc_tuple),
                    tuple(self.subtask_alloc),
                )
            ] = min(
                [
                    self.Q(
                        state=x,
                        belief=x_belief,
                        task_alloc_probs=x_task_alloc_probs,
                        action=a,
                        value_f=self.v_u,
                    )
                    for a in actions
                ]
            )
            self.v_l[(x_repr, self.subtask)] = min(
                [self.Q(state=x, action=a, value_f=self.v_l) for a in actions]
            )

    def main(self):
        """Main loop function for BRTDP."""
        main_counter = 0
        start_repr = self.start.get_repr()
        start_belief_tuple = self.start_belief.to_tuple()
        start_task_alloc_tuple = self.start_task_alloc_probs.to_tuple()

        upper = self.v_u[
            (
                (start_repr, start_belief_tuple, start_task_alloc_tuple),
                tuple(self.subtask_alloc),
            )
        ]
        lower = self.v_l[
            (
                (start_repr, start_belief_tuple, start_task_alloc_tuple),
                tuple(self.subtask_alloc),
            )
        ]
        diff = upper - lower

        # Run until convergence or until you max out on iteration
        while (diff > self.alpha) and (main_counter < self.main_cap):
            print("\nstarting main loop #", main_counter)
            new_upper = self.v_u[(start_repr, self.subtask)]
            new_lower = self.v_l[(start_repr, self.subtask)]
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

    def _configure_subtask_information(self, subtask_alloc, agent_name):
        """Tracking information about subtask allocation."""

        t = [t for t in subtask_alloc if agent_name in t.subtask_agent_names][0]
        subtask, subtask_agent_names = t

        # Subtask allocation
        if isinstance(subtask_alloc, list):
            subtask_alloc = tuple(subtask_alloc)
        self.subtask_alloc = subtask_alloc
        self.subtask = subtask
        self.subtask_agent_names = subtask_agent_names

        assert len(subtask_agent_names) <= 2, (
            "Cannot have more than 2 agents! Hm... {}".format(subtask_agent_names)
        )
        self.is_joint = len(subtask_agent_names) == 2

        # Relevant objects for subtask allocation.
        self.start_obj, self.goal_obj = nav_utils.get_subtask_obj(subtask)
        self.subtask_action_obj = nav_utils.get_subtask_action_obj(subtask)

    def _define_goal_state(self, env, subtask):
        """Defining a goal state (termination condition on state) for subtask."""
        if subtask is None:
            self.is_goal_state = lambda h, b: False
            return

        if self.is_joint:
            self.is_goal_state = None
            agent = env.get_visible_agent()
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
            self.is_goal_state = lambda h: self.has_more_obj(
                len(self.repr_to_env_dict[h].world.get_all_object_locs(self.goal_obj))
            )
            if self.removed_object is not None and self.removed_object == self.goal_obj:
                self.is_subtask_complete = lambda w: self.has_more_obj(
                    len(w.get_all_object_locs(self.goal_obj)) + 1
                )
            else:
                # Get current count of desired objects.
                self.cur_obj_count = len(env.world.get_all_object_locs(self.goal_obj))
                # Goal state is reached when the number of desired objects has increased.
                self.has_more_obj = lambda x: int(x) > self.cur_obj_count
                self.is_goal_state = lambda e, b: self.has_more_obj(
                    len(e.world.get_all_object_locs(self.goal_obj))
                )

    def _configure_planner_space(self, subtask_agent_names):
        """Configure planner to either plan in joint space or single-agent space."""
        assert len(subtask_agent_names) <= 2, (
            "Cannot have more than 2 agents! Hm... {}".format(subtask_agents)
        )

        self.is_joint = len(subtask_agent_names) == 2

    def set_settings(self, env, subtask, subtask_agent_names, other_agent_planners={}):
        """Configure planner."""
        # Configuring subtask related information.
        agent_name = env.get_visible_agent().name

        self._configure_subtask_information(subtask_alloc, agent_name)

        # Defining what the goal is for this planner.
        self._define_goal_state(env=env, subtask=self.subtask)

        # Make sure termination counter has been reset.
        self.counter = 0
        self.num_explorations = 0
        self.stop = False

        # Set start state.
        self.start_subtask_alloc = copy.copy(subtask_alloc)
        self.start_task_alloc_probs = copy.copy(task_alloc_probs)
        self.start_belief = copy.copy(beliefs)
        self.start = copy.copy(env)
        self.repr_init(
            env_state=env, expected_belief=beliefs, task_alloc=task_alloc_probs
        )
        self.value_init(
            env_state=env, belief_state=beliefs, task_alloc_probs=task_alloc_probs
        )

    def get_agents(self, env_state):
        """Return subtask agent for this planner given state."""
        subtask_agents = list(
            filter(lambda a: a.name in self.subtask_agent_names, env_state.sim_agents)
        )

        return subtask_agents

    def repr_init(self, env_state, expected_belief, task_alloc):
        """Initialize repr for environment state."""

        es_repr = env_state.get_repr()
        belief_tuple = expected_belief.to_tuple()
        task_alloc_tuple = task_alloc.to_tuple()

        if es_repr not in self.repr_to_env_dict:
            self.repr_to_env_dict[es_repr] = copy.copy(env_state)

        return (es_repr, belief_tuple, task_alloc_tuple)

    def value_init(self, env_state, belief_state, task_alloc_probs):
        subtask_alloc = self.subtask_alloc

        es_repr = env_state.get_repr()
        if (es_repr, self.subtask) in self.v_l and (es_repr, self.subtask) in self.v_u:
            return

        # Goal state has value 0.
        if self.is_goal_state(env_state, belief_state):
            self.v_l[((es_repr, belief_tuple, task_alloc_tuple), subtask_alloc)] = 0.0
            self.v_u[((es_repr, belief_tuple, task_alloc_tuple), subtask_alloc)] = 0.0
            return

        # Determine lower bound on this environment state.
        lower = env_state.get_bound_for_subtask_alloc(
            belief_state, subtask_alloc, task_alloc_probs, _type="lower"
        )

        upper = env_state.get_bound_for_subtask_alloc(
            belief_state, subtask_alloc, task_alloc_probs, _type="upper"
        )

        lower = lower * (self.time_cost + self.action_cost)
        upper = upper * (self.time_cost + self.action_cost)

        # By BRTDP assumption, this should never be negative.
        assert lower > 0, "lower: {}, {}, {}".format(
            lower, env_state.display(), env_state.print_agents()
        )

        self.v_l[((es_repr, belief_tuple, task_alloc_tuple), subtask_alloc)] = lower
        self.v_u[((es_repr, belief_tuple, task_alloc_tuple), subtask_alloc)] = upper

    def Q(self, state, action, value_f):
        """Get Q value using value_f of (state, action)."""
        # Q(s,a) = c(x,a) + \sum_{y \in S} P(x, a, y) * v(y)
        cost = self.cost(state, action)

        _ = self.repr_init(
            env_state=state, expected_belief=belief, task_alloc=task_alloc_probs
        )
        self.value_init(
            env_state=state, belief_state=belief, task_alloc_probs=task_alloc_probs
        )

        # Get next state.
        next_state, next_belief, next_task_alloc = self.T(
            state=state, belief=belief, task_alloc_probs=task_alloc_probs, action=action
        )

        # Initialize new state if it's new.
        ns_repr, belief_tuple, task_alloc_tuple = self.repr_init(
            env_state=next_state,
            expected_belief=next_belief,
            task_alloc=next_task_alloc,
        )
        self.value_init(
            env_state=next_state,
            belief_state=next_belief,
            task_alloc_probs=task_alloc_probs,
        )
        try:
            expected_value = (
                1.0
                * value_f[
                    (
                        (ns_repr, belief_tuple, task_alloc_tuple),
                        self.subtask_alloc,
                    )
                ]
            )
            return float(cost + expected_value)
        except Exception as e:
            print(e)
            breakpoint()

    def V(self, state, belief, task_alloc_probs, _type):
        """Get V*(x) = min_{a \in A} Q_{v*}(x, a)."""

        # Initialize state if it's new.
        s_repr, b_tuple, t_tuple = self.repr_init(
            env_state=state, expected_belief=belief, task_alloc=task_alloc_probs
        )

        # Check if this is the desired goal state.
        if self.is_goal_state(state, belief):
            return 0

        # Use lower bound on value function.
        if _type == "lower":
            return min(
                [
                    self.Q(
                        state=state,
                        belief=belief,
                        task_alloc_probs=task_alloc_probs,
                        action=action,
                        value_f=self.v_l,
                    )
                    for action in self.get_actions(state_repr=s_repr)
                ]
            )
        # Use upper bound on value function.
        elif _type == "upper":
            return min(
                [
                    self.Q(
                        state=state,
                        belief=belief,
                        task_alloc_probs=task_alloc_probs,
                        action=action,
                        value_f=self.v_u,
                    )
                    for action in self.get_actions(state_repr=s_repr)
                ]
            )
        else:
            raise ValueError(
                "Don't recognize the value state function type: {}".format(_type)
            )

    def cost(self, state, belief, task_alloc_probs, action):
        """Return cost of taking action in this state."""
        cost = self.time_cost
        if isinstance(action[0], int):
            action = tuple([action])
        for a in action:
            if a != (0, 0) and a != COMM_ACTION:
                cost += self.action_cost
            if a == COMM_ACTION:
                cost += self.comm_cost
        return cost

    def get_expected_diff(self, start_state, belief, task_alloc_probs, action):
        # Get next state.
        s_, b_, t_ = self.T(
            state=start_state,
            belief=belief,
            task_alloc_probs=task_alloc_probs,
            action=action,
        )

        # Initialize state if it's new.
        s_repr, b_tuple, t_tuple = self.repr_init(
            env_state=s_, expected_belief=b_, task_alloc=t_
        )
        self.value_init(env_state=s_, belief_state=b_, task_alloc_probs=t_)

        # Get expected diff.
        subtask_alloc = self.subtask_alloc
        b = {
            s_repr: 1.0
            * (
                self.v_u[((s_repr, b_tuple, t_tuple), subtask_alloc)]
                - self.v_l[((s_repr, b_tuple, t_tuple), subtask_alloc)]
            )
        }
        return b

    def _get_modified_state_with_other_agent_actions(self, state):
        """Do nothing if the planner level is level 0.

        Otherwise, using self.other_agent_planners, anticipate what other agents will do
        and modify the state appropriately.

        Returns the modified state and the actions of other agents that triggered
        the change.
        """
        modified_state = copy.copy(state)
        other_agent_actions = {}

        # Do nothing if the planner level is 0.
        if self.planner_level == PlannerLevel.LEVEL0:
            return modified_state, other_agent_actions

        # Otherwise, modify the state because Level 1 planners
        # consider the actions of other agents.
        for other_agent_name, other_planner in self.other_agent_planners.items():
            # Keep their recipe subtask & subtask agent fixed, but change
            # their planner state to `state`.
            # These new planners should be level 0 planners.
            other_planner.set_settings(
                env=copy.copy(state),
                subtask=other_planner.subtask,
                subtask_agent_names=other_planner.subtask_agent_names,
            )

            assert other_planner.planner_level == PlannerLevel.LEVEL0

            # Figure out what their most likely action is.
            possible_actions = other_planner.get_actions(
                state_repr=other_planner.start.get_repr()
            )
            greedy_action = possible_actions[
                argmin(
                    [
                        other_planner.Q(
                            state=other_planner.start,
                            action=action,
                            value_f=other_planner.v_l,
                        )
                        for action in possible_actions
                    ]
                )
            ]

            if other_planner.is_joint:
                greedy_action = greedy_action[
                    other_planner.subtask_agent_names.index(other_agent_name)
                ]

            # Keep track of their actions.
            other_agent_actions[other_agent_name] = greedy_action
            other_agent = list(
                filter(lambda a: a.name == other_agent_name, modified_state.sim_agents)
            )[0]
            other_agent.action = greedy_action

        # Initialize state if it's new.
        self.repr_init(env_state=modified_state)
        self.value_init(env_state=modified_state)
        return modified_state, other_agent_actions

    def get_next_action(self, env, subtask, subtask_agent_names, other_agent_planners):
        """Return next action."""
        print("-------------[e2e]-----------")
        print(f"Subtask: {subtask_alloc[0]}")
        self.removed_object = None
        start_time = time.time()

        # Configure planner settings.
        self.set_settings(
            env=env,
            beliefs=belief,
            subtask_alloc=subtask_alloc,
            task_alloc_probs=task_alloc_probs,
        )

        cur_state = copy.copy(env)
        cur_belief = copy.copy(belief)
        cur_task_alloc_probs = copy.copy(task_alloc_probs)

        # BRTDP main loop.
        actions = self.get_actions(state_repr=cur_state.get_repr())
        action_index = argmin(
            [self.Q(state=cur_state, action=a, value_f=self.v_l) for a in actions]
        )
        a = actions[action_index]
        B = sum(self.get_expected_diff(cur_state, a).values())
        diff = (
            self.v_u[
                (
                    (
                        cur_state.get_repr(),
                        belief.to_tuple(),
                        task_alloc_probs.to_tuple(),
                    ),
                    self.subtask_alloc,
                )
            ]
            - self.v_l[
                (
                    (
                        cur_state.get_repr(),
                        belief.to_tuple(),
                        task_alloc_probs.to_tuple(),
                    ),
                    self.subtask_alloc,
                )
            ]
        ) / self.tau
        self.cur_state = cur_state
        if B > diff:
            print("exploring, B: {}, diff: {}".format(B, diff))
            self.main()

        # Determine best action after BRTDP.
        if self.is_goal_state(cur_state.get_repr()):
            print("already at goal state, self.cur_obj_count:", self.cur_obj_count)
            return None
        else:
            actions = self.get_actions(state_repr=cur_state.get_repr())
            qvals = [
                self.Q(state=cur_state, action=a, value_f=self.v_l) for a in actions
            ]
            print([x for x in zip(actions, qvals)])
            print("upper is", self.v_u[(cur_state.get_repr(), self.subtask)])
            print("lower is", self.v_l[(cur_state.get_repr(), self.subtask)])

            action_index = argmin(np.array(qvals))
            a = actions[action_index]

            print("chose action:", a)
            print("cost:", self.cost(cur_state, a))
            return a

    def reset_value_caches(self, subtask):
        for state, action in list(self.v_l.keys()):
            if action == subtask:
                del self.v_l[(state, action)]
                del self.v_u[(state, action)]
