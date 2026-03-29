import copy
import time
from collections import deque
from enum import Enum
from functools import lru_cache
from typing import Tuple

import navigation_planner.utils as nav_utils
import numpy as np
from navigation_planner.utils import MinPriorityQueue as mpq
from utils.interact import interact
from utils.world import World

from gym_cooking import recipe_planner
from gym_cooking.recipe_planner.utils import Deliver
from gym_cooking.utils.core import FoodState


class PlannerLevel(Enum):
    LEVEL1 = 1
    LEVEL0 = 0


def argmin(vector):
    e_x = np.array(vector) == min(vector)
    return np.where(np.random.multinomial(1, e_x / e_x.sum()))[0][0]


def argmax(vector):
    e_x = np.array(vector) == max(vector)
    return np.where(np.random.multinomial(1, e_x / e_x.sum()))[0][0]


class E2E_B3RTDP:
    """
    Belief Branch and Bound Real-Time Dynamic Programming Algorithm

    Details on this algorithm: https://www.researchgate.net/publication/364689660_B3RTDP_A_Belief_Branch_and_Bound_Real-Time_Dynamic_Programming_Approach_to_Solving_POMDPs
    """

    def __init__(self, D, alpha, epsilon, beta, tau, depth, main_cap):
        """
        Initializes B3RTDP algorithm with its hyper-parameters.
        Refer to B3RTDP paper for how these hyper-parameters are used in their algorithm.

        https://www.researchgate.net/publication/364689660_B3RTDP_A_Belief_Branch_and_Bound_Real-Time_Dynamic_Programming_Approach_to_Solving_POMDPs

        Args:
            D: Belief discretization factor.
            alpha: Action convergence probability threshold.
            epsilon: Minimum value gap.
            beta: Minimum Convergence Frontier probability.
            tau: Trial termination ratio.
        """

        self.D = D
        self.alpha = alpha
        self.epsilon = epsilon
        self.beta = beta
        self.tau = tau

        self.max_depth = depth
        self.main_cap = main_cap

        self.v_l = {}
        self.v_u = {}
        self.t = {}
        self.repr_to_env_dict = dict()
        self.start = None
        self.pq = mpq()
        self.actions = World.NAV_ACTIONS
        self.is_joint = False
        self.planner_level = PlannerLevel.LEVEL0
        self.cur_object_count = 0
        self.is_subtask_complete = lambda h: False
        self.removed_object = None
        self.goal_obj = None

        # Setting up costs for value function.
        self.time_cost = 1.0
        self.action_cost = 0.1

    def __copy__(self):
        copy_ = E2E_B3RTDP(
            D=self.D,
            alpha=self.alpha,
            epsilon=self.epsilon,
            beta=self.beta,
            tau=self.tau,
            depth=self.max_depth,
            main_cap=self.main_cap,
        )
        copy_.__dict__ = self.__dict__.copy()
        return copy_

    def set_settings(self, env, beliefs, subtask, subtask_agent_names):
        """Configure planner."""
        # Configuring subtask related information.
        self._configure_subtask_information(
            subtask=subtask, subtask_agent_names=subtask_agent_names
        )

        # Defining what the goal is for this planner.
        self._define_goal_state(env=env, subtask=subtask)

        self._canonical_beliefs = copy.deepcopy(beliefs)

        # Make sure termination counter has been reset.
        self.counter = 0
        self.num_explorations = 0
        self.stop = False
        self.num_explorations = 0

        # Set start state.
        self.start = copy.copy(env)
        self.repr_init(env_state=env, expected_belief=beliefs)
        self.value_init(env_state=env, belief_state=beliefs)

    def _configure_subtask_information(self, subtask, subtask_agent_names):
        """Tracking information about subtask allocation."""
        # Subtask allocation
        self.subtask = subtask
        self.subtask_agent_names = subtask_agent_names

        # Relevant objects for subtask allocation.
        self.start_obj, self.goal_obj = nav_utils.get_subtask_obj(subtask)
        self.subtask_action_obj = nav_utils.get_subtask_action_obj(subtask)

    def get_next_action(self, env, belief, subtask, subtask_agent_names, subtasks_set):
        """
        Return next action.
        """
        print("-------------[e2e]-----------")
        start_time = time.time()

        self.action_set_b = {}
        self.subtasks_set = subtasks_set

        cur_state = copy.copy(env)
        cur_belief = copy.copy(belief)

        self.set_settings(
            env=env,
            beliefs=belief,
            subtask=subtask,
            subtask_agent_names=subtask_agent_names,
        )
        self.main(cur_state, cur_belief)
        belief_tuple = cur_belief.to_tuple()
        if self.is_goal_state(cur_state.get_repr(), belief_tuple):
            print("already at goal state, self.cur_obj_count:", self.cur_obj_count)
            return None
        else:
            actions = list(self.action_set_b[(cur_state.get_repr(), belief_tuple)])
            qvals = [
                self.Q(state=cur_state, belief=cur_belief, action=a, value_f=self.v_l)
                for a in actions
            ]
            print([x for x in zip(actions, qvals)])
            print(
                "upper is",
                self.v_u[((cur_state.get_repr(), belief_tuple), self.subtask)],
            )
            print(
                "lower is",
                self.v_l[((cur_state.get_repr(), belief_tuple), self.subtask)],
            )

            action_index = argmin(np.array(qvals))
            a = actions[action_index]

            print("chose action:", a)
            print("cost:", self.cost(cur_state, belief, a))
            return a

    def main(self, cur_state, initial_belief):
        """
        Before we start implementation, we'll want to determine our implementation of beliefs
        and determine how the algorithm will work for those.

        Then, we'll need to figure out how to model the beliefs within Bayesian Delegation.

        This function goes:

            1. InitializeCF(convergence_frontier c, beliefs)
            2. while not TERMINATECF(c)
            3.  b := SAMPLECF(c)
            4.  b^3rtdptrial(c)
            5.  updateCF(c)
            6. return greedy action
        """
        main_counter = 0

        # Initial convergence frontier
        self.value_init(cur_state, initial_belief)
        s_repr, belief_tuple = self.repr_init(cur_state, initial_belief)

        self.cf_set = set([(s_repr, belief_tuple)])
        cf = [(1.0, (cur_state, initial_belief))]

        while not self.terminateCF(cf) and (main_counter < self.main_cap):
            print(f"Main Loop {main_counter}...")

            b = self.sampleCF(cf)
            self.b3rtdpTrial(b)
            self.updateCF(cf)

            main_counter += 1

    def sampleCF(self, cf):
        G = 0
        g = []
        keys = []
        for belief_prob, (cur_state, existence_beliefs) in cf:
            weight = belief_prob * (
                self.V(cur_state, existence_beliefs, "upper")
                - self.V(cur_state, existence_beliefs, "lower")
            )
            g.append(weight)
            keys.append((cur_state, existence_beliefs))
            G += weight

        assert G != 0, "Invalid probability distribution."

        probs = np.array(g) / G
        idx = np.random.choice(len(cf), p=probs)

        return keys[idx]

    def b3rtdpTrial(self, belief_t):
        visited = deque()

        i = 0
        b = belief_t
        while (len(visited) < self.max_depth) and b is not None:
            # print(f"b3rtdpTrial {i}")

            visited.append(b)

            cur_state, existence_beliefs = b

            action_set = self.action_set_b[
                (cur_state.get_repr(), existence_beliefs.to_tuple())
            ]

            a_idx = argmax(
                [
                    self.Q(cur_state, existence_beliefs, a_c, self.v_u)
                    for a_c in action_set
                ]
            )
            a = action_set[a_idx]

            b = self.pick_next_belief(b, belief_t, a)
            i += 1

        while len(visited):
            b = visited.pop()

            cur_state, existence_beliefs = b

            action_set = self.action_set_b[
                (cur_state.get_repr(), existence_beliefs.to_tuple())
            ]
            a_idx = argmax(
                [
                    self.Q(cur_state, existence_beliefs, a_c, self.v_u)
                    for a_c in action_set
                ]
            )
            a = action_set[a_idx]

            self.prune(b, a, action_set)
            self.v_u[
                (
                    (cur_state.get_repr(), existence_beliefs.to_tuple()),
                    self.subtask,
                )
            ] = self.Q(cur_state, existence_beliefs, a, value_f=self.v_u)

            self.v_l[
                (
                    (cur_state.get_repr(), existence_beliefs.to_tuple()),
                    self.subtask,
                )
            ] = max(
                [
                    self.Q(cur_state, existence_beliefs, a_c, self.v_l)
                    for a_c in action_set
                ]
            )

    def prune(self, b, a_best, action_set):
        cur_state, existence_beliefs = b

        for a in list(action_set):
            if a != a_best:
                Q_l_a_best = self.Q(cur_state, existence_beliefs, a_best, self.v_l)
                Q_u_a_best = self.Q(cur_state, existence_beliefs, a_best, self.v_u)

                Q_l_a = self.Q(cur_state, existence_beliefs, a, self.v_l)
                Q_u_a = self.Q(cur_state, existence_beliefs, a, self.v_u)

                Q_u_min = min(Q_u_a_best, Q_u_a)

                p_q_a_best_leq_q_a = 0
                if Q_u_a_best <= Q_l_a:
                    p_q_a_best_leq_q_a = 1
                elif Q_u_a <= Q_l_a_best:
                    p_q_a_best_leq_q_a = 0
                else:
                    p_q_a_best_leq_q_a += (Q_l_a - Q_l_a_best) / (
                        Q_u_a_best - Q_l_a_best
                    )

                    p_q_a_best_leq_q_a += (
                        2 * Q_u_a * Q_u_min - 2 * Q_u_a * Q_l_a - Q_u_min**2 + Q_l_a**2
                    ) / (2 * (Q_u_a_best - Q_l_a_best) * (Q_u_a - Q_l_a))

                if p_q_a_best_leq_q_a < self.epsilon:
                    action_set.remove(a)

    def pick_next_belief(self, b, b_T, a):
        env_state, existence_beliefs = b

        O = self.T(env_state, existence_beliefs, a)

        G = 0
        g = []
        for p, o in O:
            o_state, o_beliefs = o
            val = p * (
                self.V(o_state, o_beliefs, "upper")
                - self.V(o_state, o_beliefs, "lower")
            )
            g.append(val)
            G += val

        b_T_state, b_T_belief = b_T
        if (
            G
            < (
                self.V(b_T_state, b_T_belief, "upper")
                - self.V(b_T_state, b_T_belief, "lower")
            )
            / self.tau
        ):
            return None

        # Samples a belief from O
        probs = np.array(g) / G
        idx = np.random.choice(len(g), p=probs)
        next_state, next_beliefs = O[idx][1]
        return (next_state, next_beliefs)

    def terminateCF(self, convergence_frontier):
        condition_1 = sum([p for p, _ in convergence_frontier]) < self.beta

        condition_2 = (
            sum(
                [
                    p
                    * (
                        self.V(
                            cur_state,
                            existence_belief,
                            "upper",
                        )
                        - self.V(
                            cur_state,
                            existence_belief,
                            "lower",
                        )
                    )
                    for p, (cur_state, existence_belief) in convergence_frontier
                ]
            )
            < self.epsilon
        )

        return condition_1 or condition_2

    def updateCF(self, convergence_frontier):
        new_frontier = []
        new_frontier_index = {}

        for p_b, (cur_state, existence_belief) in convergence_frontier:
            cur_state_repr, existence_belief_tuple = self.repr_init(
                env_state=cur_state, expected_belief=existence_belief
            )

            gap = self.V(cur_state, existence_belief, "upper") - self.V(
                cur_state, existence_belief, "lower"
            )
            if gap < self.epsilon:
                continue

            action_set = self.action_set_b[(cur_state_repr, existence_belief_tuple)]
            if len(action_set) == 1:
                a = list(action_set)[0]
                O = self.T(cur_state, existence_belief, a)
                for p, o in O:
                    next_state, next_existence_belief = o
                    next_state_repr, next_belief_tuple = self.repr_init(
                        next_state, next_existence_belief
                    )

                    increment = p_b * (
                        self.V(next_state, next_existence_belief, "upper")
                        - self.V(next_state, next_existence_belief, "lower")
                    )
                    key = (next_state_repr, next_belief_tuple)
                    if key not in new_frontier_index:
                        new_frontier_index[key] = len(new_frontier)
                        new_frontier.append(
                            (increment, (next_state, next_existence_belief))
                        )
                    else:
                        idx = new_frontier_index[key]
                        prev_p, prev_b = new_frontier[idx]
                        new_frontier[idx] = (prev_p + increment, prev_b)
            else:
                key = (cur_state_repr, existence_belief_tuple)
                if key not in new_frontier_index:
                    new_frontier_index[key] = len(new_frontier)
                    new_frontier.append((p_b, (cur_state, existence_belief)))
                else:
                    idx = new_frontier_index[key]
                    prev_p, prev_b = new_frontier[idx]
                    new_frontier[idx] = (prev_p + p_b, prev_b)

        convergence_frontier[:] = new_frontier
        self.cf_set = set(new_frontier_index.keys())

    def cost(self, state, belief, action):
        """Return cost of taking action in this state."""
        cost = self.time_cost
        if isinstance(action[0], int):
            action = tuple([action])
        for a in action:
            if a != (0, 0):
                cost += self.action_cost
        return cost

    def reset_value_caches(self, subtask):
        for state, action in list(self.v_l.keys()):
            if action == subtask:
                del self.v_l[(state, action)]
                del self.v_u[(state, action)]

    def repr_init(self, env_state, expected_belief):
        """Initialize repr for environment state."""

        es_repr = env_state.get_repr()
        belief_tuple = expected_belief.to_tuple()

        if es_repr not in self.repr_to_env_dict:
            self.repr_to_env_dict[es_repr] = copy.copy(env_state)

        if (es_repr, belief_tuple) not in self.action_set_b:
            self.action_set_b[(es_repr, belief_tuple)] = self.get_actions(es_repr)
        return (es_repr, belief_tuple)

    def get_agents(self, env_state):
        """Return subtask agent for this planner given state."""
        subtask_agents = list(
            filter(lambda a: a.name in self.subtask_agent_names, env_state.sim_agents)
        )

        return subtask_agents

    def get_actions(self, state_repr):
        """Returns list of possible actions from current state."""
        if self.subtask is None:
            return [(0, 0)]
        # Convert repr into an environment object.
        state = self.repr_to_env_dict[state_repr]

        subtask_agents = self.get_agents(env_state=state)

        agent = next(
            (agent for agent in subtask_agents if agent.location is not None), None
        )
        if agent is None:
            raise Exception("Could not find agent in get_actions()")
        output_actions = nav_utils.get_single_actions(env=state, agent=agent)

        return output_actions

    def T(self, state, belief, action, subtask=None):
        if subtask is None:
            subtask = self.subtask

        subtask_agents = self.get_agents(env_state=state)

        if not subtask.is_joint:
            agent = subtask_agents[0]
            sim_state = copy.copy(state)
            sim_agent = list(
                filter(lambda a: a.name == agent.name, sim_state.sim_agents)
            )[0]
            sim_agent.action = action
            try:
                interact(agent=sim_agent, world=sim_state.world)
            except Exception as e:
                print(e)
                breakpoint()
        else:
            agent_1, agent_2 = subtask_agents
            # Corrects so the observable agent is agent_1
            if agent_1.location is None:
                agent_1, agent_2 = agent_2, agent_1

            sim_state = copy.copy(state)
            sim_agent_1 = list(
                filter(lambda a: a.name == agent_1.name, sim_state.sim_agents)
            )[0]
            sim_agent_2 = list(
                filter(lambda a: a.name == agent_2.name, sim_state.sim_agents)
            )[0]

            sim_agent_1.action = action

            start_obj, _ = nav_utils.get_subtask_obj(subtask)
            if not isinstance(start_obj, list):
                start_obj = [start_obj]

            locs = set()
            for obj in start_obj:
                for loc in sim_state.world.get_all_object_locs(obj):
                    locs.add(loc)

            min_loc = sim_state.world.share_space_locs[0]
            min_dist = float("inf")
            for loc in state.world.share_space_locs:
                dist = nav_utils.manhattan_dist(sim_agent_1.location, loc)
                if dist < min_dist:
                    min_dist = dist
                    min_loc = loc

            if not any([loc not in sim_state.world.shared_space_locs for loc in locs]):
                start_obj.location = min_loc
                sim_state.world.insert(start_obj)
            else:
                # Used for planning purposes
                if sim_agent_1.observable_cols[0] == 0:
                    sim_agent_2.location = (min_loc[0] + 1, min_loc[1])
                else:
                    sim_agent_2.location = (min_loc[0] - 1, min_loc[1])

            interact(agent=sim_agent_1, world=sim_state.world)
            assert sim_agent_1.location != sim_agent_2.location, (
                "action {} led to state {}".format(action, sim_state.get_repr())
            )

        self.repr_init(env_state=sim_state, expected_belief=belief)
        self.value_init(env_state=sim_state, belief_state=belief)
        return [(1.0, (sim_state, belief))]

    def V(self, cur_state, belief, _type, subtask=None):
        """Get V*(b) = min_{a \in A} Q_{v*}(b, a)."""
        if subtask is None:
            subtask = self.subtask

        # Initialize state if it's new.
        s_repr, belief_tuple = self.repr_init(
            env_state=cur_state, expected_belief=belief
        )

        if _type == "lower" and ((s_repr, belief_tuple), subtask) in self.v_l:
            return self.v_l[((s_repr, belief_tuple), subtask)]
        elif _type == "upper" and ((s_repr, belief_tuple), subtask) in self.v_u:
            return self.v_u[((s_repr, belief_tuple), subtask)]

        # Check if this is the desired goal state.
        if self.is_goal_state(s_repr, belief_tuple):
            return 0

        # Use lower bound on value function.
        if _type == "lower":
            return min(
                [
                    self.Q(
                        state=cur_state,
                        belief=belief,
                        action=action,
                        value_f=self.v_l,
                        subtask=subtask,
                    )
                    for action in self.get_actions(state_repr=s_repr)
                ]
            )
        # Use upper bound on value function.
        elif _type == "upper":
            return min(
                [
                    self.Q(
                        state=cur_state,
                        belief=belief,
                        action=action,
                        value_f=self.v_u,
                        subtask=subtask,
                    )
                    for action in self.get_actions(state_repr=s_repr)
                ]
            )
        else:
            raise ValueError(
                "Don't recognize the value state function type: {}".format(_type)
            )

    def Q(self, state, belief, action, value_f, subtask=None):
        """Get Q value using value_f of (state, belief, action)."""
        if subtask is None:
            subtask = self.subtask

        cost = self.cost(state, belief, action)

        s_repr, belief_tuple = self.repr_init(env_state=state, expected_belief=belief)
        self.value_init(env_state=state, belief_state=belief)

        O = self.T(state=state, belief=belief, action=action)

        expected_value = 0
        for p, o in O:
            next_state, next_beliefs = o
            ns_repr, ns_belief_tuple = self.repr_init(
                env_state=next_state, expected_belief=next_beliefs
            )
            self.value_init(env_state=next_state, belief_state=next_beliefs)

            expected_value += p * value_f[((ns_repr, ns_belief_tuple), subtask)]

        return float(cost + expected_value)

    def value_init(self, env_state, belief_state, subtask=None):
        if subtask is None:
            subtask = self.subtask

        es_repr = env_state.get_repr()
        belief_tuple = belief_state.to_tuple()
        if ((es_repr, belief_tuple), subtask) in self.v_l and (
            (es_repr, belief_tuple),
            subtask,
        ) in self.v_u:
            return

        # Goal state has value 0.
        if self.is_goal_state(es_repr, belief_tuple):
            self.v_l[((es_repr, belief_tuple), subtask)] = 0.0
            self.v_u[((es_repr, belief_tuple), subtask)] = 0.0
            return

        # Determine lower bound on this environment state.
        lower = env_state.get_bound_for_subtask_given_objs(
            subtask=subtask,
            subtask_agent_names=self.subtask_agent_names,
            start_obj=self.start_obj,
            goal_obj=self.goal_obj,
            action_obj=self.subtask_action_obj,
            _type="lower",
        )

        upper = env_state.get_bound_for_subtask_given_objs(
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

        self.v_l[((es_repr, belief_tuple), subtask)] = lower
        self.v_u[((es_repr, belief_tuple), subtask)] = upper

    def _define_goal_state(self, env, subtask):
        """Defining a goal state (termination condition on state) for subtask."""

        if subtask is None:
            self.is_goal_state = lambda h, b: True

        # Termination condition is when desired object is at a Deliver location.
        elif isinstance(subtask, Deliver):
            self.cur_obj_count = len(
                list(
                    filter(
                        lambda o: (
                            o
                            in set(
                                env.world.get_all_object_locs(self.subtask_action_obj)
                            )
                        ),
                        env.world.get_object_locs(self.goal_obj, is_held=False),
                    )
                )
            )
            self.has_more_obj = lambda x: int(x) > self.cur_obj_count
            self.is_goal_state = lambda h, b: self.has_more_obj(
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
                            self.repr_to_env_dict[h].world.get_object_locs(
                                self.goal_obj, is_held=False
                            ),
                        )
                    )
                )
            )

            if self.removed_object is not None and self.removed_object == self.goal_obj:
                self.is_subtask_complete = lambda w, b: self.has_more_obj(
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
                                w.get_object_locs(self.goal_obj, is_held=False),
                            )
                        )
                    )
                    + 1
                )
            else:
                self.is_subtask_complete = lambda w, b: self.has_more_obj(
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
                                w.get_object_locs(obj=self.goal_obj, is_held=False),
                            )
                        )
                    )
                )
        else:
            # Get current count of desired objects.
            self.cur_obj_count = len(env.world.get_all_object_locs(self.goal_obj))
            # Goal state is reached when the number of desired objects has increased.
            self.has_more_obj = lambda x: int(x) > self.cur_obj_count
            self.is_goal_state = lambda h, b: self.has_more_obj(
                len(self.repr_to_env_dict[h].world.get_all_object_locs(self.goal_obj))
            )
            if self.removed_object is not None and self.removed_object == self.goal_obj:
                self.is_subtask_complete = lambda w, b: self.has_more_obj(
                    len(w.get_all_object_locs(self.goal_obj)) + 1
                )
            else:
                self.is_subtask_complete = lambda w, b: self.has_more_obj(
                    len(w.get_all_object_locs(self.goal_obj))
                )
