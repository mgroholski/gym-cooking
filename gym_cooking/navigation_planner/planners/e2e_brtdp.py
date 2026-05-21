# Recipe planning
import copy
import time
from functools import lru_cache

import navigation_planner.utils as nav_utils
import numpy as np
from navigation_planner.utils import MinPriorityQueue as mpq
from recipe_planner.utils import Deliver, Merge

# Navigation planning
from utils.belief import get_cnt_str
from utils.core import *
from utils.interact import interact
from utils.utils import BELIEF_THRESHOLD

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

        self.v_l = {}
        self.v_u = {}

        self.start = None
        self.start_belief = None
        self.pq = mpq()
        self.actions = World.NAV_ACTIONS
        self.cur_object_count = 0
        self.is_subtask_complete = lambda h, b: False
        self.removed_object = None
        self.goal_obj = None

        self.state_repr_to_env = {}
        self.belief_repr_to_belief = {}

        # Setting up costs for value function.
        self.time_cost = 1.0
        self.action_cost = 0.1

    def __copy__(self):
        copy_ = E2E_BRTDP(
            alpha=self.alpha, tau=self.tau, cap=self.cap, main_cap=self.main_cap
        )
        copy_.__dict__ = self.__dict__.copy()
        return copy_

    @lru_cache(maxsize=10000)
    def T(self, state_repr, belief_repr, action, subtask, subtask_agent_names):
        """Return next states and their probabilities when taking action from state."""
        state, belief = (
            self.state_repr_to_env[state_repr],
            self.belief_repr_to_belief[belief_repr],
        )

        base_state = copy.copy(state)
        base_state.sim_agents[0].action = action
        a_dict = {base_state.sim_agents[0].name: action}

        interact(agent=base_state.sim_agents[0], world=base_state.world)

        if len(subtask_agent_names) == 1:
            # For a single agent task, we don't care what the other agent is going
            # thus we simulate a transition in the observable environment.
            return [(1.0, base_state, belief)]

        start_obj, goal_obj = nav_utils.get_subtask_obj(subtask)
        action_obj = nav_utils.get_subtask_action_obj(subtask)
        if not isinstance(start_obj, (list, tuple)):
            start_obj = [start_obj]

        s_j_area = ((base_state.world.height - 2) * (base_state.world.width - 3)) / 2.0
        s_j_area_prob = 1.0 / s_j_area

        base_state_prob = 1.0

        transition_probs_and_states = []  # (prob, state, belief)

        # For each subtask start object and S^s, we expect
        # the other agent to pick it up if the agent is in the correct location
        # and the other needed item is within S^j, the action object or, in the
        # case of Merge tasks, the other merge object.
        open_shared_locs = []
        for loc in base_state.world.shared_space_locs:
            gs = base_state.world.get_gridsquare_at(loc)
            gs_holding = gs.holding

            if gs.holding is not None:
                if isinstance(subtask, Merge):
                    if gs_holding.full_name == start_obj[0].full_name:
                        other_prob = belief[start_obj[1].full_name]
                    elif gs_holding.full_name == start_obj[1].full_name:
                        other_prob = belief[start_obj[1].full_name]
                    else:
                        other_prob = 0.0
                else:
                    other_prob = belief[action_obj.name]

                if other_prob > 0.0:
                    new_state = copy.copy(base_state)
                    new_state_gs = new_state.world.get_gridsquare_at(loc)
                    new_state_gs_holding = new_state_gs.holding
                    new_state.world.remove(new_state_gs_holding)
                    new_state_gs.holding = None

                    new_belief = copy.copy(belief)
                    new_belief.update(
                        new_state, state, a_dict, self.ta_probs, verbose=False
                    )

                    p = s_j_area_prob * other_prob
                    transition_probs_and_states.append((p, new_state, new_belief))
                    base_state_prob *= 1.0 - p
            else:
                open_shared_locs.append(loc)

        if isinstance(subtask, Deliver) and not len(
            base_state.world.objects["Delivery"]
        ):
            for idx, task in enumerate(base_state.task_queue):
                if (
                    not task.is_complete
                    and task.recipe.full_state_plate_name == goal_obj.full_name
                ):
                    new_state = copy.copy(base_state)
                    new_state.task_queue[idx].is_complete = True

                    start_obj_name = task.recipe.full_state_plate_name
                    start_obj_belief = belief[start_obj_name]
                    delivery_belief = belief["Delivery"]

                    new_belief = copy.copy(belief)
                    new_belief.update(
                        new_state, state, a_dict, self.ta_probs, verbose=False
                    )

                    p = (
                        s_j_area_prob * start_obj_belief * delivery_belief
                    )  # correct location and start obj exists and delivery exists
                    transition_probs_and_states.append(
                        (
                            p,
                            new_state,
                            new_belief,
                        )
                    )
                    base_state_prob *= 1.0 - p
        else:
            # For each object that's a start or
            # goal object we simulate the agent setting down into the middle based
            # on the probability of the object existing and the agent being in the correct position.
            for obj in start_obj + [goal_obj]:
                if obj is None:
                    continue

                k = obj.full_name
                v = belief[k]

                if v > 0.0:
                    updated_belief = None
                    for open_shared_loc in open_shared_locs:
                        new_state = copy.copy(base_state)
                        new_state_gs = new_state.world.get_gridsquare_at(
                            open_shared_loc
                        )

                        new_state_obj = belief.get_name_to_obj(k)
                        new_state_obj.location = open_shared_loc
                        new_state_gs.holding = new_state_obj
                        new_state.world.insert(new_state_obj)

                        if updated_belief is None:
                            updated_belief = copy.copy(belief)
                            updated_belief.update(
                                new_state,
                                state,
                                a_dict,
                                self.ta_probs,
                                verbose=False,
                            )
                        new_belief = copy.copy(updated_belief)

                        p = s_j_area_prob * v
                        base_state_prob *= 1.0 - p
                        transition_probs_and_states.append((p, new_state, new_belief))

        assert base_state_prob >= 0.0
        base_belief = copy.copy(belief)
        base_belief.update(base_state, state, a_dict, self.ta_probs, verbose=False)
        transition_probs_and_states.append((base_state_prob, base_state, base_belief))

        for _, s_, b_ in transition_probs_and_states:
            self.repr_init(s_, b_)
            self.value_init(s_, b_)

        return transition_probs_and_states

    def get_actions(self, state):
        """Returns list of possible actions from current state."""
        if self.subtask is None:
            return [(0, 0)]

        agent = state.sim_agents[0]
        output_actions = nav_utils.get_single_actions(env=state, agent=agent)
        return output_actions

    def runSampleTrial(self):
        """runSampleTrial from BRTDP paper."""
        start_time = time.time()
        x_state, x_belief = self.start, self.start_belief
        traj = nav_utils.Stack()

        # Terminating if this takes too long e.g. path is infeasible.
        counter = 0
        start_repr, start_belief_repr = (
            self.start.get_repr(),
            self.start_belief.get_repr(),
        )

        diff = (
            self.v_u[
                (
                    (start_repr, start_belief_repr),
                    (self.subtask, self.subtask_agent_names),
                )
            ]
            - self.v_l[
                (
                    (start_repr, start_belief_repr),
                    (self.subtask, self.subtask_agent_names),
                )
            ]
        )
        print("DIFF AT START: {}".format(diff))

        while True:
            counter += 1
            if counter > self.cap:
                break
            traj.push((x_state, x_belief))

            # Get repr of current environment state.
            x_repr, x_belief_repr = self.repr_init(x_state, x_belief)

            # Get available actions from this state.
            actions = self.get_actions(state=x_state)

            # We pick actions based on expected state.
            new_upper = min(
                [
                    self.Q(state=x_state, belief=x_belief, action=a, value_f=self.v_u)
                    for a in actions
                ]
            )
            self.v_u[
                ((x_repr, x_belief_repr), (self.subtask, self.subtask_agent_names))
            ] = new_upper

            action_index = argmin(
                [
                    self.Q(state=x_state, belief=x_belief, action=a, value_f=self.v_l)
                    for a in actions
                ]
            )
            a = actions[action_index]

            new_lower = self.Q(
                state=x_state, belief=x_belief, action=a, value_f=self.v_l
            )
            self.v_l[
                ((x_repr, x_belief_repr), (self.subtask, self.subtask_agent_names))
            ] = new_lower

            b = self.get_expected_diff(x_state.get_repr(), x_belief.get_repr(), a)
            B = sum(b.values())
            diff = (
                self.v_u[
                    (
                        (start_repr, start_belief_repr),
                        (self.subtask, self.subtask_agent_names),
                    )
                ]
                - self.v_l[
                    (
                        (start_repr, start_belief_repr),
                        (self.subtask, self.subtask_agent_names),
                    )
                ]
            ) / self.tau
            if B <= diff:
                break

            # Track this new state in repr dict and value function
            # if it's new.
            self.value_init(env_state=x_state, belief_state=x_belief)
            self.repr_init(state=x_state, belief=x_belief)

        print(
            "RUN SAMPLE EXPLORED {} STATES, took {}".format(
                len(traj), time.time() - start_time
            )
        )
        while not (traj.empty()):
            x_state, x_belief = traj.pop()
            x_repr, x_belief_tuple = x_state.get_repr(), x_belief.get_repr()
            actions = self.get_actions(state=x_state)
            self.v_u[
                ((x_repr, x_belief_tuple), (self.subtask, self.subtask_agent_names))
            ] = min(
                [
                    self.Q(state=x_state, belief=x_belief, action=a, value_f=self.v_u)
                    for a in actions
                ]
            )
            self.v_l[
                ((x_repr, x_belief_tuple), (self.subtask, self.subtask_agent_names))
            ] = min(
                [
                    self.Q(state=x_state, belief=x_belief, action=a, value_f=self.v_l)
                    for a in actions
                ]
            )

    def main(self):
        """Main loop function for BRTDP."""
        main_counter = 0
        start_repr, start_belief_repr = (
            self.start.get_repr(),
            self.start_belief.get_repr(),
        )

        upper = self.v_u[
            ((start_repr, start_belief_repr), (self.subtask, self.subtask_agent_names))
        ]
        lower = self.v_l[
            ((start_repr, start_belief_repr), (self.subtask, self.subtask_agent_names))
        ]
        diff = upper - lower

        # Run until convergence or until you max out on iteration
        while (diff > self.alpha) and (main_counter < self.main_cap):
            print("\nstarting main loop #", main_counter)
            new_upper = self.v_u[
                (
                    (start_repr, start_belief_repr),
                    (self.subtask, self.subtask_agent_names),
                )
            ]
            new_lower = self.v_l[
                (
                    (start_repr, start_belief_repr),
                    (self.subtask, self.subtask_agent_names),
                )
            ]
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

    def _configure_subtask_information(self, subtask, subtask_agent_names, ta_probs):
        """Tracking information about subtask allocation."""
        # Task Allocation Distribution
        self.ta_probs = copy.copy(ta_probs)

        # Subtask allocation
        self.subtask = subtask
        self.subtask_agent_names = subtask_agent_names

        # Relevant objects for subtask allocation.
        self.start_obj, self.goal_obj = nav_utils.get_subtask_obj(subtask)
        self.subtask_action_obj = nav_utils.get_subtask_action_obj(subtask)

    def _define_goal_state(self, env, subtask):
        """Defining a goal state (termination condition on state) for subtask."""

        if subtask is None:
            self.is_goal_state = lambda h, b: True

        # Termination condition is when desired object is at a Deliver location.
        elif isinstance(subtask, Deliver):
            self.task_idx = next(
                (
                    i
                    for i, t in enumerate(env.task_queue)
                    if t.recipe.goal == self.goal_obj.to_predicate()
                    and not t.is_complete
                ),
                None,
            )

            def delivered_goal_state(w, b):
                return w.task_queue[self.task_idx].is_complete

            self.is_goal_state = lambda e, b: delivered_goal_state(e.world, b)
            self.is_subtask_complete = delivered_goal_state
        else:
            # Get current count of desired objects.
            self.cur_obj_count = len(env.world.get_all_object_locs(self.goal_obj))
            # Goal state is reached when the number of desired objects has increased.
            self.has_more_obj = lambda x: int(x) > self.cur_obj_count

            def is_goal_state(w, b):
                return (
                    self.has_more_obj(len(w.get_all_object_locs(self.goal_obj)))
                    or b[get_cnt_str(self.goal_obj)] >= BELIEF_THRESHOLD
                )  # Observable objects or meets threshold

            self.is_goal_state = lambda e, b: is_goal_state(e.world, b)
            self.is_subtask_complete = is_goal_state

    def set_settings(self, env, belief, ta_probs, subtask, subtask_agent_names):
        """Configure planner."""
        # Configuring subtask related information.
        self._configure_subtask_information(
            subtask=subtask, subtask_agent_names=subtask_agent_names, ta_probs=ta_probs
        )

        # Defining what the goal is for this planner.
        self._define_goal_state(env=env, subtask=subtask)

        # Make sure termination counter has been reset.
        self.counter = 0
        self.num_explorations = 0
        self.stop = False
        self.num_explorations = 0

        # Set start state.
        self.start = copy.copy(env)
        self.start_belief = copy.copy(belief)
        self.value_init(env_state=env, belief_state=belief)
        self.repr_init(state=env, belief=belief)

    def repr_init(self, state, belief):
        state_repr, belief_repr = state.get_repr(), belief.get_repr()
        if state_repr not in self.state_repr_to_env:
            state.comms = {}
            self.state_repr_to_env[state_repr] = state

        if belief_repr not in self.belief_repr_to_belief:
            self.belief_repr_to_belief[belief_repr] = belief

        return state_repr, belief_repr

    def value_init(self, env_state, belief_state):
        """Initialize value for environment state."""
        # Skip if already initialized.
        es_repr, belief_repr = env_state.get_repr(), belief_state.get_repr()
        if (
            (es_repr, belief_repr),
            (self.subtask, self.subtask_agent_names),
        ) in self.v_l and (
            (es_repr, belief_repr),
            (self.subtask, self.subtask_agent_names),
        ) in self.v_u:
            return

        # Goal state has value 0.
        if self.is_goal_state(env_state, belief_state):
            self.v_l[
                ((es_repr, belief_repr), (self.subtask, self.subtask_agent_names))
            ] = 0.0
            self.v_u[
                ((es_repr, belief_repr), (self.subtask, self.subtask_agent_names))
            ] = 0.0
            return

        # Determine lower bound on this environment state.
        lower = env_state.get_bound_for_subtask_given_objs(
            belief=belief_state,
            subtask=self.subtask,
            subtask_agent_names=self.subtask_agent_names,
            start_obj=self.start_obj,
            goal_obj=self.goal_obj,
            action_obj=self.subtask_action_obj,
            _type="lower",
        )

        upper = env_state.get_bound_for_subtask_given_objs(
            belief=belief_state,
            subtask=self.subtask,
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

        self.v_l[((es_repr, belief_repr), (self.subtask, self.subtask_agent_names))] = (
            lower
        )
        self.v_u[((es_repr, belief_repr), (self.subtask, self.subtask_agent_names))] = (
            upper
        )

    def Q(self, state, belief, action, value_f):
        """Get Q value using value_f of (state, action)."""
        # Q(s,a) = c(x,a) + \sum_{y \in S} P(x, a, y) * v(y)
        cost = self.cost(state, belief, action)

        # Initialize state if it's new.
        state_repr, belief_repr = self.repr_init(state=state, belief=belief)
        self.value_init(env_state=state, belief_state=belief)

        # Get next state.
        next_state_list = self.T(
            state_repr=state_repr,
            belief_repr=belief_repr,
            action=action,
            subtask=self.subtask,
            subtask_agent_names=self.subtask_agent_names,
        )

        expected_value = 0.0
        for p, ns, nb in next_state_list:
            # Initialize new state if it's new.
            ns_repr, nb_repr = self.repr_init(ns, nb)
            self.value_init(env_state=ns, belief_state=nb)
            expected_value += (
                p
                * value_f[
                    ((ns_repr, nb_repr), (self.subtask, self.subtask_agent_names))
                ]
            )

        return float(cost + expected_value)

    def cost(self, state, belief, action):
        """Return cost of taking action in this state."""
        cost = self.time_cost
        if isinstance(action[0], int):
            action = tuple([action])
        for a in action:
            if a != (0, 0):
                cost += self.action_cost
        return cost

    def get_expected_diff(self, start_state_repr, start_belief_repr, action):
        # Get next state.
        next_state_list = self.T(
            state_repr=start_state_repr,
            belief_repr=start_belief_repr,
            action=action,
            subtask=self.subtask,
            subtask_agent_names=self.subtask_agent_names,
        )

        b = {}

        for p, s_, b_ in next_state_list:
            # Initialize state if it's new.
            self.value_init(env_state=s_, belief_state=b_)

            s_repr, b_repr = self.repr_init(s_, b_)

            b[(s_repr, b_repr)] = p * (
                self.v_u[((s_repr, b_repr), (self.subtask, self.subtask_agent_names))]
                - self.v_l[((s_repr, b_repr), (self.subtask, self.subtask_agent_names))]
            )

        return b

    def get_next_action(self, env, belief, subtask, subtask_agent_names, ta_probs):
        """Return next action."""
        print("-------------[e2e]-----------")
        self.removed_object = None
        start_time = time.time()

        # Configure planner settings.
        self.set_settings(
            env=env,
            belief=belief,
            ta_probs=ta_probs,
            subtask=subtask,
            subtask_agent_names=subtask_agent_names,
        )

        cur_state, cur_belief = env, belief

        # BRTDP main loop.
        actions = self.get_actions(state=cur_state)
        action_index = argmin(
            [
                self.Q(state=cur_state, belief=cur_belief, action=a, value_f=self.v_l)
                for a in actions
            ]
        )
        a = actions[action_index]
        B = sum(
            self.get_expected_diff(
                cur_state.get_repr(), cur_belief.get_repr(), a
            ).values()
        )
        diff = (
            self.v_u[
                (
                    (cur_state.get_repr(), cur_belief.get_repr()),
                    (self.subtask, self.subtask_agent_names),
                )
            ]
            - self.v_l[
                (
                    (cur_state.get_repr(), cur_belief.get_repr()),
                    (self.subtask, self.subtask_agent_names),
                )
            ]
        ) / self.tau
        self.cur_state = cur_state
        self.cur_belief = cur_belief
        if B > diff:
            print("exploring, B: {}, diff: {}".format(B, diff))
            self.main()

        # Determine best action after BRTDP.
        if self.is_goal_state(cur_state, cur_belief):
            print("already at goal state, self.cur_obj_count:", self.cur_obj_count)
            return None
        else:
            actions = self.get_actions(state=cur_state)
            qvals_lower = [
                (
                    self.Q(
                        state=cur_state,
                        belief=cur_belief,
                        action=a,
                        value_f=self.v_l,
                    )
                )
                for a in actions
            ]
            qvals_upper = [
                (
                    self.Q(
                        state=cur_state,
                        belief=cur_belief,
                        action=a,
                        value_f=self.v_u,
                    )
                )
                for a in actions
            ]
            print("Upper Q Vals: ", [x for x in zip(actions, qvals_upper)])
            print("Lower Q Vals: ", [x for x in zip(actions, qvals_lower)])
            print(
                "upper is",
                self.v_u[
                    (
                        (cur_state.get_repr(), cur_belief.get_repr()),
                        (self.subtask, self.subtask_agent_names),
                    )
                ],
            )
            print(
                "lower is",
                self.v_l[
                    (
                        (cur_state.get_repr(), cur_belief.get_repr()),
                        (self.subtask, self.subtask_agent_names),
                    )
                ],
            )

            action_index = argmin(np.array(qvals_lower))
            a = actions[action_index]

            print("chose action:", a)
            print("cost:", self.cost(cur_state, cur_belief, a))
            return a

    def reset_value_caches(self, subtask):
        for state, (subtask_c, subtask_agent_names_c) in list(self.v_l.keys()):
            if subtask == subtask_c:
                del self.v_l[(state, (subtask_c, subtask_agent_names_c))]
                del self.v_u[(state, (subtask_c, subtask_agent_names_c))]
