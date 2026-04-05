import copy
import math
from typing import Dict, List, Tuple

import navigation_planner.utils as nav_utils
import numpy as np
import scipy as sp

from gym_cooking.utils.core import Food


def agent_settings(arglist, agent_name):
    if agent_name[-1] == "1":
        return arglist.model1
    elif agent_name[-1] == "2":
        return arglist.model2
    elif agent_name[-1] == "3":
        return arglist.model3
    elif agent_name[-1] == "4":
        return arglist.model4
    else:
        raise ValueError("Agent name doesn't follow the right naming, `agent-<int>`")


def init_ingredient_belief(obj, obs):
    """
    1. If object is is not in initial state, then b(x_m) = 0
    2. If object is in initial state and not reachable by agent then, b(x_m) = 1
    3. Otherwise, b(x_m) = 0.5
    """

    if len(obj.contents) > 1 or (
        hasattr(obj.contents[0], "state_index") and obj.contents[0].state_index != 0
    ):
        return 0

    # Anything in the observable state is reachable by the observer.
    if not len(obs.world.get_all_object_locs(obj)):
        return 1

    return 0.5


class ExistenceBeliefs:
    """
    Belief state over independent binary existence variables.

    Each entry in `self.existence_beliefs` is the current probability that a
    named object/state exists. Beliefs are updated independently via:

        B_t(X=1) ∝ B_{t-1}(X=1) * P(a_t | b^(X=1), H)
        B_t(X=0) ∝ B_{t-1}(X=0) * P(a_t | b^(X=0), H)

    followed by binary normalization for each state.
    """

    def __init__(self, obs, beta, subtasks_by_recipe, can_communciate, D=5):
        self.beta = float(beta)
        self.D = int(D)
        self.can_communicate = can_communciate

        beliefs: Dict[str, float] = {}

        for recipe_subtasks in subtasks_by_recipe.values():
            for subtask in recipe_subtasks:
                start_objs, goal_obj = nav_utils.get_subtask_obj(subtask=subtask)

                if isinstance(start_objs, (list, tuple)):
                    for start_obj in start_objs:
                        beliefs[start_obj.full_name] = init_ingredient_belief(
                            start_obj, obs
                        )
                else:
                    beliefs[start_objs.full_name] = init_ingredient_belief(
                        start_objs, obs
                    )

                beliefs[goal_obj.full_name] = init_ingredient_belief(goal_obj, obs)

                action_obj = nav_utils.get_subtask_action_obj(subtask)
                if action_obj is not None:
                    # At t=0, if no such object exists in the world, belief is 1.0.
                    # Otherwise initialize with uncertainty 0.5.
                    beliefs[action_obj.name] = (
                        1.0
                        if not len(obs.world.get_all_object_locs(action_obj))
                        else 0.5
                    )

        self.existence_beliefs: Dict[str, float] = beliefs
        self.states: List[str] = list(beliefs.keys())
        self._state_to_idx: Dict[str, int] = {
            state: idx for idx, state in enumerate(self.states)
        }

    def __copy__(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, memo):
        new_obj = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_obj
        new_obj.beta = self.beta
        new_obj.D = self.D
        new_obj.existence_beliefs = copy.deepcopy(self.existence_beliefs, memo)
        new_obj.states = copy.deepcopy(self.states, memo)
        new_obj._state_to_idx = copy.deepcopy(self._state_to_idx, memo)
        return new_obj

    def __getitem__(self, key):
        return self.existence_beliefs[key]

    def __setitem__(self, key, value):
        self.existence_beliefs[key] = float(value)

    def __contains__(self, key):
        return key in self.existence_beliefs

    def __repr__(self):
        return f"ExistenceBeliefs({self.existence_beliefs})"

    def __str__(self):
        s = ""
        for state in self.states:
            s += str(state) + ": " + str(self.existence_beliefs[state]) + "\n"
        return s

    def copy(self):
        return copy.deepcopy(self)

    def belief_update(
        self, obs_tm1, observed_joint_action, subtask_alloc_dist, planner
    ):
        """
        Update each binary existence belief using the observed joint action.

        Args:
            observed_joint_action:
                The joint action actually observed at time t.
            subtask_alloc_dist:
                Distribution over subtask allocations.
            Q:
                Callable with signature:
                    Q(belief, partial_action, subtask) -> float

                `belief` is passed as a numpy array in self.states order.

        Returns:
            dict[str, float]: Updated existence beliefs.
        """
        likelihood_dist_cache: Dict[str, Dict[Tuple, float]] = {}
        updated_beliefs: Dict[str, float] = {}

        for state in self.states:
            prior_true = self._clamp_prob(self.existence_beliefs[state])
            prior_false = 1.0 - prior_true

            belief_true = self._belief_array_with_clamped_state(state, 1.0)
            like_true = self._likelihood_of_observed_action(
                obs_tm1,
                belief_true,
                subtask_alloc_dist,
                observed_joint_action,
                planner,
                cache=likelihood_dist_cache,
            )

            belief_false = self._belief_array_with_clamped_state(state, 0.0)
            like_false = self._likelihood_of_observed_action(
                obs_tm1,
                belief_false,
                subtask_alloc_dist,
                observed_joint_action,
                planner,
                cache=likelihood_dist_cache,
            )

            numerator = prior_true * like_true
            denominator = numerator + (prior_false * like_false)

            if np.isclose(denominator, 0.0):
                updated_beliefs[state] = prior_true
            else:
                updated_beliefs[state] = numerator / denominator

        self.existence_beliefs.update(updated_beliefs)

    def _likelihood_of_observed_action(
        self,
        obs_tm1,
        belief_tm1,
        subtask_alloc_dist,
        observed_joint_action,
        planner,
        cache=None,
    ):
        """
        Return P(observed_joint_action | belief, H) by first constructing the
        normalized action distribution.
        """
        cache_key = (obs_tm1.get_repr(), belief_tm1.to_tuple())

        if cache is not None and cache_key in cache:
            action_dist = cache[cache_key]
        else:
            action_dist = self._likelihood_helper(
                obs_tm1=obs_tm1,
                belief_tm1=belief_tm1,
                subtask_alloc_dist=subtask_alloc_dist,
                planner=planner,
            )
            if cache is not None:
                cache[cache_key] = action_dist

        if not action_dist:
            return self._fallback_action_prob(
                obs_tm1=obs_tm1,
                observed_joint_action=observed_joint_action,
            )

        if observed_joint_action not in action_dist:
            breakpoint()

        return action_dist[observed_joint_action]

    def _fallback_action_prob(self, obs_tm1, observed_joint_action):
        """
        Fallback probability when no action distribution exists.
        Mirrors the None-subtask action selection logic from RealAgent.plan.
        """
        none_action_prob = getattr(obs_tm1, "none_action_prob", None)
        if none_action_prob is None:
            none_action_prob = 1.0

        joint_prob = 1.0
        for sim_agent, action in zip(obs_tm1.sim_agents, observed_joint_action):
            if sim_agent.location is not None:
                actions = nav_utils.get_single_actions(
                    env=obs_tm1, agent=sim_agent, can_communicate=self.can_communicate
                )
                if len(actions) <= 1:
                    action_prob = 1.0 if action == (0, 0) else 0.0
                elif action == (0, 0):
                    action_prob = none_action_prob
                else:
                    action_prob = (1.0 - none_action_prob) / (len(actions) - 1)
                joint_prob *= action_prob

        return joint_prob

    def _likelihood_helper(
        self,
        obs_tm1,
        belief_tm1,
        subtask_alloc_dist,
        planner,
    ):
        """
        P(a_t|b_t^(X)) = (Sigma_ta Prod_i exp (Beta * Q(b_t^(X), a_t))P(ta|H_{t-1})) / ( Sigma_ta Sigma a_t Prod_i exp (Beta * Q(b_t^(X), a_t))P(ta|H_{t-1}))
        P(ta|H_{t-1}) = the task allocation belief at the last timestep
        """

        total_score = 0.0
        action_prob = {}
        agent_name = obs_tm1.get_visible_agent().name
        for subtask_alloc, alloc_prob in subtask_alloc_dist.get_list():
            if alloc_prob <= 0.0:
                continue

            log_weight_by_action = {}

            agent_subtask = [t for t, n in subtask_alloc if agent_name in n][0]
            if agent_subtask is None:
                continue

            planner.set_settings(
                env=obs_tm1,
                beliefs=belief_tm1,
                subtask_alloc=subtask_alloc,
                task_alloc_probs=subtask_alloc_dist,
            )

            joint_actions = planner.get_actions(obs_tm1.get_repr())
            for joint_action in joint_actions:
                q_val = float(
                    planner.Q(
                        obs_tm1,
                        belief_tm1,
                        subtask_alloc_dist,
                        joint_action,
                        planner.v_l,
                    )
                )
                log_weight_by_action[joint_action] = (
                    log_weight_by_action.get(joint_action, 0.0) + self.beta * q_val
                )

            for joint_action, log_weight in log_weight_by_action.items():
                score = float(alloc_prob) * math.exp(log_weight)
                action_prob[joint_action] = action_prob.get(joint_action, 0.0) + score
                total_score += score

        # Normalize
        if total_score == 0:
            return action_prob

        for joint_action in action_prob:
            action_prob[joint_action] /= total_score

        return action_prob

    def _belief_array_with_clamped_state(self, state, value):
        """
        Return the current belief vector as a numpy array, with one state forced
        to the supplied value.
        """
        existence_belief = copy.deepcopy(self)
        existence_belief[state] = value
        return existence_belief

    @staticmethod
    def _clamp_prob(value):
        return min(max(float(value), 0.0), 1.0)

    def to_list(self):
        return [self.discretize()[state] for state in self.states]

    def to_tuple(self):
        return tuple(self.to_list())

    def to_np_array(self):
        return np.asarray(self.to_list(), dtype=float)

    def discretize(self):
        """
        Discretize each probability to the nearest multiple of 1 / D.
        """
        try:
            discretized = {}

            for state in self.states:
                prob = self._clamp_prob(self.existence_beliefs[state])
                discretized_prob = round(prob * self.D) / self.D
                discretized[state] = self._clamp_prob(discretized_prob)

            return discretized

        except Exception as e:
            print(e)
            breakpoint()
