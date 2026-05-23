import copy
import itertools
from collections import Counter, OrderedDict
from enum import Enum
from functools import lru_cache

import navigation_planner.utils as nav_utils
import numpy as np
import recipe_planner.utils as recipe_utils
from delegation_planner.utils import NEG_INF_LOG_VAL
from recipe_planner.recipe import Recipe
from recipe_planner.stripsworld import STRIPSWorld
from utils.core import Object, Plate
from utils.interact import interact
from utils.world import World

UNCERTAIN_INIT_PROB = 0.0


class EvidenceType(Enum):
    NONE = 0
    PICK_UP = 1
    SET_DOWN = 2
    DELIVER = 3
    COMM = 4


def is_initial_status_ing(obj):
    return len(obj.contents) == 1 and (
        (hasattr(obj.contents[0], "state_index") and obj.contents[0].state_index == 0)
        or (obj.contents[0].name == "Plate")
    )


def init_ingredient_belief(obj, obs):
    """
    1. If object is is not in initial state, then b(x_m) = 0
    2. If object is in initial state and not reachable by agent then, b(x_m) = 1
    3. Otherwise, b(x_m) = UNCERTAIN_INIT_PROB
    """

    if not is_initial_status_ing(obj):
        return 0.0

    # Anything in the observable state is reachable by the observer.
    if not len(obs.world.get_all_object_locs(obj)):
        return 1.0

    return UNCERTAIN_INIT_PROB


def get_cnt_str(obj):
    return f"C({obj.full_name})"


def get_sum_cnt_str(obj):
    cnt_str = get_cnt_str(obj)
    return f"Sum({cnt_str})"


def get_dispenser_str(obj):
    return f"D({obj.full_name})"


def get_cnt_str_from_str(s):
    return f"C({s})"


def get_sum_cnt_str_from_str(s):
    return f"Sum({s})"


def get_dispenser_str_from_str(s):
    return f"D({s})"


class BeliefState:
    def __init__(self, obs, max_num_subtasks, comm_func, init_strips=True):
        self.beliefs = OrderedDict()
        self.taken_name_to_obj = {}
        self.taken_name_cnt = {}
        self.b_tm1 = None
        self.comm_func = comm_func

        self.obs = copy.copy(obs)
        self.max_num_subtasks = max_num_subtasks

        self.initial_ing_key_set = set()
        self.ing_key_set = set()
        self.cnt_key_set = set()
        self.sum_cnt_key_set = set()

        self.name_to_obj = dict()

        self.G_x = {}

        self.cur_agent = obs.sim_agents[0]

        if init_strips:
            # Gets subtasks for all possible recipes.
            sw = STRIPSWorld(obs.world, obs.recipes)
            subtasks_per_recipe = sw.get_subtask_per_recipe(max_num_subtasks)

            for recipe_subtasks in subtasks_per_recipe:
                for subtask in self._order_recipe_subtasks(recipe_subtasks):
                    start_objs, goal_obj = nav_utils.get_subtask_obj(subtask=subtask)

                    if not isinstance(start_objs, (list, tuple)):
                        start_objs = [start_objs]

                    for start_obj in start_objs:
                        self._set_start_obj_beliefs(start_obj, obs)
                        if start_obj.full_name not in self.G_x:
                            self.G_x[start_obj.full_name] = set()
                        self.G_x[start_obj.full_name].add(goal_obj.full_name)
                        self.name_to_obj[start_obj.full_name] = start_obj

                    self.beliefs[goal_obj.full_name] = init_ingredient_belief(
                        goal_obj, obs
                    )
                    self.name_to_obj[goal_obj.full_name] = goal_obj

                    action_obj = nav_utils.get_subtask_action_obj(subtask)
                    if action_obj is not None:
                        self.beliefs[action_obj.name] = (
                            1.0
                            if not len(obs.world.get_all_object_locs(action_obj))
                            else UNCERTAIN_INIT_PROB
                        )
                        self.name_to_obj[action_obj.name] = action_obj

        # Converts all probs to log probs
        for k, v in self.beliefs.items():
            if k in self.sum_cnt_key_set:
                continue

            if v != 0:
                self.beliefs[k] = np.log(v)
            else:
                self.beliefs[k] = NEG_INF_LOG_VAL

    def get_repr(self):
        return tuple(
            [
                (self.taken_name_to_obj[k].get_repr(), v)
                for k, v in self.taken_name_cnt.items()
            ]
            + sorted(
                [
                    (k, self._get_union_prob(v))
                    for k, v in self.beliefs.items()
                    if k in self.sum_cnt_key_set
                ],
                key=lambda a: a[0],
            )
        )

    def _order_recipe_subtasks(self, subtask_set):
        goal_obj_dict = {}

        delivery_subtask = None

        for subtask in subtask_set:
            if isinstance(subtask, recipe_utils.Deliver):
                delivery_subtask = subtask
                continue

            _, goal_obj = nav_utils.get_subtask_obj(subtask=subtask)

            if goal_obj is not None:
                if goal_obj.full_name not in goal_obj_dict:
                    goal_obj_dict[goal_obj.full_name] = []
                goal_obj_dict[goal_obj.full_name].append(subtask)

        stack = [(delivery_subtask, False)]
        ordered_list = []
        visited = set()

        while stack:
            subtask, expanded = stack.pop()
            if subtask is None:
                continue

            if expanded:
                ordered_list.append(subtask)
                continue

            if subtask in visited:
                continue
            visited.add(subtask)

            stack.append((subtask, True))

            start_objs, _ = nav_utils.get_subtask_obj(subtask=subtask)
            if start_objs is None:
                continue

            if not isinstance(start_objs, (list, tuple)):
                start_objs = [start_objs]

            for start_obj in reversed(start_objs):
                for dependent_subtask in reversed(
                    goal_obj_dict.get(start_obj.full_name, [])
                ):
                    if dependent_subtask not in visited:
                        stack.append((dependent_subtask, False))

        return ordered_list[::-1]

    def _set_start_obj_beliefs(self, start_obj, obs):
        cnt_str = get_cnt_str(start_obj)
        self.beliefs[cnt_str] = 0.0
        self.cnt_key_set.add(cnt_str)

        sum_cnt_str = get_sum_cnt_str(start_obj)
        self.beliefs[sum_cnt_str] = []
        self.sum_cnt_key_set.add(sum_cnt_str)

        if is_initial_status_ing(start_obj):
            obj_p = init_ingredient_belief(start_obj, obs)
            self.beliefs[start_obj.full_name] = obj_p
            self.beliefs[get_dispenser_str(start_obj)] = obj_p
            self.initial_ing_key_set.add(start_obj.full_name)
        else:
            self.beliefs[start_obj.full_name] = 0.0
            self.ing_key_set.add(start_obj.full_name)

    def _add_to_taken(self, obj):
        obj_full_name = obj.full_name
        if obj_full_name not in self.taken_name_cnt:
            self.taken_name_cnt[obj_full_name] = 0
            self.taken_name_to_obj[obj_full_name] = obj

        self.taken_name_cnt[obj_full_name] += 1

    def _remove_from_taken(self, obj):
        obj_full_name = obj.full_name

        if obj_full_name in self.taken_name_cnt:
            self.taken_name_cnt[obj_full_name] -= 1
            if self.taken_name_cnt[obj_full_name] == 0:
                del self.taken_name_cnt[obj_full_name]
                del self.taken_name_to_obj[obj_full_name]

    def _update_beliefs_from_evidence(self, evidence_obj, obs):
        exclude_set = set()

        shortest_action_path = self.get_shortest_action_path_to(
            evidence_obj, obs.recipes
        )

        initial_ingredients_disp_keys = set()
        action_objs_names = set()

        for action in shortest_action_path:
            if isinstance(action, recipe_utils.Get):
                obj_pred = action.post_add[0]
                ingredient_name = obj_pred.args[0]
                if ingredient_name == "Plate":
                    ing = Plate()
                else:
                    ing = nav_utils.StringToObject[ingredient_name]()

                ing_obj = Object(location=(-1, -1), contents=[ing])
                initial_ingredients_disp_keys.add(get_dispenser_str(ing_obj))

            start_obj, _ = nav_utils.get_subtask_obj(action)
            if not isinstance(start_obj, (list, tuple)):
                start_obj = [start_obj]

            for obj in start_obj:
                obj_sum_str = get_sum_cnt_str(obj)
                self.beliefs[obj.full_name] = self._get_sum_cnt_log_prob(obj)
                self.beliefs[obj_sum_str] = []
                exclude_set.add(obj.full_name)
                exclude_set.add(obj_sum_str)

                # If the object was taken and used in this path
                # we infer that the agent no longer has it on their side.
                self._remove_from_taken(obj)

            action_obj = nav_utils.get_subtask_action_obj(action)
            if action_obj is not None:
                action_objs_names.add(action_obj.name)

        for ing_obj_dispenser_name in initial_ingredients_disp_keys:
            exclude_set.add(ing_obj_dispenser_name)
            self.beliefs[ing_obj_dispenser_name] = np.log(1.0)

        for action_obj_name in action_objs_names:
            exclude_set.add(action_obj_name)
            self.beliefs[action_obj_name] = np.log(1.0)

        return exclude_set

    def update(self, obs, obs_tm1, a_tm1, ta_probs, verbose=True):
        self.b_tm1 = copy.copy(self)
        self.obs = copy.copy(obs)
        evidence_type, evidence_obj = self.get_evidence(obs, obs_tm1, a_tm1)

        other_agent_comm = ""
        other_agent_name = nav_utils.get_other_agent_name(obs)
        if other_agent_name in obs.comms:
            other_agent_comm = obs.comms[other_agent_name]

        comm_evidence_type, comm_evidence_objs = self.get_comm_evidence(
            ta_probs, other_agent_name, other_agent_comm
        )

        if verbose:
            print(
                f"[Agent-{self.cur_agent} Belief Update] Evidence of type {evidence_type}."
            )

        if evidence_type != EvidenceType.NONE and verbose:
            print(
                f"[Agent-{self.cur_agent} Belief Update] Evidence object is {evidence_obj.full_name}"
            )

        exclude_set = set()
        if evidence_type == EvidenceType.PICK_UP:
            self._add_to_taken(evidence_obj)
            self.beliefs[evidence_obj.full_name] = np.log(1.0)
        elif (
            evidence_type == EvidenceType.SET_DOWN
            or evidence_type == EvidenceType.DELIVER
        ):
            exclude = self._update_beliefs_from_evidence(evidence_obj, obs)
            exclude_set = exclude | exclude_set

            evidence_obj_name = evidence_obj.full_name
            self.beliefs[evidence_obj_name] = self._get_sum_cnt_log_prob(evidence_obj)
            exclude_set.add(evidence_obj_name)

            sum_cnt_str = get_sum_cnt_str(evidence_obj)
            self.beliefs[sum_cnt_str] = []
            exclude_set.add(sum_cnt_str)

        if comm_evidence_type == EvidenceType.COMM:
            for evidence_obj in comm_evidence_objs:
                if evidence_obj is None:
                    continue

                shared_list = []
                for shared_loc in obs.world.shared_space_locs:
                    gs = obs.world.get_gridsquare_at(shared_loc)
                    if gs.holding is not None:
                        gs_holding = gs.holding
                        shared_list.append(gs.holding)
                        self._add_to_taken(gs_holding)

                exclude = self._update_beliefs_from_evidence(evidence_obj, obs)
                exclude_set = exclude | exclude_set

                for obj in shared_list:
                    self._remove_from_taken(obj)

                evidence_obj_name = evidence_obj.full_name
                self.beliefs[evidence_obj_name] = np.log(1.0)
                exclude_set.add(evidence_obj_name)

        for k, _ in self.beliefs.items():
            if k in exclude_set:
                continue

            if verbose:
                print(f"[Agent-{self.cur_agent} Belief Update] Updating {k}...")

            if k in self.ing_key_set or k in self.initial_ing_key_set:
                if k in self.initial_ing_key_set:
                    created_log_prob = self._get_log_prob_by_key(
                        get_dispenser_str_from_str(k)
                    )
                else:
                    created_log_prob = self.get_created_prob(k, ta_probs)

                exist_before_log_prob = self._get_log_prob_by_key(k)
                goal_not_created_log_prob = 0.0
                for goal_obj_str in self.G_x[k]:
                    goal_not_created_log_prob += np.log(
                        1.0 - self.get_prob_by_key(goal_obj_str)
                    )

                not_created_log_prob = np.log(1.0 - np.exp(created_log_prob))

                not_created_and_existed_and_goal_not_created_log_prob = (
                    not_created_log_prob
                    + exist_before_log_prob
                    + goal_not_created_log_prob
                )

                self.beliefs[k] = np.logaddexp(
                    not_created_and_existed_and_goal_not_created_log_prob,
                    created_log_prob,
                )

            elif k in self.cnt_key_set:
                if k[2:-1] in self.initial_ing_key_set:
                    self.beliefs[k] = self._get_log_prob_by_key(
                        get_dispenser_str_from_str(k[2:-1]),
                    )
                else:
                    self.beliefs[k] = self.get_created_prob(k[2:-1], ta_probs)
            elif k in self.sum_cnt_key_set:
                # Probability of count increasing and existing before
                timestep_prob = self._get_log_prob_by_key(
                    k[6:-2]
                ) + self._get_log_prob_by_key(k[4:-1])
                self.beliefs[k].append(timestep_prob)

        return

    def get_shortest_action_path_to(self, evidence_obj, recipes):
        taken_world = World(None)

        for k in self.taken_name_cnt.keys():
            if self.taken_name_cnt[k] > 0:
                taken_world.insert(self.taken_name_to_obj[k])

        taken_recipe = Recipe("taken-recipe")
        taken_recipe.goal = evidence_obj.to_predicate()
        for recipe in recipes:
            taken_recipe.actions = taken_recipe.actions | recipe.actions

        sw = STRIPSWorld(taken_world, [taken_recipe])
        action_path = [
            action_wrapper.action for action_wrapper in sw.get_subtask_cnts()
        ]

        return action_path

    def get_created_prob(self, key, ta_probs):
        p = NEG_INF_LOG_VAL
        for ta in ta_probs.enumerate_subtask_allocs():
            for t in ta:
                if (
                    self.cur_agent not in t.subtask_agent_names
                    and t.subtask is not None
                ):
                    # We use only other the agent because this is over S^j.
                    item_prob = []
                    is_goal_obj = False

                    start_objs, goal_obj = nav_utils.get_subtask_obj(t.subtask)
                    action_obj = nav_utils.get_subtask_action_obj(t.subtask)

                    if goal_obj.full_name == key:
                        is_goal_obj = True

                    starts = (
                        start_objs
                        if isinstance(start_objs, (list, tuple))
                        else [start_objs]
                    )
                    for start_obj in starts:
                        item_prob.append(self._get_log_prob_by_key(start_obj.full_name))

                    if action_obj is not None:
                        item_prob.append(self._get_log_prob_by_key(action_obj.name))

                    if is_goal_obj and len(item_prob) >= 2:
                        log_a = item_prob[0]
                        log_b = item_prob[1]
                        log_ta = np.log(ta_probs.get(ta))
                        p = np.logaddexp(p, log_a + log_b + log_ta)

        return p

    def get_evidence(self, obs_t, obs_tm1, a_tm1):
        sim_obs_tm1 = copy.copy(obs_tm1)
        for idx, sim_agent in enumerate(sim_obs_tm1.sim_agents):
            if sim_agent.name not in a_tm1:
                raise Exception(f"Visible agent {sim_agent.name} without an action.")
            else:
                sim_obs_tm1.sim_agents[idx].action = a_tm1[sim_agent.name]

        interact(agent=sim_obs_tm1.sim_agents[0], world=sim_obs_tm1.world)
        expected_obs_t = sim_obs_tm1

        expected_objs_list = [
            o for o_type in expected_obs_t.world.get_dynamic_objects() for o in o_type
        ]

        expected_objs_cnt = Counter(obj.full_name for obj in expected_objs_list)

        real_objs_list = [
            o for o_type in obs_t.world.get_dynamic_objects() for o in o_type
        ]

        real_objs_cnt = Counter(obj.full_name for obj in real_objs_list)

        # Set Down Evidence
        for obj in real_objs_list:
            if obj.full_name not in expected_objs_cnt:
                if obj.is_delivered:
                    # We remove is_delivered attribute so that we don't make any inferences
                    # about a delivery station in S^j when it was delivered to a visible
                    # delivery station.
                    obj = copy.copy(obj)
                    obj.is_delivered = False

                return (EvidenceType.SET_DOWN, obj)
            else:
                expected_objs_cnt[obj.full_name] -= 1
                if not expected_objs_cnt[obj.full_name]:
                    del expected_objs_cnt[obj.full_name]

        # Pick Up Evidence
        for obj in expected_objs_list:
            if obj.full_name not in real_objs_cnt:
                return (EvidenceType.PICK_UP, obj)
            else:
                real_objs_cnt[obj.full_name] -= 1
                if not real_objs_cnt[obj.full_name]:
                    del real_objs_cnt[obj.full_name]

        # Delivery Evidence
        for i in range(min(len(expected_obs_t.task_queue), len(obs_t.task_queue))):
            # When we call env.step() above, it could trigger a new order to be released
            # that wasn't realeased in obs_t and vice-versa. We don't care about this
            # and ignore this by using the minimum length queue.
            expected_task = expected_obs_t.task_queue[i]
            real_task = obs_t.task_queue[i]

            if expected_task.is_complete != real_task.is_complete:
                return (
                    EvidenceType.DELIVER,
                    nav_utils.get_predicate_obj(expected_task.recipe.goal),
                )

        # No Evidence
        return (EvidenceType.NONE, None)

    def get_comm_evidence(self, ta_probs, other_agent_name, other_agent_comm):
        if not len(other_agent_comm):
            return EvidenceType.NONE, None

        print(
            f'[Agent-{self.cur_agent}.get_comm_evidence] Getting evidence for {other_agent_name}\'s comm "{other_agent_comm}"'
        )

        ta_j = self.comm_func.get_most_probable_ta_from_comm(
            ta_probs, other_agent_name, other_agent_comm
        )

        print(
            f"[Agent-{self.cur_agent}.get_comm_evidence] {ta_j} is the most probable task allocation."
        )

        if ta_j is None:
            return EvidenceType.NONE, None

        subtask_j = None

        for t in ta_j:
            if other_agent_name in t.subtask_agent_names:
                if len(t.subtask_agent_names) != 1:
                    return EvidenceType.NONE, None

                subtask_j = t.subtask
                break

        start_obj, _ = nav_utils.get_subtask_obj(subtask_j)

        if not isinstance(start_obj, (list, tuple)):
            start_obj = [start_obj]

        print(
            f'[Agent-{self.cur_agent}.get_comm_evidence] Evidence objects are {[o.full_name if o is not None else None for o in start_obj]} for {other_agent_name}\'s comm: "{other_agent_comm}"'
        )

        return EvidenceType.COMM, start_obj

    def get_belief_prob(self):
        # P(x_0, x_1, ..., x_|b| | ...) = P(x_0|x_1,...)P(x_1|...)...
        return np.exp(sum(self.beliefs.values()))

    def get_existence_prob(self, obj):
        return self.get_prob_by_key(obj.full_name)

    def get_dispenser_prob(self, obj):
        return self.get_prob_by_key(get_dispenser_str(obj))

    def get_cnt_prob(self, obj):
        return self.get_prob_by_key(get_cnt_str(obj))

    def get_prob_by_key(self, key):
        return np.exp(self.beliefs[key])

    def __getitem__(self, key):
        return np.exp(self.beliefs[key])

    def _get_log_prob_by_key(self, key):
        return self.beliefs[key]

    def _print_values(self):
        print("Belief Values: ")
        for k, v in self.beliefs.items():
            print(f"\t{k}: {np.exp(v)}")

    def get_name_to_obj(self, name):
        if name not in self.name_to_obj:
            raise Exception(f"{name} is not in name_to_obj dict!")

        return copy.copy(self.name_to_obj[name])

    def __copy__(self):
        new_belief = BeliefState(self.obs, self.max_num_subtasks, self.comm_func, False)
        new_belief.beliefs = copy.deepcopy(self.beliefs)
        new_belief.taken_name_cnt = copy.deepcopy(self.taken_name_cnt)
        new_belief.taken_name_to_obj = copy.deepcopy(self.taken_name_to_obj)
        new_belief.initial_ing_key_set = copy.deepcopy(self.initial_ing_key_set)
        new_belief.ing_key_set = copy.deepcopy(self.ing_key_set)
        new_belief.sum_cnt_key_set = copy.deepcopy(self.sum_cnt_key_set)
        new_belief.cnt_key_set = copy.deepcopy(self.cnt_key_set)
        new_belief.name_to_obj = copy.deepcopy(self.name_to_obj)
        new_belief.G_x = copy.deepcopy(self.G_x)

        return new_belief

    def get_all_ing_existence_beliefs(self):
        existence_beliefs = {}
        for k in self.beliefs.keys():
            if "(" not in k and k not in nav_utils.ACTION_OBJECT_NAME_SET:
                existence_beliefs[k] = self.get_prob_by_key(k)

        return existence_beliefs

    def __str__(self):
        s = ""
        for k, v in self.beliefs.items():
            if k not in self.sum_cnt_key_set:
                s += f"\t{k}: {np.exp(v)}\n"
            else:
                s += f"\t{k}: {np.exp(self._get_union_log_prob(v))}\n"

        return s

    def _get_sum_cnt_log_prob(self, evidence_obj):
        prob_list = self.beliefs[get_sum_cnt_str(evidence_obj)]
        return self._get_union_log_prob(tuple(prob_list))

    def _get_union_log_prob(self, prob_tuple):
        if not prob_tuple:
            return NEG_INF_LOG_VAL

        if len(prob_tuple) == 1:
            return min(prob_tuple[0], 0.0)

        # prob_tuple elements are log-probabilities <= 0
        lp = np.asarray(prob_tuple, dtype=np.float64)

        # log(1 - exp(lp)) for each event (stable for lp <= 0)
        log_not = np.log1p(-np.exp(lp))
        log_prod_not = np.sum(log_not)

        # log(1 - exp(log_prod_not)) in a stable way
        prob = np.log1p(-np.exp(log_prod_not))

        return min(prob, 0.0)

    def reset_subtask(self, subtask):
        if isinstance(subtask, recipe_utils.Deliver):
            return

        _, goal_obj = nav_utils.get_subtask_obj(subtask)
        if goal_obj is not None:
            cnt_str = get_cnt_str(goal_obj)
            self.beliefs[cnt_str] = np.log(0.0)

            print(f"[BeliefState.reset_subtask] Resetting {cnt_str} belief to 0.0.")
