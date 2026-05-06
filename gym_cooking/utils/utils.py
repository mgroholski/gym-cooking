from readline import get_current_history_length
from typing import Dict

import navigation_planner.utils as nav_utils
from recipe_planner.stripsworld import STRIPSWorld


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


def is_initial_status_ing(obj):
    return len(obj.contents) == 1 and (
        (hasattr(obj.contents[0], "state_index") and obj.contents[0].state_index == 0)
        or (obj.contents[0].name == "Plate")
    )


def init_ingredient_belief(obj, obs):
    """
    1. If object is is not in initial state, then b(x_m) = 0
    2. If object is in initial state and not reachable by agent then, b(x_m) = 1
    3. Otherwise, b(x_m) = 0.5
    """

    if not is_initial_status_ing(obj):
        return 0.0

    # Anything in the observable state is reachable by the observer.
    if not len(obs.world.get_all_object_locs(obj)):
        return 1.0

    return 0.5


def get_cnt_str(obj):
    return f"C({obj.full_name})"


def get_dispenser_str(obj):
    return f"D({obj.full_name})"


class BeliefState:
    def __init__(self, obs, max_num_subtasks):
        beliefs: Dict[str, float] = {}

        # Gets subtasks for all possible recipes.
        sw = STRIPSWorld(obs.world, obs.recipes)
        subtasks_per_recipe = sw.get_subtask_per_recipe(max_num_subtasks)

        all_subtasks = set()
        for recipe_subtask_set in subtasks_per_recipe:
            all_subtasks = all_subtasks | recipe_subtask_set

        for subtask in all_subtasks:
            start_objs, goal_obj = nav_utils.get_subtask_obj(subtask=subtask)

            if isinstance(start_objs, (list, tuple)):
                for start_obj in start_objs:
                    if is_initial_status_ing(start_obj):
                        obj_p = init_ingredient_belief(start_obj, obs)
                        beliefs[start_obj.full_name] = obj_p
                        beliefs[get_dispenser_str(start_obj)] = obj_p
                    else:
                        beliefs[start_obj.full_name] = 0.0
            else:
                if is_initial_status_ing(start_objs):
                    obj_p = init_ingredient_belief(start_objs, obs)
                    beliefs[start_objs.full_name] = obj_p
                    beliefs[get_dispenser_str(start_objs)] = obj_p
                else:
                    beliefs[start_objs.full_name] = 0.0

            beliefs[goal_obj.full_name] = init_ingredient_belief(goal_obj, obs)
            beliefs[get_cnt_str(goal_obj)] = 0.0

            action_obj = nav_utils.get_subtask_action_obj(subtask)
            if action_obj is not None:
                # At t=0, if no such object exists in the world, belief is 1.0.
                # Otherwise initialize with uncertainty 0.5.
                beliefs[action_obj.name] = (
                    1.0 if not len(obs.world.get_all_object_locs(action_obj)) else 0.5
                )

        self.beliefs: Dict[str, float] = beliefs

    def update(self, obs):
        raise NotImplementedError()

    def __copy__(self):
        raise NotImplementedError()
