import copy

import matplotlib.pyplot as plt

# helpers
import networkx as nx
import recipe_planner.utils as recipe_utils

# core modules
from utils.core import Object, Plate


class STRIPSWorld:
    def __init__(self, world, recipes):
        self.initial = recipe_utils.STRIPSState()
        self.recipes = recipes

        # set initial state
        self.initial.add_predicate(recipe_utils.NoPredicate())

        # We need to edit this to consider object with other predicates than fresh.
        for obj in world.get_object_list():
            if isinstance(obj, Object):
                """
                if single object then
                    if single object is plate -> Fresh(plate)
                    otherwise single object is whichever state it's it
                """
                if len(obj.contents) == 1:
                    content = obj.contents[0]
                    if isinstance(content, Plate):
                        self.initial.add_predicate(recipe_utils.Fresh(obj.name))
                    else:
                        if content.state == recipe_utils.Fresh:
                            self.initial.add_predicate(recipe_utils.Fresh(obj.name))
                        elif content.state == recipe_utils.Chopped:
                            self.initial.add_predicate(recipe_utils.Chopped(obj.name))
                        elif content.state == recipe_utils.Cooked:
                            self.initial.add_predicate(recipe_utils.Cooked(obj.name))
                else:
                    self.initial.add_predicate(recipe_utils.Merged(obj.name))

    def generate_graph(self, tasks, max_path_length):
        all_actions = set()
        for task in tasks:
            all_actions = all_actions | task.actions

        goal_state = None

        new_preds = set()
        graph = nx.DiGraph()
        graph.add_node(self.initial, obj=self.initial)
        frontier = set([self.initial])
        next_frontier = set()

        for i in range(max_path_length):
            # print("CHECKING FRONTIER #:", i)
            # if i > 5:
            #     breakpoint()
            for state in frontier:
                # for each action, check whether from this state
                for action in all_actions:
                    if action.is_valid_in(state):
                        next_state = action.get_next_from(state)
                        for p in next_state.predicates:
                            new_preds.add(str(p))
                        graph.add_node(next_state, obj=next_state)
                        graph.add_edge(state, next_state, obj=action)

                        # as soon as goal is found, break and return
                        if self.check_goal(next_state) and goal_state is None:
                            goal_state = next_state
                            return graph, goal_state

                        next_frontier.add(next_state)

            frontier = next_frontier.copy()

        if goal_state is None:
            print("goal state could not be found, try increasing --max-num-subtasks")
            import sys

            sys.exit(1)

        return graph, goal_state

    def get_subtask_cnts(self, max_path_length=100, draw_graph=False):
        graph, goal_state = self.generate_graph(self.recipes, max_path_length)

        if draw_graph:  # not recommended for path length > 4
            nx.draw(graph, with_labels=True)
            plt.show()

        all_state_paths = nx.all_shortest_paths(graph, self.initial, goal_state)

        union_action_path = set()
        for state_path in all_state_paths:
            action_path = [
                graph[state_path[i]][state_path[i + 1]]["obj"]
                for i in range(len(state_path) - 1)
            ]

            action_path_dict = {}
            for subtask in action_path:
                if subtask in action_path_dict:
                    action_path_dict[subtask].cnt += 1
                else:
                    action_path_dict[subtask] = recipe_utils.ActionCntWrapper(subtask)

            union_action_path = union_action_path | set(action_path_dict.values())

        return union_action_path

    def check_goal(self, state):
        # check if this state satisfies completion of this recipe
        state_copy = copy.deepcopy(state)
        for recipe in self.recipes:
            if not state_copy.contains(recipe.goal):
                return False
            state_copy.delete_predicate(recipe.goal)

        return True
