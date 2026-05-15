import copy

import matplotlib.pyplot as plt

# helpers
import networkx as nx
import recipe_planner.utils as recipe_utils

# core modules
from utils.core import Object


class STRIPSWorld:
    def __init__(self, world, recipes, beliefs=None):
        self.initial = recipe_utils.STRIPSState()
        self.recipes = recipes

        # set initial state
        self.initial.add_predicate(recipe_utils.NoPredicate())

        for obj in world.get_object_list():
            if isinstance(obj, Object):
                if obj.is_delivered:
                    continue

                self.initial.add_predicate(obj.to_predicate())

        if beliefs is not None:
            existence_beliefs = beliefs.get_all_ing_existence_beliefs()
            for k, v in existence_beliefs.items():
                if v == 1.0:
                    obj = beliefs.get_name_to_obj(k)
                    self.initial.add_predicate(obj.to_predicate())

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

        visited = set([self.initial])

        for i in range(max_path_length):
            # print("CHECKING FRONTIER #:", i)
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
                        if self.check_goal(next_state, tasks) and goal_state is None:
                            goal_state = next_state
                            return graph, goal_state

                        if next_state not in visited:
                            visited.add(next_state)
                            next_frontier.add(next_state)

            frontier = next_frontier
            next_frontier = set()

        if goal_state is None:
            print("goal state could not be found, try increasing --max-num-subtasks")
            import sys

            sys.exit(1)

        return graph, goal_state

    def get_subtask_per_recipe(self, max_path_length=20, draw_graph=False):
        action_paths = []

        for recipe in self.recipes:
            graph, goal_state = self.generate_graph([recipe], max_path_length)

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
                union_action_path = union_action_path | set(action_path)
            # print('all tasks for recipe {}: {}\n'.format(recipe, ', '.join([str(a) for a in union_action_path])))
            action_paths.append(union_action_path)

        return action_paths

    def get_subtask_cnts(self, max_path_length=20, draw_graph=False):
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

    def check_goal(self, state, tasks):
        # check if this state satisfies completion of this recipe
        state_copy = copy.deepcopy(state)
        for recipe in tasks:
            if not state_copy.contains(recipe.goal):
                return False
            state_copy.delete_predicate(recipe.goal)

        return True
