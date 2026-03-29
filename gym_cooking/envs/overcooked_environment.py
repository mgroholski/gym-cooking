# Recipe planning
import copy
from collections import namedtuple
from itertools import combinations, permutations, product
from typing import List

import gym

# Navigation planning
import navigation_planner.utils as nav_utils
import networkx as nx
import numpy as np
import recipe_planner.utils as recipe

# Delegation planning
from delegation_planner.bayesian_delegator import BayesianDelegator
from gym import error, spaces, utils
from gym.utils import seeding
from misc.game.gameimage import GameImage
from pygame.transform import chop
from recipe_planner.recipe import *
from recipe_planner.stripsworld import STRIPSWorld
from utils.agent import COLORS, SimAgent
from utils.core import *

# Other core modules
from utils.interact import interact
from utils.world import World

CollisionRepr = namedtuple("CollisionRepr", "time agent_names agent_locations")


class OvercookedEnvironment(gym.Env):
    """Environment object for Overcooked."""

    def __init__(self, arglist):
        self.arglist = arglist
        self.t = 0
        self.set_filename()

        # For visualizing episode.
        self.rep = []

        # For tracking data during an episode.
        self.collisions = []
        self.termination_info = ""
        self.successful = False
        self.order_queue = []
        self.hidden_order_queue = []

        self.known_agents_names = []

        # Heuristic
        self.h_u = {}
        self.h_l = {}

    def get_repr(self):
        return self.world.get_repr() + tuple(
            [
                agent.get_repr()
                for agent in self.sim_agents
                if agent.location is not None
            ]
        )

    def __str__(self):
        # Print the world and agents.
        _display = list(map(lambda x: "".join(map(lambda y: y + " ", x)), self.rep))
        return "\n".join(_display)

    def __eq__(self, other):
        return self.get_repr() == other.get_repr()

    def __copy__(self):
        new_env = OvercookedEnvironment(self.arglist)
        new_env.__dict__ = self.__dict__.copy()
        new_env.world = copy.copy(self.world)
        new_env.sim_agents = [copy.copy(a) for a in self.sim_agents]
        new_env.distances = self.distances
        new_env.order_queue = copy.deepcopy(self.order_queue)
        new_env.hidden_order_queue = copy.deepcopy(self.hidden_order_queue)
        new_env.world.order_queue = new_env.order_queue
        new_env.h_u = self.h_u
        new_env.h_l = self.h_l

        # Make sure new objects and new agents' holdings have the right pointers.
        for a in new_env.sim_agents:
            if a.holding is not None:
                a.holding = new_env.world.get_object_at(
                    location=a.location, desired_obj=None, find_held_objects=True
                )
        return new_env

    def set_filename(self):
        self.filename = "{}_agents{}_seed{}_orders{}".format(
            self.arglist.level,
            self.arglist.num_agents,
            self.arglist.seed,
            self.arglist.order_queue_size,
        )
        model = ""
        if self.arglist.model1 is not None:
            model += "_model1-{}".format(self.arglist.model1)
        if self.arglist.model2 is not None:
            model += "_model2-{}".format(self.arglist.model2)
        if self.arglist.model3 is not None:
            model += "_model3-{}".format(self.arglist.model3)
        if self.arglist.model4 is not None:
            model += "_model4-{}".format(self.arglist.model4)
        if self.arglist.partially_observable:
            model += "_partially-observable"
        else:
            model += "_fully-observable"
        self.filename += model

    def get_agent_obs(self, agent_idx):
        env_copy = copy.copy(self)
        if self.arglist.partially_observable:
            # Obfuscates the world
            observable_col_rng = self.sim_agents[agent_idx].observable_cols

            for key, obj_list in env_copy.world.objects.items():
                env_copy.world.objects[key] = [
                    objs
                    for objs in obj_list
                    if observable_col_rng[0]
                    <= objs.location[0]
                    <= observable_col_rng[1]
                ]

            for sim_agent in env_copy.sim_agents:
                if not (
                    observable_col_rng[0]
                    <= sim_agent.location[0]
                    <= observable_col_rng[1]
                ):
                    sim_agent.action = None
                    sim_agent.location = None
                    sim_agent.holding = None

        return env_copy

    def load_level(self, level, num_agents):
        x = 0
        y = 0

        with open("utils/levels/{}.txt".format(level), "r") as file:
            # Mark the phases of reading.
            phase = 1
            agent_idx = 0
            for line in file:
                line = line.strip("\n")
                if line == "":
                    phase += 1

                # Phase 1: Read in kitchen map.
                elif phase == 1:
                    for x, rep in enumerate(line):
                        # Object, i.e. Tomato, Lettuce, Onion, Plate. Potato, MeatPatty.
                        if rep in "tlopPM":
                            counter = Counter(location=(x, y))
                            obj = Object(location=(x, y), contents=RepToClass[rep]())
                            counter.acquire(obj=obj)
                            self.world.insert(obj=counter)
                            self.world.insert(obj=obj)
                            counter.is_dispenser = True

                        # GridSquare, i.e. Floor, Counter, Cutboard, Delivery.
                        elif rep in RepToClass:
                            newobj = RepToClass[rep]((x, y))
                            self.world.objects.setdefault(newobj.name, []).append(
                                newobj
                            )
                        else:
                            # Empty. Set a Floor tile.
                            f = Floor(location=(x, y))
                            self.world.objects.setdefault("Floor", []).append(f)
                    y += 1
                # Phase 2: Read in recipe list.
                elif phase == 2:
                    self.recipes.append(globals()[line]())

                # Phase 3: Read in agent locations (up to num_agents).
                elif phase == 3:
                    if len(self.sim_agents) < num_agents:
                        loc = line.split(" ")
                        sim_agent = SimAgent(
                            name="agent-" + str(len(self.sim_agents) + 1),
                            id_color=COLORS[len(self.sim_agents)],
                            location=(int(loc[0]), int(loc[1])),
                        )
                        self.sim_agents.append(sim_agent)
                        self.known_agents_names.append(sim_agent.name)

                elif phase == 4:
                    if agent_idx < len(self.sim_agents):
                        col_rng = line.split(" ")
                        self.sim_agents[agent_idx].observable_cols = (
                            int(col_rng[0]),
                            int(col_rng[1]),
                        )
                    agent_idx += 1

        self.distances = {}
        self.world.width = x + 1
        self.world.height = y
        self.world.perimeter = 2 * (self.world.width + self.world.height)
        self.world.shared_space_locs = set()

        # Adds is_shared attribute
        for v in self.world.objects.values():
            for obj in v:
                if isinstance(obj, Counter):
                    x, y = obj.location
                    if not (
                        x == 0
                        or x == (self.world.width - 1)
                        or y == 0
                        or y == (self.world.height - 1)
                    ):
                        self.world.shared_space_locs.add((x, y))

    def reset(self):
        self.world = World(arglist=self.arglist)
        self.recipes = []
        self.sim_agents = []
        self.agent_actions = {}
        self.t = 0

        # For visualizing episode.
        self.rep = []

        # For tracking data during an episode.
        self.collisions = []
        self.termination_info = ""
        self.successful = False
        self.order_queue = []

        # Load world & distances.
        self.load_level(level=self.arglist.level, num_agents=self.arglist.num_agents)
        self.initialize_order_queue()
        self.all_subtasks = self.run_recipes()
        self.world.make_loc_to_gridsquare()
        self.world.make_reachability_graph()
        self.obs_tm1 = copy.copy(self)

        if self.arglist.record or self.arglist.with_image_obs:
            self.game = GameImage(
                filename=self.filename,
                world=self.world,
                sim_agents=self.sim_agents,
                record=self.arglist.record,
            )
            self.game.on_init()
            if self.arglist.record:
                self.game.save_image_obs(self.t)

        return copy.copy(self)

    def close(self):
        return

    def step(self, action_dict):
        # Track internal environment info.
        self.t += 1
        print("===============================")
        print("[environment.step] @ TIMESTEP {}".format(self.t))
        print("===============================")

        # Get actions.
        for sim_agent in self.sim_agents:
            sim_agent.action = action_dict[sim_agent.name]

        self.obs_tm1 = copy.copy(self)

        # Execute.
        self.execute_navigation()
        self.update_order_queue()

        # Visualize.
        self.display()
        self.print_agents()
        if self.arglist.record:
            self.game.save_image_obs(self.t)

        # Get a plan-representation observation.
        new_obs = copy.copy(self)
        # Get an image observation
        image_obs = self.game.get_image_obs()

        done = self.done()
        reward = self.reward()
        info = {
            "t": self.t,
            "obs": new_obs,
            "image_obs": image_obs,
            "done": done,
            "termination_info": self.termination_info,
        }
        return new_obs, reward, done, info

    def done(self):
        # Done if the episode maxes out
        if self.t >= self.arglist.max_num_timesteps and self.arglist.max_num_timesteps:
            self.termination_info = "Terminating because passed {} timesteps".format(
                self.arglist.max_num_timesteps
            )
            self.successful = False
            return True

        # Done if all orders in the queue have been delivered.
        if all([o.is_complete for o in self.order_queue]) and not len(
            self.hidden_order_queue
        ):
            self.termination_info = "Terminating because all orders were delivered"
            self.successful = True
            return True

        self.termination_info = ""
        self.successful = False
        return False

    def reward(self):
        return 1 if self.successful else 0

    def print_agents(self):
        for sim_agent in self.sim_agents:
            sim_agent.print_status()

    def display(self):
        self.update_display()
        print(str(self))

    def update_display(self):
        self.rep = self.world.update_display()
        for agent in self.sim_agents:
            x, y = agent.location
            self.rep[y][x] = str(agent)

    def is_collision(self, agent1_loc, agent2_loc, agent1_action, agent2_action):
        """Returns whether agents are colliding.

        Collisions happens if agent collide amongst themselves or with world objects."""
        # Tracks whether agents can execute their action.
        execute = [True, True]

        # Collision between agents and world objects.
        if agent1_loc is not None:
            agent1_next_loc = tuple(np.asarray(agent1_loc) + np.asarray(agent1_action))
            if self.world.get_gridsquare_at(location=agent1_next_loc).collidable:
                # Revert back because agent collided.
                agent1_next_loc = agent1_loc

        if agent2_loc is not None:
            agent2_next_loc = tuple(np.asarray(agent2_loc) + np.asarray(agent2_action))
            if self.world.get_gridsquare_at(location=agent2_next_loc).collidable:
                # Revert back because agent collided.
                agent2_next_loc = agent2_loc

        return execute

    def get_agent_names(self):
        return self.known_agents_names

    def run_recipes(self):
        """Returns different permutations of completing recipes."""
        active_orders = self.order_queue

        if not len(active_orders):
            return []

        self.sw = STRIPSWorld(self.world, self.recipes)
        # [path for recipe 1, path for recipe 2, ...] where each path is a list of actions.
        subtasks_by_recipe = self.sw.get_subtasks(
            max_path_length=self.arglist.max_num_subtasks
        )

        all_subtasks = {}
        for order in active_orders:
            for subtask in subtasks_by_recipe[order.recipe.name]:
                if subtask in all_subtasks:
                    all_subtasks[subtask].cnt += 1
                else:
                    all_subtasks[subtask] = recipe.ActionCntWrapper(subtask)

        print("All Subtasks:", [k for k in all_subtasks.keys()], "\n")

        return all_subtasks

    def add_order_to_queue(self):
        if len(self.hidden_order_queue):
            popped_order = self.hidden_order_queue.pop(0)
            self.order_queue.append(popped_order)
            popped_order.start_t = self.t
            print(
                f"APPENDING ORDER: Adding {popped_order.recipe.name} to order queue at timestep {self.t}."
            )

    def initialize_order_queue(self):
        self.order_queue_size = int(getattr(self.arglist, "order_queue_size", 1))
        if self.order_queue_size <= 0 or not self.recipes:
            self.hidden_order_queue = []
        else:
            recipe_indices = np.random.choice(
                len(self.recipes), size=self.order_queue_size, replace=True
            )
            self.hidden_order_queue = [
                Order(self.recipes[i], idx) for idx, i in enumerate(recipe_indices)
            ]

        print("Order Queue: ", [r.get_repr() for r in self.hidden_order_queue])

        if self.arglist.play or not self.arglist.partially_observable:
            self.order_queue = self.hidden_order_queue
            self.hidden_order_queue = []
        else:
            self.order_queue = []
            if len(self.hidden_order_queue):
                self.add_order_to_queue()

        self.world.order_queue = self.order_queue

    def execute_navigation(self):
        for agent in self.sim_agents:
            interact(agent=agent, world=self.world)
            self.agent_actions[agent.name] = agent.action

    def update_order_queue(self):
        if self.arglist.partially_observable:
            new_order_prob = self.arglist.r
            if len(self.hidden_order_queue) and (
                (not len(self.order_queue)) or (np.random.random() < new_order_prob)
            ):
                self.add_order_to_queue()

    def get_bound_for_subtask_given_objs(
        self,
        subtask,
        subtask_agent_names,
        start_obj,
        goal_obj,
        action_obj,
        _type,
    ):
        """Return the bound under this subtask between objects."""

        agent_locs = []
        null_obs_penalty = 0.0

        # Calculate extra holding penalty if the object is irrelevant.
        holding_penalty = 0.0
        HOLDING_PENALTY = 1.0
        for agent in self.sim_agents:
            if agent.location is not None:
                agent_locs.append(agent.location)

            if agent.action is None and subtask.is_joint:
                start_obj, _ = nav_utils.get_subtask_obj(subtask)
                if not isinstance(start_obj, list):
                    start_obj = [start_obj]

                locs = set()
                for obj in start_obj:
                    for loc in self.world.get_all_object_locs(obj):
                        locs.add(loc)

                if not any([loc not in self.world.shared_space_locs for loc in locs]):
                    null_obs_penalty += 1.0

            if agent.name in subtask_agent_names:
                # Check for whether the agent is holding something.
                if agent.holding is not None:
                    if isinstance(subtask, recipe.Merge):
                        if agent.holding not in start_obj and agent.holding != goal_obj:
                            holding_penalty += HOLDING_PENALTY
                    else:
                        if agent.holding != start_obj and agent.holding != goal_obj:
                            # Add one "distance"-unit cost
                            holding_penalty += HOLDING_PENALTY
        # Account for two-agents where we DON'T want to overpenalize.
        holding_penalty = min(holding_penalty, HOLDING_PENALTY)

        A_locs, B_locs = self.get_AB_locs_given_objs(
            subtask, subtask_agent_names, start_obj, goal_obj, action_obj
        )

        if len(subtask_agent_names) == 1:
            # Single task
            dist = float("inf")

            for agent_loc in agent_locs:
                for a_loc in A_locs:
                    agent_to_a_dist = nav_utils.manhattan_dist(agent_loc, a_loc)
                    if agent_to_a_dist < dist:
                        for b_loc in B_locs:
                            dist = min(
                                dist,
                                nav_utils.manhattan_dist(a_loc, b_loc)
                                + agent_to_a_dist,
                            )
                # If merge recipe, agent can pick up object B before object A
                if isinstance(subtask, recipe.Merge):
                    for b_loc in B_locs:
                        agent_to_b_dist = nav_utils.manhattan_dist(agent_loc, b_loc)
                        if agent_to_b_dist < dist:
                            for a_loc in A_locs:
                                dist = min(
                                    dist,
                                    nav_utils.manhattan_dist(a_loc, b_loc)
                                    + agent_to_b_dist,
                                )

            return dist
        else:
            # Joint task
            # h_b(x_m) (calculated in build_heuristic in world.py) + holding_penalty
            if not isinstance(start_obj, List):
                start_obj = [start_obj]

            h = 0
            for obj in start_obj:
                if _type == "lower":
                    h += self.h_l[obj.full_name]
                elif _type == "upper":
                    h += self.h_u[obj.full_name]

            if action_obj is not None:
                if _type == "lower":
                    h += self.h_l[action_obj.name]
                elif _type == "upper":
                    h += self.h_u[action_obj.name]

            return h + holding_penalty + null_obs_penalty

    def build_heuristic(self, ordered_subtasks_by_recipe, beliefs):
        """Computes the heuristics for each unique subtask."""

        self.h_u = {}
        self.h_l = {}

        agent_locs = [a.location for a in self.sim_agents if a.location is not None]
        max_bound_loc = self.world.get_dist_bound_helper(agent_locs, "upper")

        # Add plate to h_u and h_l
        plate = Plate()
        plate_obj = Object((-1, -1), [plate])

        plate_belief = beliefs[plate_obj.full_name]
        plate_locs = self.world.get_all_object_locs(plate_obj)

        expected_failure = (
            0
            if plate_belief == 1
            else (
                (1 - plate_belief)
                * self.world.get_direct_dist_between(agent_locs, plate_locs)
            )
        )

        self.h_u[plate_obj.name] = (
            plate_belief
            * self.world.get_dist_bound_between(agent_locs, plate_locs, "upper")
            + expected_failure
        )

        self.h_l[plate_obj.name] = (
            plate_belief
            * self.world.get_dist_bound_between(agent_locs, plate_locs, "lower")
            + expected_failure
        )

        for subtasks in ordered_subtasks_by_recipe.values():
            for subtask in subtasks:
                if str(subtask) == "Get(Plate)":
                    continue

                start_obj, goal_obj = nav_utils.get_subtask_obj(subtask=subtask)
                subtask_action_obj = nav_utils.get_subtask_action_obj(subtask=subtask)

                goal_obj_belief = beliefs[goal_obj.full_name]
                goal_obj_locs = self.world.get_all_object_locs(goal_obj)

                if subtask_action_obj is not None:
                    action_obj_belief = beliefs[subtask_action_obj.name]
                    subtask_action_obj_locs = self.world.get_all_object_locs(
                        subtask_action_obj
                    )

                    expected_failure = (
                        0
                        if action_obj_belief == 1
                        else (
                            (1 - action_obj_belief)
                            * self.world.get_direct_dist_between(
                                agent_locs, subtask_action_obj_locs
                            )
                        )
                    )

                    self.h_u[subtask_action_obj.name] = (
                        action_obj_belief
                        * self.world.get_dist_bound_between(
                            agent_locs, subtask_action_obj_locs, "upper"
                        )
                        + expected_failure
                    )

                    self.h_l[subtask_action_obj.name] = (
                        action_obj_belief
                        * self.world.get_dist_bound_between(
                            agent_locs, subtask_action_obj_locs, "lower"
                        )
                        + expected_failure
                    )

                if not isinstance(subtask, recipe.Merge) and not isinstance(
                    subtask, recipe.Deliver
                ):
                    if len(start_obj.contents) == 1:
                        if start_obj.contents[0].state_index == 0:
                            start_obj_locs = self.world.get_all_object_locs(start_obj)
                            start_obj_belief = beliefs[start_obj.full_name]

                            expected_failure = (
                                0
                                if start_obj_belief == 1
                                else (
                                    (1 - start_obj_belief)
                                    * self.world.get_direct_dist_between(
                                        agent_locs, start_obj_locs
                                    )
                                )
                            )

                            self.h_u[start_obj.full_name] = (
                                start_obj_belief
                                * self.world.get_dist_bound_between(
                                    agent_locs, start_obj_locs, "upper"
                                )
                                + expected_failure
                            )

                            self.h_l[start_obj.full_name] = (
                                start_obj_belief
                                * self.world.get_dist_bound_between(
                                    agent_locs, start_obj_locs, "lower"
                                )
                                + expected_failure
                            )

                        d_fail_u = 0
                        d_fail_l = 0

                        # Calculates d_fail
                        start_obj_prev_state = copy.copy(start_obj)
                        start_obj_prev_state.contents[0].state_index = (
                            start_obj.contents[0].state_index - 1
                        )
                        start_obj_prev_state.contents[0].update_names()
                        start_obj_prev_state.update_names()

                        d_fail_u = (
                            self.h_l[start_obj_prev_state.full_name] + max_bound_loc
                        )  # Furthest away after completing failure
                        d_fail_l = self.h_l[
                            start_obj_prev_state.full_name
                        ]  # Agent is holding it after completing failure

                        if subtask_action_obj is not None:
                            d_fail_u += self.h_u[subtask_action_obj.name]
                            d_fail_u += self.h_u[subtask_action_obj.name]

                        self.h_u[goal_obj.full_name] = (
                            goal_obj_belief
                            * self.world.get_dist_bound_between(
                                agent_locs, goal_obj_locs, "upper"
                            )
                            + (1 - goal_obj_belief) * d_fail_u
                        )

                        self.h_l[goal_obj.full_name] = (
                            goal_obj_belief
                            * self.world.get_dist_bound_between(
                                agent_locs, goal_obj_locs, "lower"
                            )
                            + (1 - goal_obj_belief) * d_fail_l
                        )
                    else:
                        raise Exception(
                            f"Non-Merge or Delivered task a multiple contents {subtask} and {start_obj.full_name}"
                        )
                else:
                    # Merge subtask and Deliver Subtask
                    if isinstance(subtask, recipe.Merge):
                        start_objs = start_obj
                    else:
                        start_objs = [start_obj]

                    d_fail_l = 0
                    d_fail_u = 0
                    for start_obj in start_objs:
                        """
                        If the object doesn't exist then we get the heuristic for
                        the object minus everyone of it's contents
                        """
                        start_obj_locs = self.world.get_all_object_locs(start_obj)
                        start_obj_belief = beliefs[start_obj.full_name]

                        min_d_l = float("inf")
                        min_d_u = float("inf")

                        start_obj_copy = copy.copy(start_obj)
                        for content in start_obj.contents:
                            start_obj_copy.contents.remove(content)
                            start_obj_copy.update_names()

                            d_metric_l = 0
                            d_metric_u = 0
                            if start_obj_copy.full_name != "":
                                if (
                                    start_obj_copy.full_name not in self.h_l
                                    or start_obj_copy.full_name not in self.h_u
                                ):
                                    continue

                                d_metric_l += self.h_l[start_obj_copy.full_name]
                                d_metric_u += self.h_u[start_obj_copy.full_name]

                            content_obj = Object((-1, -1), content)

                            d_metric_l += self.h_l[content_obj.full_name]
                            d_metric_u += self.h_u[content_obj.full_name]

                            min_d_l = min(d_metric_l, min_d_l)
                            min_d_u = min(min_d_u, d_metric_u)
                            start_obj_copy.contents.append(content)
                            del content_obj
                        d_fail_l += min_d_l
                        d_fail_u += min_d_u
                        d_fail_u += (
                            max_bound_loc  # Furthest away after completing failure
                        )

                    self.h_u[goal_obj.full_name] = (
                        goal_obj_belief
                        * self.world.get_dist_bound_between(
                            agent_locs, goal_obj_locs, "upper"
                        )
                        + (1 - goal_obj_belief) * d_fail_u
                    )

                    self.h_l[goal_obj.full_name] = (
                        goal_obj_belief
                        * self.world.get_dist_bound_between(
                            agent_locs, start_obj_locs, "lower"
                        )
                        + (1 - goal_obj_belief) * d_fail_l
                    )

    def get_AB_locs_given_objs(
        self, subtask, subtask_agent_names, start_obj, goal_obj, subtask_action_obj
    ):
        """Returns list of locations relevant for subtask's Merge operator.

        See Merge operator formalism in our paper, under Fig. 11:
        https://arxiv.org/pdf/2003.11778.pdf"""

        # For Merge operator on Chop subtasks, we look at objects that can be
        # chopped and the cutting board objects.
        if isinstance(subtask, recipe.Chop):
            # A: Object that can be chopped.
            A_locs = self.world.get_object_locs(obj=start_obj, is_held=False) + list(
                map(
                    lambda a: a.location,
                    list(
                        filter(
                            lambda a: (
                                a.name in subtask_agent_names and a.holding == start_obj
                            ),
                            self.sim_agents,
                        )
                    ),
                )
            )

            # B: Cutboard objects.
            B_locs = self.world.get_all_object_locs(obj=subtask_action_obj)

        # For Merge operator on Cook subtasks, we look at objects that can be
        # cooked and the cooking objects.
        elif isinstance(subtask, recipe.Cook):
            # A: Object that can be cooked.
            A_locs = self.world.get_object_locs(obj=start_obj, is_held=False) + list(
                map(
                    lambda a: a.location,
                    list(
                        filter(
                            lambda a: (
                                a.name in subtask_agent_names and a.holding == start_obj
                            ),
                            self.sim_agents,
                        )
                    ),
                )
            )

            # B: Cook objects.
            B_locs = self.world.get_all_object_locs(obj=subtask_action_obj)

        # For Merge operator on Deliver subtasks, we look at objects that can be
        # delivered and the Delivery object.
        elif isinstance(subtask, recipe.Deliver):
            # B: Delivery objects.
            B_locs = self.world.get_all_object_locs(obj=subtask_action_obj)

            # A: Object that can be delivered.
            A_locs = self.world.get_object_locs(obj=start_obj, is_held=False) + list(
                map(
                    lambda a: a.location,
                    list(
                        filter(
                            lambda a: (
                                a.name in subtask_agent_names and a.holding == start_obj
                            ),
                            self.sim_agents,
                        )
                    ),
                )
            )
            A_locs = list(filter(lambda a: a not in B_locs, A_locs))

        # For Merge operator on Merge subtasks, we look at objects that can be
        # combined together. These objects are all ingredient objects (e.g. Tomato, Lettuce).
        elif isinstance(subtask, recipe.Merge):
            A_locs = self.world.get_object_locs(obj=start_obj[0], is_held=False) + list(
                map(
                    lambda a: a.location,
                    list(
                        filter(
                            lambda a: (
                                a.name in subtask_agent_names
                                and a.holding == start_obj[0]
                            ),
                            self.sim_agents,
                        )
                    ),
                )
            )
            B_locs = self.world.get_object_locs(obj=start_obj[1], is_held=False) + list(
                map(
                    lambda a: a.location,
                    list(
                        filter(
                            lambda a: (
                                a.name in subtask_agent_names
                                and a.holding == start_obj[1]
                            ),
                            self.sim_agents,
                        )
                    ),
                )
            )
        elif isinstance(subtask, recipe.Get):
            A_locs = self.world.get_object_locs(obj=start_obj, is_held=False)
            B_locs = []
        else:
            return [], []

        return A_locs, B_locs
