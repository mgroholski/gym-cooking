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
        self.h_store = {}

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
        if hasattr(self, "obs_tm1") and self.obs_tm1 is not None:
            new_env.obs_tm1 = copy.copy(self.obs_tm1)

        if hasattr(self, "ordered_subtasks_by_recipe"):
            new_env.ordered_subtasks_by_recipe = self.ordered_subtasks_by_recipe

        new_env.h_store = copy.deepcopy(self.h_store)

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
            self.arglist.queue_size,
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

            if hasattr(env_copy, "obs_tm1") and env_copy.obs_tm1 is not None:
                env_copy.obs_tm1 = env_copy.obs_tm1.get_agent_obs(agent_idx)

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

        self.h_store = {}

        # Load world & distances.
        self.load_level(level=self.arglist.level, num_agents=self.arglist.num_agents)
        self.initialize_order_queue()
        self.all_subtasks = self.run_recipes()
        self.world.make_loc_to_gridsquare()
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
        print("Environment Repr: \n", self.get_repr())
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
            if agent.location is not None:
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
        self.queue_size = int(getattr(self.arglist, "queue_size", 1))
        if self.queue_size <= 0 or not self.recipes:
            self.hidden_order_queue = []
        else:
            recipe_indices = np.random.choice(
                len(self.recipes), size=self.queue_size, replace=True
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
                all([o.is_complete for o in self.order_queue])
                or (np.random.random() < new_order_prob)
            ):
                self.add_order_to_queue()

    def get_bound_for_subtask_given_objs(
        self,
        belief,
        subtask,
        subtask_agent_names,
        start_obj,
        goal_obj,
        action_obj,
        _type,
    ):
        """Return the bound under this subtask between objects."""

        # Calculate extra holding penalty if the object is irrelevant.
        holding_penalty = 0.0
        HOLDING_PENALTY = 1.0

        for agent in self.sim_agents:
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

        visible_agents = [
            agent for agent in self.sim_agents if agent.location is not None
        ]
        assert len(visible_agents) == 1, f"Have {len(visible_agents)} visible agents."
        agent = visible_agents[0]

        holding_penalty = min(holding_penalty, HOLDING_PENALTY)
        penalty = holding_penalty

        if subtask is None:
            return self.world.perimeter + 1 + penalty

        A_locs, B_locs = self.get_AB_locs_given_objs(
            subtask, subtask_agent_names, start_obj, goal_obj, action_obj
        )

        if len(subtask_agent_names) == 1:
            # Single task
            dist = self.world.perimeter + 1
            agent_loc = agent.location
            for a_loc in A_locs:
                agent_to_a_dist = nav_utils.manhattan_dist(agent_loc, a_loc)
                if agent_to_a_dist < dist:
                    for b_loc in B_locs:
                        dist = min(
                            dist,
                            nav_utils.manhattan_dist(a_loc, b_loc) + agent_to_a_dist,
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

            return dist + penalty
        else:
            # Joint task
            # See paper
            nearest_shared_space_loc = (-1, -1)
            min_dist = float("inf")
            agent_loc = agent.location
            for loc in self.world.shared_space_locs:
                dist = nav_utils.manhattan_dist(agent_loc, loc)
                if dist < min_dist:
                    min_dist = dist
                    nearest_shared_space_loc = loc

            # Adjust to an area the agent can be
            if agent.observable_cols[0] == 0:
                nearest_shared_space_loc = (
                    nearest_shared_space_loc[0] - 1,
                    nearest_shared_space_loc[1],
                )
            else:
                nearest_shared_space_loc = (
                    nearest_shared_space_loc[0] + 1,
                    nearest_shared_space_loc[1],
                )

            if isinstance(subtask, recipe.Chop) or isinstance(subtask, recipe.Cook):
                belief_so, belief_ao = (
                    belief[start_obj.full_name],
                    belief[action_obj.name],
                )
                belief_not_so, belief_not_ao = 1 - belief_so, 1 - belief_ao

                term1 = 0
                if belief_so > 0 and belief_ao > 0:
                    term1 = (
                        belief_so
                        * belief_ao
                        * (
                            self.world.get_dist_bound_helper(agent_loc, _type)
                            + self.world.get_dist_bound_helper(
                                nearest_shared_space_loc, _type
                            )
                        )
                    )

                term2 = 0
                if belief_so > 0 and belief_not_ao > 0:
                    action_obj_locs = self.world.get_all_object_locs(action_obj)
                    term2 = (
                        belief_so
                        * belief_not_ao
                        * (
                            self.world.get_dist_bound_helper(agent_loc, _type)
                            + self.world.get_direct_dist_between(
                                nearest_shared_space_loc,
                                action_obj_locs,
                            )
                        )
                    )

                term3 = 0
                if belief_not_so > 0 and belief_ao > 0:
                    start_obj_locs = self.world.get_all_object_locs(start_obj)
                    comp_func = lambda v: max(v, term3)
                    if _type == "lower":
                        term3 = self.world.perimeter + 1
                        comp_func = lambda v: min(v, term3)

                    for start_obj_loc in start_obj_locs:
                        v = (
                            belief_not_so
                            * belief_ao
                            * (
                                self.world.get_direct_dist_between(
                                    agent_loc, [start_obj_loc]
                                )
                                + (
                                    self.world.get_dist_bound_helper(
                                        start_obj_loc, _type
                                    )
                                )  # Subtract one because we're not precisely @ start_obj_loc but rather one place inner map
                            )
                        )

                        term3 = comp_func(v)

                term4 = 0
                if belief_not_so > 0 and belief_not_ao > 0:
                    start_obj_locs = self.world.get_all_non_delivered_object_locs(
                        start_obj
                    )
                    action_obj_locs = self.world.get_all_object_locs(action_obj)

                    for start_obj_loc in start_obj_locs:
                        term4 = min(
                            term4,
                            self.world.get_direct_dist_between(
                                agent_loc, [start_obj_loc]
                            )
                            + (
                                self.world.get_direct_dist_between(
                                    start_obj_loc, action_obj_locs
                                )
                            ),  # Subtract one because we're not precisely @ start_obj_loc but rather one place inner map
                        )

                    term4 *= belief_not_so * belief_not_ao
                h = term1 + term2 + term3 + term4
            elif isinstance(subtask, recipe.Merge):
                so_1_locs, so_2_locs = (
                    self.world.get_all_non_delivered_object_locs(start_obj[0]),
                    self.world.get_all_non_delivered_object_locs(start_obj[1]),
                )

                belief_so_1, belief_so_2 = (
                    belief[start_obj[0].full_name],
                    belief[start_obj[1].full_name],
                )
                if len(so_1_locs) and len(so_2_locs):
                    h = self.world.perimeter + 1
                    for so_1_loc in so_1_locs:
                        for so_2_loc in so_2_locs:
                            h = min(
                                h,
                                (
                                    self.world.get_direct_dist_between(
                                        agent_loc, [so_1_loc]
                                    )
                                    + self.world.get_direct_dist_between(
                                        so_1_loc, [so_2_loc]
                                    )
                                ),
                                (
                                    self.world.get_direct_dist_between(
                                        agent_loc, [so_2_loc]
                                    )
                                    + self.world.get_direct_dist_between(
                                        so_2_loc, [so_1_loc]
                                    )
                                ),
                            )
                elif len(so_1_locs) > 0 and belief_so_2 == 1:
                    h = self.world.perimeter + 1
                    for so_1_loc in so_1_locs:
                        h = min(
                            h,
                            (
                                self.world.get_direct_dist_between(
                                    agent_loc, [so_1_loc]
                                )
                                + self.world.get_dist_bound_helper(so_1_loc, _type)
                            ),
                            (
                                self.world.get_dist_bound_helper(agent_loc, _type)
                                + self.world.get_direct_dist_between(
                                    nearest_shared_space_loc, [so_1_loc]
                                )
                            ),
                        )
                elif belief_so_1 == 1 and len(so_2_locs) > 0:
                    h = self.world.perimeter + 1
                    for so_2_loc in so_2_locs:
                        h = min(
                            h,
                            (
                                self.world.get_direct_dist_between(
                                    agent_loc, [so_2_loc]
                                )
                                + self.world.get_dist_bound_helper(so_2_loc, _type)
                            ),
                            (
                                self.world.get_dist_bound_helper(agent_loc, _type)
                                + self.world.get_direct_dist_between(
                                    nearest_shared_space_loc, [so_2_loc]
                                )
                            ),
                        )
                else:
                    h = self.world.perimeter + 1
            elif isinstance(subtask, recipe.Deliver):
                so_locs = self.world.get_all_non_delivered_object_locs(start_obj)
                so_belief = belief[start_obj.full_name]

                ao_locs = self.world.get_all_object_locs(action_obj)
                ao_belief = belief[action_obj.name]
                ao_not_belief = 1 - ao_belief

                if len(so_locs):
                    h = self.world.perimeter + 1
                    for so_loc in so_locs:
                        if len(ao_locs):
                            for ao_loc in ao_locs:
                                h = min(
                                    h,
                                    (
                                        ao_belief
                                        * (
                                            self.world.get_direct_dist_between(
                                                agent_loc, [so_loc]
                                            )
                                            + self.world.get_dist_bound_helper(
                                                so_loc, _type
                                            )
                                        )
                                    )
                                    + (
                                        ao_not_belief
                                        * (
                                            self.world.get_direct_dist_between(
                                                agent_loc, [so_loc]
                                            )
                                            + self.world.get_direct_dist_between(
                                                so_loc, [ao_loc]
                                            )
                                        )
                                    ),
                                )
                        else:
                            assert ao_belief == 1, "Invalid case!"
                            h = min(
                                h,
                                (
                                    ao_belief
                                    * (
                                        self.world.get_direct_dist_between(
                                            agent_loc, [so_loc]
                                        )
                                        + self.world.get_dist_bound_helper(
                                            so_loc, _type
                                        )
                                    )
                                ),
                            )
                elif so_belief == 1:
                    h = self.world.perimeter + 1
                    if len(ao_locs):
                        for ao_loc in ao_locs:
                            h = min(
                                h,
                                (
                                    ao_belief
                                    * (
                                        self.world.get_dist_bound_helper(
                                            agent_loc, _type
                                        )
                                        + self.world.get_dist_bound_helper(
                                            nearest_shared_space_loc, _type
                                        )
                                    )
                                )
                                + (
                                    ao_not_belief
                                    * (
                                        self.world.get_dist_bound_helper(
                                            agent_loc, _type
                                        )
                                        + self.world.get_direct_dist_between(
                                            nearest_shared_space_loc, [ao_loc]
                                        )
                                    )
                                ),
                            )
                    else:
                        assert ao_belief == 1, "Invalid case!"
                        h = min(
                            h,
                            (
                                ao_belief
                                * (
                                    self.world.get_dist_bound_helper(agent_loc, _type)
                                    + self.world.get_dist_bound_helper(
                                        nearest_shared_space_loc, _type
                                    )
                                )
                            ),
                        )
                else:
                    h = self.world.perimeter + 1
            elif isinstance(subtask, recipe.Get):
                h = self.world.perimeter + 1
            else:
                raise Exception(f"Unaccounted for subtask: {subtask}")
            return h + penalty

    def get_visible_agent(self):
        return [a for a in self.sim_agents if a.location is not None][0]

    def get_bound_for_subtask_alloc(
        self,
        belief,
        subtask_alloc,
        task_alloc_probs,
        _type,
    ):
        if not (
            hasattr(self, "heuristic_store")
            and (
                self.get_repr(),
                belief.to_tuple(),
            )
            in self.heuristic_store
        ):
            if not hasattr(self, "heuristic_store"):
                self.heuristic_store = {}

            h = {}

            agent_name = self.get_visible_agent().name
            for subtask_alloc in task_alloc_probs.keys:
                if task_alloc_probs.get(subtask_alloc) == 0:
                    h[tuple(subtask_alloc)] = 0

                subtask_alloc_tuple = tuple(subtask_alloc)
                h[subtask_alloc_tuple] = 0
                for (
                    subtask,
                    subtask_agent_names,
                ) in subtask_alloc:
                    # We only care if the visible agent is within the task
                    if agent_name not in subtask_agent_names:
                        continue

                    start_obj, goal_obj = nav_utils.get_subtask_obj(subtask)
                    action_obj = nav_utils.get_subtask_action_obj(subtask)
                    h[subtask_alloc_tuple] += self.get_bound_for_subtask_given_objs(
                        belief,
                        subtask,
                        subtask_agent_names,
                        start_obj,
                        goal_obj,
                        action_obj,
                        _type,
                    )

            self.heuristic_store[(self.get_repr(), belief.to_tuple())] = h

        heur = self.heuristic_store[(self.get_repr(), belief.to_tuple())]
        expected_heur = 0
        for task_alloc, p in task_alloc_probs.probs.items():
            expected_heur += p * heur[tuple(task_alloc)]

        return expected_heur

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
