# Recipe planning
import copy
from collections import namedtuple
from itertools import combinations, permutations, product

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

    def get_repr(self):
        return self.world.get_repr() + tuple(
            [agent.get_repr() for agent in self.sim_agents]
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
                    <= objs.location[1]
                    <= observable_col_rng[1]
                ]
        else:
            return env_copy

    def load_level(self, level, num_agents):
        x = 0
        y = 0

        with open("utils/levels/{}.txt".format(level), "r") as file:
            # Mark the phases of reading.
            phase = 1
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

        self.distances = {}
        self.world.width = x + 1
        self.world.height = y
        self.world.perimeter = 2 * (self.world.width + self.world.height)

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
        self.cache_distances()
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

        # Check collisions.
        self.check_collisions()
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
        if not len(self.order_queue) and not len(self.hidden_order_queue):
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

    def get_agent_names(self):
        return [agent.name for agent in self.sim_agents]

    def run_recipes(self):
        """Returns different permutations of completing recipes."""
        active_orders = self.order_queue

        if active_orders:
            recipes = list({o.recipe.name: o.recipe for o in active_orders}.values())
        else:
            return []

        self.sw = STRIPSWorld(self.world, recipes)
        # [path for recipe 1, path for recipe 2, ...] where each path is a list of actions.
        subtasks_by_recipe = self.sw.get_subtasks(
            max_path_length=self.arglist.max_num_subtasks
        )

        all_subtasks = []
        for order in active_orders:
            for subtask in subtasks_by_recipe[order.recipe.name]:
                if any(
                    st.name == subtask.name and st.args == subtask.args
                    for st in all_subtasks
                ):
                    idx = next(
                        i
                        for i, st in enumerate(all_subtasks)
                        if st.name == subtask.name and st.args == subtask.args
                    )
                    all_subtasks[idx].cnt += 1
                else:
                    all_subtasks.append(subtask)

        print("All Subtasks:", all_subtasks, "\n")

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

        self.world.order_queue = copy.deepcopy(self.order_queue)

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

        else:
            return [], []

        return A_locs, B_locs

    def get_lower_bound_for_subtask_given_objs(
        self, subtask, subtask_agent_names, start_obj, goal_obj, subtask_action_obj
    ):
        """Return the lower bound distance (shortest path) under this subtask between objects."""
        assert len(subtask_agent_names) <= 2, (
            "passed in {} agents but can only do 1 or 2".format(len(agents))
        )

        # Calculate extra holding penalty if the object is irrelevant.
        holding_penalty = 0.0
        for agent in self.sim_agents:
            if agent.name in subtask_agent_names:
                # Check for whether the agent is holding something.
                if agent.holding is not None:
                    if isinstance(subtask, recipe.Merge):
                        continue
                    else:
                        if agent.holding != start_obj and agent.holding != goal_obj:
                            # Add one "distance"-unit cost
                            holding_penalty += 1.0
        # Account for two-agents where we DON'T want to overpenalize.
        holding_penalty = min(holding_penalty, 1)

        # Get current agent locations.
        agent_locs = [
            agent.location
            for agent in list(
                filter(lambda a: a.name in subtask_agent_names, self.sim_agents)
            )
        ]
        A_locs, B_locs = self.get_AB_locs_given_objs(
            subtask=subtask,
            subtask_agent_names=subtask_agent_names,
            start_obj=start_obj,
            goal_obj=goal_obj,
            subtask_action_obj=subtask_action_obj,
        )

        # Add together distance and holding_penalty.
        return (
            self.world.get_lower_bound_between(
                subtask=subtask,
                agent_locs=tuple(agent_locs),
                A_locs=tuple(A_locs),
                B_locs=tuple(B_locs),
            )
            + holding_penalty
        )

    def is_collision(self, agent1_loc, agent2_loc, agent1_action, agent2_action):
        """Returns whether agents are colliding.

        Collisions happens if agent collide amongst themselves or with world objects."""
        # Tracks whether agents can execute their action.
        execute = [True, True]

        # Collision between agents and world objects.
        agent1_next_loc = tuple(np.asarray(agent1_loc) + np.asarray(agent1_action))
        if self.world.get_gridsquare_at(location=agent1_next_loc).collidable:
            # Revert back because agent collided.
            agent1_next_loc = agent1_loc

        agent2_next_loc = tuple(np.asarray(agent2_loc) + np.asarray(agent2_action))
        if self.world.get_gridsquare_at(location=agent2_next_loc).collidable:
            # Revert back because agent collided.
            agent2_next_loc = agent2_loc

        # Inter-agent collision.
        if agent1_next_loc == agent2_next_loc:
            if agent1_next_loc == agent1_loc and agent1_action != (0, 0):
                execute[1] = False
            elif agent2_next_loc == agent2_loc and agent2_action != (0, 0):
                execute[0] = False
            else:
                execute[0] = False
                execute[1] = False

        # Prevent agents from swapping places.
        elif (agent1_loc == agent2_next_loc) and (agent2_loc == agent1_next_loc):
            execute[0] = False
            execute[1] = False
        return execute

    def check_collisions(self):
        """Checks for collisions and corrects agents' executable actions.

        Collisions can either happen amongst agents or between agents and world objects."""
        execute = [True for _ in self.sim_agents]

        # Check each pairwise collision between agents.
        for i, j in combinations(range(len(self.sim_agents)), 2):
            agent_i, agent_j = self.sim_agents[i], self.sim_agents[j]
            exec_ = self.is_collision(
                agent1_loc=agent_i.location,
                agent2_loc=agent_j.location,
                agent1_action=agent_i.action,
                agent2_action=agent_j.action,
            )

            # Update exec array and set path to do nothing.
            if not exec_[0]:
                execute[i] = False
            if not exec_[1]:
                execute[j] = False

            # Track collisions.
            if not all(exec_):
                collision = CollisionRepr(
                    time=self.t,
                    agent_names=[agent_i.name, agent_j.name],
                    agent_locations=[agent_i.location, agent_j.location],
                )
                self.collisions.append(collision)

        print("\nexecute array is:", execute)

        # Update agents' actions if collision was detected.
        for i, agent in enumerate(self.sim_agents):
            if not execute[i]:
                agent.action = (0, 0)
            print(
                "{} has action {}".format(color(agent.name, agent.color), agent.action)
            )

    def execute_navigation(self):
        for agent in self.sim_agents:
            interact(agent=agent, world=self.world)
            self.agent_actions[agent.name] = agent.action

    def update_order_queue(self):
        if len(self.world.delivered_dishes):
            for delivered_dish in self.world.delivered_dishes:
                matching_order_idx = next(
                    (
                        idx
                        for idx, order in enumerate(self.order_queue)
                        if order.recipe.full_state_plate_name == delivered_dish
                    ),
                    -1,
                )

                if matching_order_idx >= 0:
                    removed_dish = self.order_queue.pop(matching_order_idx)
                    print(
                        f"Delivered {delivered_dish} which was {matching_order_idx}: {removed_dish.recipe.full_name}."
                    )

                    self.world.delivered_dishes = []
                    self.world.order_queue = self.order_queue

        new_order_prob = self.arglist.r
        if len(self.hidden_order_queue) and (
            (not len(self.order_queue)) or (np.random.random() < new_order_prob)
        ):
            self.add_order_to_queue()

    def cache_distances(self):
        """Saving distances between world objects."""
        counter_grid_names = [
            name
            for name in self.world.objects
            if "Supply" in name
            or "Counter" in name
            or "Delivery" in name
            or "Cut" in name
        ]
        # Getting all source objects.
        source_objs = copy.copy(self.world.objects["Floor"])
        for name in counter_grid_names:
            source_objs += copy.copy(self.world.objects[name])
        # Getting all destination objects.
        dest_objs = source_objs

        # From every source (Counter and Floor objects),
        # calculate distance to other nodes.
        for source in source_objs:
            self.distances[source.location] = {}
            # Source to source distance is 0.
            self.distances[source.location][source.location] = 0
            for destination in dest_objs:
                # Possible edges to approach source and destination.
                source_edges = [(0, 0)] if not source.collidable else World.NAV_ACTIONS
                destination_edges = (
                    [(0, 0)] if not destination.collidable else World.NAV_ACTIONS
                )
                # Maintain shortest distance.
                shortest_dist = np.inf
                for source_edge, dest_edge in product(source_edges, destination_edges):
                    try:
                        dist = nx.shortest_path_length(
                            self.world.reachability_graph,
                            (source.location, source_edge),
                            (destination.location, dest_edge),
                        )
                        # Update shortest distance.
                        if dist < shortest_dist:
                            shortest_dist = dist
                    except:
                        continue
                # Cache distance floor -> counter.
                self.distances[source.location][destination.location] = shortest_dist

        # Save all distances under world as well.
        self.world.distances = self.distances
