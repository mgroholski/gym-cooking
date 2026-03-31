# Recipe planning
import copy
from collections import namedtuple
from itertools import product
from typing import Dict, List

import navigation_planner.utils as nav_utils
import numpy as np
import recipe_planner.utils as recipe_utils

# Delegation planning
from delegation_planner.bayesian_delegator import BayesianDelegator

# Navigation planner
from recipe_planner.stripsworld import STRIPSWorld
from recipe_planner.utils import *
from termcolor import colored as color

# Other core modules
from utils.core import CookingPan, Counter, Cutboard
from utils.utils import ExistenceBeliefs, agent_settings

from gym_cooking.navigation_planner.planners.e2e_b3rtdp import E2E_B3RTDP
from gym_cooking.utils.interact import interact

AgentRepr = namedtuple("AgentRepr", "name location holding")

# Colors for agents.
COLORS = ["blue", "magenta", "yellow", "green"]


class RealAgent:
    """Real Agent object that performs task inference and plans."""

    def __init__(self, arglist, name, id_color, recipes, obs):
        self.arglist = arglist
        self.name = name
        self.color = id_color
        self.recipes = recipes

        # Bayesian Delegation.
        self.reset_subtasks()
        self.new_subtask = None
        self.new_subtask_agent_names = []
        self.incomplete_subtasks = []
        self.subtask_to_wrapper_dict = {}
        self.signal_reset_delegator = False
        self.is_subtask_complete = lambda w: False
        self.beta_bd = arglist.beta_bd
        self.none_action_prob = 0.5

        self.world = copy.copy(obs.world)

        self.model_type = agent_settings(arglist, name)
        if self.model_type == "up":
            self.priors = "uniform"
        else:
            self.priors = "spatial"

        self.subtask_removed = False

        # Navigation planner.
        self.partially_observable = arglist.partially_observable
        self.planner = E2E_B3RTDP(
            D=arglist.D,
            alpha=arglist.alpha,
            epsilon=arglist.epsilon,
            beta=arglist.beta,
            tau=arglist.tau,
            depth=arglist.depth,
            main_cap=arglist.main_cap,
        )

    def __str__(self):
        return color(self.name[-1], self.color)

    def __copy__(self):
        a = Agent(
            arglist=self.arglist,
            name=self.name,
            id_color=self.color,
            recipes=self.recipes,
        )
        a.subtask = self.subtask
        a.new_subtask = self.new_subtask
        a.subtask_agent_names = self.subtask_agent_names
        a.new_subtask_agent_names = self.new_subtask_agent_names
        a.__dict__ = self.__dict__.copy()
        a.incomplete_subtasks = copy.deepcopy(self.incomplete_subtasks)
        if self.holding is not None:
            a.holding = copy.copy(self.holding)
        return a

    def get_holding(self):
        if self.holding is None:
            return "None"
        return self.holding.full_name

    def select_action(self, obs):
        """Return best next action for this agent given observations."""
        sim_agent = list(filter(lambda x: x.name == self.name, obs.sim_agents))[0]
        self.location = sim_agent.location
        self.holding = sim_agent.holding
        self.action = sim_agent.action

        if obs.t == 0:
            self.setup_subtasks(env=obs)
            self.init_beliefs(obs)

        obs.build_heuristic(self.sw.ordered_subtasks_by_recipe, self.existence_beliefs)

        if obs.t != 0:
            self.belief_update(obs=obs)

            if getattr(obs, "obs_tm1"):
                obs.obs_tm1.build_heuristic(
                    self.sw.ordered_subtasks_by_recipe, self.existence_beliefs_tm1
                )

        # Select subtask based on Bayesian Delegation.
        self.update_subtasks(env=obs)
        self.new_subtask, self.new_subtask_agent_names = self.delegator.select_subtask(
            agent_name=self.name,
        )
        self.plan(copy.copy(obs))
        return self.action

    def get_subtasks(self, recipes, world) -> Dict:
        """Return different subtask permutations for active orders."""

        active_orders = list(getattr(world, "order_queue", []))

        if not len(active_orders):
            return {}

        self.sw = STRIPSWorld(world, recipes)
        # [path for recipe 1, path for recipe 2, ...] where each path is a list of actions.
        self.subtasks_by_recipe = self.sw.get_subtasks(
            max_path_length=self.arglist.max_num_subtasks
        )

        all_subtasks = {}
        for order in active_orders:
            for subtask in self.subtasks_by_recipe[order.recipe.name]:
                if subtask in all_subtasks:
                    all_subtasks[subtask].cnt += 1
                else:
                    all_subtasks[subtask] = ActionCntWrapper(subtask)

        # Uncomment below to view graph for recipe path i
        # i = 0
        # pg = recipe_utils.make_predicate_graph(self.sw.initial, recipe_paths[i])
        # ag = recipe_utils.make_action_graph(self.sw.initial, recipe_paths[i])
        return all_subtasks

    def setup_subtasks(self, env):
        """Initializing subtasks and subtask allocator, Bayesian Delegation."""
        self.subtask_to_wrapper_dict = self.get_subtasks(env.recipes, env.world)
        self.incomplete_subtasks = [k for k in self.subtask_to_wrapper_dict.keys()]

        print(
            f"{color(self.name, self.color)} has incomplete subtasks: {self.incomplete_subtasks}"
        )

        self.delegator = BayesianDelegator(
            agent_name=self.name,
            all_agent_names=env.get_agent_names(),
            model_type=self.model_type,
            planner=self.planner,
            none_action_prob=self.none_action_prob,
        )

    def reset_subtasks(self):
        """Reset subtasks---relevant for Bayesian Delegation."""
        self.subtask = None
        self.subtask_agent_names = []
        self.subtask_complete = False

    def refresh_subtasks(self, world):
        """Refresh subtasks---relevant for Bayesian Delegation."""
        # Checks if the task queue has changed.
        if len(self.world.order_queue) != len(world.order_queue):
            """
            Get sub-tasks from new_orders then increment or add to the incomplete subtask
            list.
            """
            new_orders = world.order_queue[
                len(self.world.order_queue) + 1 : len(world.order_queue)
            ]

            for order in new_orders:
                for subtask in self.subtasks_by_recipe[order.recipe.name]:
                    if subtask in self.subtask_to_wrapper_dict:
                        self.subtask_to_wrapper_dict[subtask].cnt += 1
                    else:
                        self.subtask_to_wrapper_dict[subtask] = ActionCntWrapper(
                            subtask
                        )
                    # Adds back to incomplete subtask list if new or incremented from 0.
                    if self.subtask_to_wrapper_dict[subtask] == 1:
                        self.incomplete_subtasks.append(subtask)

        # Check whether any incomplete subtask is complete.
        self.subtask_complete = False
        if not (self.subtask is None or len(self.subtask_agent_names) == 0):
            self.subtask_complete = self.is_subtask_complete(world)
            print(
                "{} done with {} according to planner: {}\nplanner has subtask {} with subtask object {}".format(
                    color(self.name, self.color),
                    self.subtask,
                    self.is_subtask_complete(world),
                    self.planner.subtask,
                    self.planner.goal_obj,
                )
            )

            # Refresh for incomplete subtasks.
            if self.subtask_complete:
                if self.subtask in self.incomplete_subtasks:
                    self.remove_subtask(self.subtask)
                    self.subtask_complete = True

        else:
            print("{} has no subtask".format(color(self.name, self.color)))

        for incomplete_subtask in self.incomplete_subtasks:
            if self.check_incomplete_subtask(world, incomplete_subtask) and not (
                incomplete_subtask == self.subtask and self.subtask_complete
            ):
                self.remove_subtask(incomplete_subtask)
                break

        self.world = copy.copy(world)

        print(
            "{} incomplete subtasks:".format(color(self.name, self.color)),
            ", ".join(str(t) for t in self.incomplete_subtasks),
        )

    def remove_subtask(self, subtask):
        if (
            subtask in self.subtask_to_wrapper_dict
            and self.subtask_to_wrapper_dict[subtask].cnt > 1
        ):
            self.subtask_to_wrapper_dict[subtask].cnt -= 1
            self.delegator.planner.reset_value_caches(subtask)
        else:
            if subtask in self.subtask_to_wrapper_dict:
                del self.subtask_to_wrapper_dict[subtask]
            self.incomplete_subtasks.remove(subtask)

        self.subtask_removed = True

    def update_subtasks(self, env):
        """Update incomplete subtasks---relevant for Bayesian Delegation."""
        if (self.subtask_removed) or (
            self.delegator.should_reset_priors(
                obs=copy.copy(env),
                belief=self.existence_beliefs,
                incomplete_subtasks=self.incomplete_subtasks,
            )
        ):
            self.reset_subtasks()
            self.delegator.set_priors(
                obs=copy.copy(env),
                belief=self.existence_beliefs,
                incomplete_subtasks=self.incomplete_subtasks,
                priors_type=self.priors,
            )
        else:
            if self.subtask is None:
                self.delegator.set_priors(
                    obs=copy.copy(env),
                    belief=self.existence_beliefs,
                    incomplete_subtasks=self.incomplete_subtasks,
                    priors_type=self.priors,
                )
            else:
                self.delegator.bayes_update(
                    obs_tm1=copy.copy(env.obs_tm1),
                    b_tm1=copy.copy(self.existence_beliefs_tm1),
                    a_tm1=self.action,
                    beta=self.beta_bd,
                )
        self.subtask_removed = False

    def all_done(self):
        """Return whether this agent is all done.
        An agent is done if all Deliver subtasks are completed."""
        if any([isinstance(t, Deliver) for t in self.incomplete_subtasks]):
            return False
        return True

    def get_action_location(self):
        """Return location if agent takes its action---relevant for navigation planner."""
        return tuple(np.asarray(self.location) + np.asarray(self.action))

    def plan(self, env, initializing_priors=False):
        """Plan next action---relevant for navigation planner."""
        print(
            "right before planning, {} had old subtask {}, new subtask {}, subtask complete {}".format(
                self.name, self.subtask, self.new_subtask, self.subtask_complete
            )
        )

        # Check whether this subtask is done.
        if self.new_subtask is not None:
            self.def_subtask_completion(env=env)

        # If subtask is None, then do nothing.

        if (self.new_subtask is None) or (not self.new_subtask_agent_names):
            actions = nav_utils.get_single_actions(env=env, agent=self)
            probs = []
            for a in actions:
                if a == (0, 0):
                    probs.append(self.none_action_prob)
                else:
                    probs.append((1.0 - self.none_action_prob) / (len(actions) - 1))
            self.action = actions[np.random.choice(len(actions), p=probs)]
        # Otherwise, plan accordingly.
        else:
            if self.model_type == "greedy" or initializing_priors:
                other_agent_planners = {}
            else:
                backup_subtask = (
                    self.new_subtask if self.new_subtask is not None else self.subtask
                )

            print(
                "[ {} Planning ] Task: {}, Task Agents: {}".format(
                    self.name, self.new_subtask, self.new_subtask_agent_names
                )
            )

            self.action = self.planner.get_next_action(
                env=env,
                belief=self.existence_beliefs,
                subtask=self.new_subtask,
                subtask_agent_names=self.new_subtask_agent_names,
            )

        # Update subtask.
        self.subtask = self.new_subtask
        self.subtask_agent_names = self.new_subtask_agent_names
        self.new_subtask = None
        self.new_subtask_agent_names = []

        print("{} proposed action: {}\n".format(self.name, self.action))

    def check_incomplete_subtask(self, world, subtask):
        _, goal_obj = nav_utils.get_subtask_obj(subtask=subtask)
        subtask_action_object = nav_utils.get_subtask_action_obj(subtask=subtask)

        if isinstance(subtask, Deliver):
            # Checks count in the stored world.
            cur_obj_count = len(
                list(
                    filter(
                        lambda o: (
                            o
                            in set(
                                self.world.get_all_object_locs(subtask_action_object)
                            )
                        ),
                        self.world.get_object_locs(obj=goal_obj, is_held=False),
                    )
                )
            )

            # Check count in the current world.
            new_obj_count = len(
                list(
                    filter(
                        lambda o: (
                            o in set(world.get_all_object_locs(subtask_action_object))
                        ),
                        world.get_object_locs(obj=goal_obj, is_held=False),
                    )
                )
            )

            return new_obj_count > cur_obj_count
        else:
            cur_obj_cnt = len(self.world.get_all_object_locs(obj=goal_obj))
            more_cur_obj = len(world.get_all_object_locs(obj=goal_obj)) > cur_obj_cnt

            if more_cur_obj:
                print(f"{subtask} is finished. Removing {subtask}.")
            return more_cur_obj

    def def_subtask_completion(self, env):
        # Determine desired objects.
        self.start_obj, self.goal_obj = nav_utils.get_subtask_obj(
            subtask=self.new_subtask
        )
        self.subtask_action_object = nav_utils.get_subtask_action_obj(
            subtask=self.new_subtask
        )

        # Define termination conditions for agent subtask.
        # For Deliver subtask, desired object should be at a Deliver location.
        if isinstance(self.new_subtask, Deliver):
            self.cur_obj_count = len(
                list(
                    filter(
                        lambda o: (
                            o
                            in set(
                                env.world.get_all_object_locs(
                                    self.subtask_action_object
                                )
                            )
                        ),
                        env.world.get_object_locs(obj=self.goal_obj, is_held=False),
                    )
                )
            )
            self.has_more_obj = lambda x: int(x) > self.cur_obj_count
            self.is_subtask_complete = lambda w: self.has_more_obj(
                len(
                    list(
                        filter(
                            lambda o: (
                                o
                                in set(
                                    env.world.get_all_object_locs(
                                        obj=self.subtask_action_object
                                    )
                                )
                            ),
                            w.get_object_locs(obj=self.goal_obj, is_held=False),
                        )
                    )
                )
            )
        # Otherwise, for other subtasks, check based on # of objects.
        else:
            # Current count of desired objects.
            self.cur_obj_count = len(env.world.get_all_object_locs(obj=self.goal_obj))
            # Goal state is reached when the number of desired objects has increased.
            self.is_subtask_complete = lambda w: (
                len(w.get_all_object_locs(obj=self.goal_obj)) > self.cur_obj_count
            )

    def init_beliefs(self, obs):
        """
        Initializes the beliefs of the existence of all the objects
        required to complete sub-tasks for each recipe.
        """

        self.existence_beliefs = ExistenceBeliefs(
            obs, self.arglist.beta, self.subtasks_by_recipe, self.arglist.D
        )

    def belief_update(self, obs):
        """Conducts belief updates."""
        self.existence_beliefs_tm1 = copy.copy(self.existence_beliefs)

        actions = [(0, 0), (0, 1), (1, 0), (-1, 0), (0, -1)]
        null_actions = [None]
        if obs.sim_agents[0].location is None or obs.sim_agents[0].action is None:
            joint_actions = list(product(null_actions, actions))
        else:
            joint_actions = list(product(actions, null_actions))

        self.existence_beliefs.belief_update(
            obs,
            tuple([a.action for a in obs.sim_agents]),
            self.delegator.probs,
            joint_actions,
            self.delegator.planner,
        )


class SimAgent:
    """Simulation agent used in the environment object."""

    def __init__(self, name, id_color, location, observable_cols=(-1, -1)):
        self.name = name
        self.color = id_color
        self.location = location
        self.holding = None
        self.action = (0, 0)
        self.observable_cols = observable_cols

    def __str__(self):
        return color(self.name[-1], self.color)

    def __copy__(self):
        a = SimAgent(
            name=self.name,
            id_color=self.color,
            location=self.location,
            observable_cols=self.observable_cols,
        )
        a.__dict__ = self.__dict__.copy()
        if self.holding is not None:
            a.holding = copy.copy(self.holding)
        return a

    def get_repr(self):
        return AgentRepr(
            name=self.name, location=self.location, holding=self.get_holding()
        )

    def get_holding(self):
        if self.holding is None:
            return "None"
        return self.holding.full_name

    def print_status(self):
        print(
            "{} currently at {}, action {}, holding {}".format(
                color(self.name, self.color),
                self.location,
                self.action,
                self.get_holding(),
            )
        )

    def acquire(self, obj):
        if self.holding is None:
            self.holding = obj
            self.holding.is_held = True
            self.holding.location = self.location
        else:
            self.holding.merge(obj)  # Obj(1) + Obj(2) => Obj(1+2)

    def release(self):
        self.holding.is_held = False
        self.holding = None

    def move_to(self, new_location):
        self.location = new_location
        if self.holding is not None:
            self.holding.location = new_location
