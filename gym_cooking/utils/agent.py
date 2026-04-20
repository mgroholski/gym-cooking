# Recipe planning
import copy
from collections import namedtuple
from typing import Dict

import navigation_planner.utils as nav_utils
import numpy as np
import recipe_planner.utils as recipe_utils
from communication.comm_functions import CommunicationFunctions

# Delegation planning
from delegation_planner.bayesian_delegator import BayesianDelegator

# Navigation planner
from navigation_planner.planners.e2e_brtdp import E2E_BRTDP
from recipe_planner.stripsworld import STRIPSWorld
from recipe_planner.utils import *
from termcolor import colored as color

# Other core modules
from utils.core import CookingPan, Counter, Cutboard
from utils.utils import agent_settings

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
        self.beta = arglist.beta
        self.none_action_prob = 0.5

        self.world = copy.copy(obs.world)

        self.model_type = agent_settings(arglist, name)
        if self.model_type == "up":
            self.priors = "uniform"
        else:
            self.priors = "spatial"

        self.subtask_removed = False

        # Navigation planner.
        self.planner = E2E_BRTDP(
            alpha=arglist.alpha,
            tau=arglist.tau,
            cap=arglist.cap,
            main_cap=arglist.main_cap,
            epsilon=arglist.epsilon,
            can_communicate=arglist.comm,
        )

        self.comm_func = CommunicationFunctions(self.arglist)

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

        print("===============================")
        print(f"[{self.name}.select_action] @ TIMESTEP {obs.t}")
        print("===============================")
        print(f"Pre-Update Task Allocation Probabilities:\n{str(self.delegator.probs)}")
        print(
            "Incomplete Subtasks:\n",
            "\n".join([f"\t{str(v)}" for v in self.subtask_to_wrapper_dict.values()]),
        )

        # Select subtask based on Bayesian Delegation.
        self.update_subtasks(env=obs)
        self.new_subtask, self.new_subtask_agent_names, self.new_task_alloc = (
            self.delegator.select_subtask(
                agent_name=self.name,
            )
        )

        print(
            f"Post-Update Task Allocation Probabilities:\n{str(self.delegator.probs)}"
        )
        self.plan(copy.copy(obs))

        if self.new_task_alloc is not None:
            self.task_alloc_p_tm1 = self.delegator.probs.get(self.new_task_alloc)
        else:
            self.task_alloc_p_tm1 = 1

        comm = None
        if self.action == nav_utils.COMM_ACTION:
            comm = self.generate_communication(obs, self.new_task_alloc)
        return self.action, comm

    def generate_communication(self, obs, task_alloc):
        return self.comm_func.speak(self.name, obs, task_alloc)

    def get_subtasks(self, world) -> Dict:
        """Return different subtask permutations for active orders."""

        active_tasks = list(getattr(world, "task_queue", []))

        if not len(active_tasks):
            return {}

        recipes = [o.recipe for o in active_tasks]
        self.sw = STRIPSWorld(world, recipes)
        # [path for recipe 1, path for recipe 2, ...] where each path is a list of actions.
        subtask_cnts = self.sw.get_subtask_cnts(
            max_path_length=self.arglist.max_num_subtasks
        )

        all_subtasks = {}
        for subtask_cnt in subtask_cnts:
            all_subtasks[subtask_cnt.action] = subtask_cnt

        return all_subtasks

    def setup_subtasks(self, env):
        """Initializing subtasks and subtask allocator, Bayesian Delegation."""

        self.subtask_to_wrapper_dict = self.get_subtasks(env.world)
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
            epsilon=self.arglist.epsilon,
            can_communicate=self.arglist.comm,
            comm_funcs=self.comm_func,
        )

    def reset_subtasks(self):
        """Reset subtasks---relevant for Bayesian Delegation."""
        self.subtask = None
        self.subtask_agent_names = []
        self.subtask_complete = False

    def refresh_subtasks(self, world):
        """Refresh subtasks---relevant for Bayesian Delegation."""
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
                    print(f"Non-Agent Remove: Removing {self.subtask}...")
                    self.remove_subtask(self.subtask)
                    self.subtask_complete = True

        else:
            print("{} has no subtask".format(color(self.name, self.color)))

        for incomplete_subtask in self.incomplete_subtasks:
            if (
                incomplete_subtask == self.subtask and self.subtask_complete
            ):  # If the agents have the same subtask and the current agent finish the subtask then we want to check if
                if self.check_incomplete_subtask(world, incomplete_subtask, cnt=1):
                    self.remove_subtask(incomplete_subtask)
                    print(f"Non-Agent Remove: Removing {incomplete_subtask}...")
                    break
            else:
                if self.check_incomplete_subtask(world, incomplete_subtask):
                    self.remove_subtask(incomplete_subtask)
                    print(f"Non-Agent Remove: Removing {incomplete_subtask}...")
                    break

        if self.subtask_removed or len(self.world.task_queue) != len(world.task_queue):
            self.subtask_to_wrapper_dict = self.get_subtasks(world)
            self.incomplete_subtasks = [x for x in self.subtask_to_wrapper_dict.keys()]

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
        elif subtask in self.subtask_to_wrapper_dict:
            del self.subtask_to_wrapper_dict[subtask]
            self.incomplete_subtasks.remove(subtask)
        else:
            raise Exception(f"{subtask} not in subtask_wrapper_dict")

        self.subtask_removed = True

    def update_subtasks(self, env):
        """Update incomplete subtasks---relevant for Bayesian Delegation."""
        if (self.subtask_removed) or (
            self.delegator.should_reset_priors(
                obs=copy.copy(env),
                subtask_to_wrapper_dict=self.subtask_to_wrapper_dict,
                incomplete_subtasks=self.incomplete_subtasks,
            )
        ):
            self.reset_subtasks()
            self.delegator.set_priors(
                obs=copy.copy(env),
                incomplete_subtasks=self.incomplete_subtasks,
                subtask_to_wrapper_dict=self.subtask_to_wrapper_dict,
                priors_type=self.priors,
            )
        else:
            if self.subtask is None:
                self.delegator.set_priors(
                    obs=copy.copy(env),
                    incomplete_subtasks=self.incomplete_subtasks,
                    subtask_to_wrapper_dict=self.subtask_to_wrapper_dict,
                    priors_type=self.priors,
                )
            else:
                comm_info = None
                if len(env.comms):
                    comm_info = self.comm_func.listen(
                        self.name, env.comms, self.delegator.probs
                    )

                comm_info_tm1 = None
                if len(env.obs_tm1.comms):
                    comm_info_tm1 = self.comm_func.listen(
                        self.name, env.obs_tm1.comms, self.delegator.probs
                    )

                self.delegator.bayes_update(
                    obs_tm1=copy.copy(env.obs_tm1),
                    actions_tm1=env.agent_actions,
                    comm_info_tm1=comm_info_tm1,
                    comm_info=comm_info,
                    beta=self.beta,
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
            actions = nav_utils.get_single_actions(
                env=env, agent=self, can_communicate=self.arglist.comm
            )
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
                # Determine other agent planners for level 1 planning.
                # Other agent planners are based on your planner---agents never
                # share planners.
                backup_subtask = (
                    self.new_subtask if self.new_subtask is not None else self.subtask
                )
                other_agent_planners = self.delegator.get_other_agent_planners(
                    obs=copy.copy(env),
                    backup_subtask=backup_subtask,
                )

            print(
                "[ {} Planning ] Task: {}, Task Agents: {}".format(
                    self.name, self.new_subtask, self.new_subtask_agent_names
                )
            )

            new_subtask_p = self.delegator.probs.get(self.new_task_alloc)
            action = self.planner.get_next_action(
                env=env,
                task_alloc_p=new_subtask_p,
                subtask=self.new_subtask,
                subtask_agent_names=self.new_subtask_agent_names,
                other_agent_planners=other_agent_planners,
            )

            # If joint subtask, pick your part of the simulated joint plan.
            if self.name not in self.new_subtask_agent_names and self.planner.is_joint:
                self.action = action[0]
            else:
                self.action = (
                    action[self.new_subtask_agent_names.index(self.name)]
                    if self.planner.is_joint
                    else action
                )

        # Update subtask.
        self.subtask = self.new_subtask
        self.subtask_agent_names = self.new_subtask_agent_names
        self.new_subtask = None
        self.new_task_alloc = None
        self.new_subtask_agent_names = []

        print("{} proposed action: {}\n".format(self.name, self.action))

    def check_incomplete_subtask(self, world, subtask, cnt=0):
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

            return new_obj_count > (cur_obj_count + cnt)
        else:
            cur_obj_cnt = len(self.world.get_all_object_locs(obj=goal_obj))
            return len(world.get_all_object_locs(obj=goal_obj)) > (cur_obj_cnt + cnt)

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


class SimAgent:
    """Simulation agent used in the environment object."""

    def __init__(self, name, id_color, location, observable_cols=(-1, -1)):
        self.name = name
        self.color = id_color
        self.location = location
        self.holding = None
        self.action = (0, 0)
        self.observable_cols = observable_cols
        self.comm = None

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
            name=self.name,
            location=self.location,
            holding=self.get_holding(),
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
