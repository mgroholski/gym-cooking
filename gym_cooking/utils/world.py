import copy
from collections import OrderedDict, defaultdict, namedtuple
from functools import lru_cache
from itertools import combinations, product

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import recipe_planner.utils as recipe
from navigation_planner.utils import manhattan_dist
from utils.core import Counter, GridSquare, Object

OrderQueueRepr = namedtuple("OrderQueueRepr", "orders")


class World:
    """World class that hold all of the non-agent objects in the environment."""

    NAV_ACTIONS = [(0, 1), (0, -1), (-1, 0), (1, 0)]

    def __init__(self, arglist):
        self.rep = []  # [row0, row1, ..., rown]
        self.arglist = arglist
        self.objects = defaultdict(lambda: [])
        self.task_queue = []

    def get_repr(self):
        return self.get_dynamic_objects_repr() + self.get_task_queue_repr()

    def get_task_queue_repr(self):
        return OrderQueueRepr(orders=tuple(o.get_repr() for o in self.task_queue))

    def __str__(self):
        _display = list(map(lambda x: "".join(map(lambda y: y + " ", x)), self.rep))
        return "\n".join(_display)

    def __copy__(self):
        new = World(self.arglist)
        new.__dict__ = self.__dict__.copy()
        new.objects = copy.deepcopy(self.objects)
        new.task_queue = copy.deepcopy(self.task_queue)
        new.reachability_graph = self.reachability_graph
        new.distances = self.distances
        return new

    def update_display(self):
        # Reset the current display (self.rep).
        self.rep = [[" " for i in range(self.width)] for j in range(self.height)]
        objs = []
        for o in self.objects.values():
            objs += o
        for obj in objs:
            self.add_object(obj, obj.location)
        for obj in self.objects["Tomato"]:
            self.add_object(obj, obj.location)
        return self.rep

    def print_objects(self):
        for k, v in self.objects.items():
            print(k, list(map(lambda o: o.location, v)))

    def make_loc_to_gridsquare(self):
        """Creates a mapping between object location and object."""
        self.loc_to_gridsquare = {}
        for obj in self.get_object_list():
            if isinstance(obj, GridSquare):
                self.loc_to_gridsquare[obj.location] = obj

    def make_reachability_graph(self):
        """Create a reachability graph between world objects."""
        self.reachability_graph = nx.Graph()
        for x in range(self.width):
            for y in range(self.height):
                location = (x, y)
                gs = self.loc_to_gridsquare[(x, y)]

                # If not collidable, add node with direction (0, 0).
                if not gs.collidable:
                    self.reachability_graph.add_node((location, (0, 0)))

                # Add nodes for collidable gs + all edges.
                for nav_action in World.NAV_ACTIONS:
                    new_location = self.inbounds(
                        location=tuple(np.asarray(location) + np.asarray(nav_action))
                    )
                    new_gs = self.loc_to_gridsquare[new_location]

                    # If collidable, add edges for adjacent noncollidables.
                    if gs.collidable and not new_gs.collidable:
                        self.reachability_graph.add_node((location, nav_action))
                        if (new_location, (0, 0)) in self.reachability_graph:
                            self.reachability_graph.add_edge(
                                (location, nav_action), (new_location, (0, 0))
                            )
                    # If not collidable and new_gs collidable, add edge.
                    elif not gs.collidable and new_gs.collidable:
                        if (
                            new_location,
                            tuple(-np.asarray(nav_action)),
                        ) in self.reachability_graph:
                            self.reachability_graph.add_edge(
                                (location, (0, 0)),
                                (new_location, tuple(-np.asarray(nav_action))),
                            )
                    # If both not collidable, add direct edge.
                    elif not gs.collidable and not new_gs.collidable:
                        if (new_location, (0, 0)) in self.reachability_graph:
                            self.reachability_graph.add_edge(
                                (location, (0, 0)), (new_location, (0, 0))
                            )
                    # If both collidable, add nothing.

        # If you want to visualize this graph, uncomment below.
        # plt.figure(figsize=(10, 8))

        # G = self.reachability_graph
        # pos = nx.spring_layout(G, k=10.0, iterations=10000, seed=42)

        # nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray")

        # edge_labels = nx.get_edge_attributes(G, "label")
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        # plt.show()

    def get_dist_bound_helper(self, loc, _type):
        if _type == "lower":
            dist = self.perimeter + 1
            bound_locs = self.shared_space_locs
            for a, b in product([loc], bound_locs):
                # Other agent cannot exist at shared location so we add one.
                dist = min(dist, manhattan_dist(a, b) + 1)
        elif _type == "upper":
            dist = 0
            shared_counters = self.shared_space_locs
            bound_locs = [
                (0, 1),
                (1, 0),
                (0, self.height - 1),
                (1, self.height),
                (self.width - 1, 0),
                (self.width, 1),
                (self.width - 1, self.height),
                (self.width, self.height - 1),
            ]

            max_dist = 0
            max_shared_counter = (-1, -1)

            for bound_loc, shared_counter in product(bound_locs, shared_counters):
                d = manhattan_dist(bound_loc, shared_counter)
                if d > max_dist:
                    max_dist = d
                    max_shared_counter = shared_counter

            dist = manhattan_dist(max_shared_counter, loc) - 2 + max_dist * 2
        else:
            raise Exception(f"Invalid _type: {_type}")

        return dist

    def get_direct_dist_between(self, A_loc, B_locs):
        dist = self.perimeter + 1
        for a, b in product([A_loc], B_locs):
            dist = min(dist, manhattan_dist(a, b))

        return dist

    def clear_object(self, position):
        """Clears object @ position in self.rep and replaces it with an empty space"""
        x, y = position
        self.rep[y][x] = " "

    def clear_all(self):
        self.rep = []

    def add_object(self, object_, position):
        x, y = position
        self.rep[y][x] = str(object_)

    def insert(self, obj):
        self.objects.setdefault(obj.name, []).append(obj)

    def remove(self, obj):
        num_objs = len(self.objects[obj.name])
        index = None
        for i in range(num_objs):
            if self.objects[obj.name][i].location == obj.location:
                index = i
        assert index is not None, "Could not find {}!".format(obj.name)
        self.objects[obj.name].pop(index)
        assert len(self.objects[obj.name]) < num_objs, (
            "Nothing from {} was removed from world.objects".format(obj.name)
        )

    def get_object_list(self):
        all_obs = []
        for o in self.objects.values():
            all_obs += o
        return all_obs

    def get_dynamic_objects(self):
        objs = list()

        for key in sorted(self.objects.keys()):
            if (
                key != "Counter"
                and key != "Floor"
                and "Supply" not in key
                and key != "Delivery"
                and key != "Cutboard"
                and key != "CookingPan"
            ):
                objs.append(list(self.objects[key]))

        return objs

    def get_dynamic_objects_repr(self):
        return tuple(
            [
                tuple([obj.get_repr() for obj in objects])
                for objects in self.get_dynamic_objects()
            ]
        )

    def get_collidable_objects(self):
        return list(filter(lambda o: o.collidable, self.get_object_list()))

    def get_collidable_object_locations(self):
        return list(map(lambda o: o.location, self.get_collidable_objects()))

    def get_dynamic_object_locations(self):
        return list(map(lambda o: o.location, self.get_dynamic_objects()))

    def is_collidable(self, location):
        return location in list(
            map(
                lambda o: o.location,
                list(filter(lambda o: o.collidable, self.get_object_list())),
            )
        )

    def is_occupied(self, location):
        o = list(
            filter(
                lambda obj: (
                    obj.location == location
                    and isinstance(obj, Object)
                    and not (obj.is_held)
                ),
                self.get_object_list(),
            )
        )
        if o:
            return True
        return False

    def process_delivery(self, obj):
        matching_order_idx = next(
            (
                idx
                for idx, order in enumerate(self.task_queue)
                if order.recipe.full_state_plate_name == obj.full_name
                and not order.is_complete
            ),
            -1,
        )

        if matching_order_idx >= 0:
            completed_dish = self.task_queue[matching_order_idx]
            completed_dish.is_complete = True
            obj.is_delivered = True
            print(
                f"Delivered {obj.full_name} which was {matching_order_idx}: {completed_dish.recipe.full_state_plate_name}."
            )

            return True

        return False

    def get_object_locs(self, obj, is_held, exclude_delivered=False):
        if obj.name not in self.objects.keys():
            return []

        delivered_cond = lambda o: True
        if exclude_delivered:
            delivered_cond = lambda o: not o.is_delivered

        if isinstance(obj, Object):
            return list(
                map(
                    lambda o: o.location,
                    list(
                        filter(
                            lambda o: (
                                obj == o and o.is_held == is_held and delivered_cond(o)
                            ),
                            self.objects[obj.name],
                        )
                    ),
                )
            )
        else:
            return list(
                map(
                    lambda o: o.location,
                    list(filter(lambda o: obj == o, self.objects[obj.name])),
                )
            )

    def get_all_non_delivered_object_locs(self, obj):

        return list(
            set(
                self.get_object_locs(obj=obj, is_held=True)
                + self.get_object_locs(obj=obj, is_held=False, exclude_delivered=True)
            )
        )

    def get_all_object_locs(self, obj):
        return list(
            set(
                self.get_object_locs(obj=obj, is_held=True)
                + self.get_object_locs(obj=obj, is_held=False)
            )
        )

    def get_object_at(self, location, desired_obj, find_held_objects):
        # Map obj => location => filter by location => return that object.
        all_objs = self.get_object_list()

        if desired_obj is None:
            objs = list(
                filter(
                    lambda obj: (
                        obj.location == location
                        and isinstance(obj, Object)
                        and obj.is_held is find_held_objects
                    ),
                    all_objs,
                )
            )
        else:
            objs = list(
                filter(
                    lambda obj: (
                        obj.name == desired_obj.name
                        and obj.location == location
                        and isinstance(obj, Object)
                        and obj.is_held is find_held_objects
                    ),
                    all_objs,
                )
            )

        assert len(objs) == 1, "looking for {}, found {} at {}".format(
            desired_obj, ",".join(o.name for o in objs), location
        )

        gs = self.get_gridsquare_at(location)
        if gs.is_dispenser and not (gs.cnt is not None and gs.cnt == 1):
            obj_copy = copy.deepcopy(objs[0])
            self.insert(obj_copy)
            return obj_copy
        return objs[0]

    def get_gridsquare_at(self, location):
        gss = list(
            filter(
                lambda o: o.location == location and isinstance(o, GridSquare),
                self.get_object_list(),
            )
        )

        assert len(gss) == 1, "{} gridsquares at {}: {}".format(len(gss), location, gss)
        return gss[0]

    def inbounds(self, location):
        """Correct locaiton to be in bounds of world object."""
        x, y = location
        return min(max(x, 0), self.width - 1), min(max(y, 0), self.height - 1)
