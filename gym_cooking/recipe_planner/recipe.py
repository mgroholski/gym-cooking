import copy
from collections import namedtuple

import recipe_planner.utils as recipe
from utils.core import *

OrderRepr = namedtuple("OrderRepr", "name is_complete")


def index_ingredient_str(ingredient_str, idx):
    if "Plate" in ingredient_str:
        ingredient_str = ingredient_str.replace("Plate", f"Plate{idx}")

    return ingredient_str


class Order:
    def __init__(self, init_recipe, idx):
        self.recipe = init_recipe
        self.idx = idx
        self.start_t = 0
        self.is_complete = False

    def get_repr(self):
        return OrderRepr(name=self.recipe.name, is_complete=self.is_complete)

    def index_recipe(self, init_recipe, idx):
        init_recipe.name = f"{init_recipe.name}{idx}"
        for action in init_recipe.actions:
            args = []
            for arg in action.args:
                args.append(index_ingredient_str(arg, idx))

            action.args = tuple(args)

            pre_l = []
            for pre in action.pre:
                args = []
                for arg in pre.args:
                    args.append(index_ingredient_str(arg, idx))
                pre.args = tuple(args)
                pre_l.append(pre)
            action.pre = pre_l

            post_l = []
            for post in action.post_add:
                args = []
                for arg in post.args:
                    args.append(index_ingredient_str(arg, idx))
                post.args = tuple(args)
                post_l.append(post)
            action.post_add = post_l
            action.set_specs()

        init_recipe.full_state_name = index_ingredient_str(
            init_recipe.full_state_name, idx
        )

        init_recipe.full_plate_name = index_ingredient_str(
            init_recipe.full_plate_name, idx
        )
        init_recipe.full_state_plate_name = index_ingredient_str(
            init_recipe.full_state_plate_name, idx
        )
        init_recipe.goal = recipe.Delivered(init_recipe.full_plate_name)
        return init_recipe


class Recipe:
    def __init__(self, name):
        self.name = name
        self.contents = []
        self.actions = set()
        self.actions.add(recipe.Get("Plate"))

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return object.__hash__(self)

    def __str__(self):
        return self.name

    def add_ingredient(self, item):
        self.contents.append(item)

        # always starts with FRESH
        self.actions.add(recipe.Get(item.name))

        if item.state_seq == FoodSequence.FRESH_CHOPPED:
            self.actions.add(recipe.Chop(item.name))
            self.actions.add(
                recipe.Merge(
                    item.name,
                    "Plate",
                    [item.state_seq[-1](item.name), recipe.Fresh("Plate")],
                    None,
                )
            )
        elif item.state_seq == FoodSequence.FRESH_COOKED:
            self.actions.add(recipe.Cook(item.name))
            self.actions.add(
                recipe.Merge(
                    item.name,
                    "Plate",
                    [item.state_seq[-1](item.name), recipe.Fresh("Plate")],
                    None,
                )
            )

    def add_goal(self):
        self.contents = sorted(
            self.contents, key=lambda x: x.name
        )  # list of Food objects
        self.contents_names = [c.name for c in self.contents]  # list of strings
        self.full_name = "-".join(sorted(self.contents_names))  # string
        self.full_plate_name = "-".join(
            sorted(self.contents_names + ["Plate"])
        )  # string

        self.state_contents_names = [(c.full_name, c.name) for c in self.contents]
        self.full_state_name = "-".join(
            [c[0] for c in sorted(self.state_contents_names, key=lambda c: c[1])]
        )  # string

        self.full_state_plate_name = "-".join(
            [
                c[0]
                for c in sorted(
                    self.state_contents_names + [("Plate", "Plate")], key=lambda c: c[1]
                )
            ]
        )  # string

        self.goal = recipe.Delivered(self.full_plate_name)
        self.actions.add(recipe.Deliver(self.full_plate_name))

    def add_merge_actions(self):
        for i in range(2, len(self.contents) + 1):
            for combo in combinations(self.contents, i):
                # Merge all with plate
                self.actions.add(
                    recipe.Merge(
                        "-".join(sorted([c.name for c in combo])),
                        "Plate",
                        [
                            recipe.Merged("-".join(sorted([c.name for c in combo]))),
                            recipe.Fresh("Plate"),
                        ],
                        None,
                    )
                )

                for new_item in combo:
                    rem = list(combo).copy()
                    rem.remove(new_item)
                    rem_str = "-".join(sorted([r.name for r in rem]))
                    plate_str = "-".join(sorted([new_item.name, "Plate"]))
                    rem_plate_str = "-".join(sorted([r.name for r in rem] + ["Plate"]))

                    if new_item.state_seq == FoodSequence.FRESH_CHOPPED:
                        new_item_pred = recipe.Chopped(new_item.name)
                    elif new_item.state_seq == FoodSequence.FRESH_COOKED:
                        new_item_pred = recipe.Cooked(new_item.name)
                    else:
                        raise Exception(
                            f"Could not find mergable predicate for {new_item.name}."
                        )

                    if len(rem) == 1:
                        if rem[0].state_seq == FoodSequence.FRESH_CHOPPED:
                            rem_pred = recipe.Chopped(rem_str)
                        elif rem[0].state_seq == FoodSequence.FRESH_COOKED:
                            rem_pred = recipe.Cooked(rem_str)
                        else:
                            raise Exception(
                                f"Could not find mergable predicate for {new_item.name}."
                            )

                        self.actions.add(
                            recipe.Merge(
                                new_item.name,
                                rem_str,
                                [new_item_pred, rem_pred],
                                None,
                            )
                        )
                        self.actions.add(
                            recipe.Merge(
                                rem_str,
                                plate_str,
                                [rem_pred, recipe.Merged(plate_str)],
                                None,
                            )
                        )
                        self.actions.add(
                            recipe.Merge(
                                new_item.name,
                                plate_str,
                                [new_item_pred, recipe.Merged(plate_str)],
                                None,
                            )
                        )
                    else:
                        self.actions.add(
                            recipe.Merge(
                                new_item.name,
                                rem_str,
                                [new_item_pred, recipe.Merged(rem_str)],
                            )
                        )
                        self.actions.add(
                            recipe.Merge(
                                plate_str,
                                rem_str,
                                [recipe.Merged(plate_str), recipe.Merged(rem_str)],
                                None,
                            )
                        )
                        self.actions.add(
                            recipe.Merge(
                                new_item.name,
                                rem_plate_str,
                                [new_item_pred, recipe.Merged(rem_plate_str)],
                            )
                        )


class SimpleTomato(Recipe):
    def __init__(self):
        Recipe.__init__(self, "Tomato")
        self.add_ingredient(Tomato(state_index=-1))
        self.add_goal()
        self.add_merge_actions()


class SimpleLettuce(Recipe):
    def __init__(self):
        Recipe.__init__(self, "Lettuce")
        self.add_ingredient(Lettuce(state_index=-1))
        self.add_goal()
        self.add_merge_actions()


class Salad(Recipe):
    def __init__(self):
        Recipe.__init__(self, "Salad")
        self.add_ingredient(Tomato(state_index=-1))
        self.add_ingredient(Lettuce(state_index=-1))
        self.add_goal()
        self.add_merge_actions()


class OnionSalad(Recipe):
    def __init__(self):
        Recipe.__init__(self, "OnionSalad")
        self.add_ingredient(Tomato(state_index=-1))
        self.add_ingredient(Lettuce(state_index=-1))
        self.add_ingredient(Onion(state_index=-1))
        self.add_goal()
        self.add_merge_actions()


class Burger(Recipe):
    def __init__(self):
        Recipe.__init__(self, "Burger")
        self.add_ingredient(Tomato(state_index=-1))
        self.add_ingredient(Lettuce(state_index=-1))
        self.add_ingredient(MeatPatty(state_index=-1))
        self.add_ingredient(Potato(state_index=-1))
        self.add_goal()
        self.add_merge_actions()
