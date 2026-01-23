import recipe_planner.utils as recipe
from utils.core import *


class Recipe:
    def __init__(self, name):
        self.name = name
        self.contents = []
        self.actions = set()
        self.actions.add(recipe.Get("Plate"))

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
        self.goal = recipe.Delivered(self.full_plate_name)
        self.actions.add(recipe.Deliver(self.full_plate_name))

    def add_merge_actions(self):
        """
        TODO

        We'll want to figure out how to fix this. The merge action should take either
        1. Cooked food and Merged Item
        2. Chopped food and Merged Item

        We'll need a better method than just look at the argument name since there's combinations
        e.g. (Lettuce-MeatPatty)

        We add a merge through
        `self.actions.add()`

        The Merge action has a init interface
        `def __init__(self, arg1, arg2, pre=None, post_add=None)`

        We can access the ingredient through self.contents.

        We want to be able to merge any order
        s.t. Merge(Tomato, Merge(Chopped(Lettuce), Plate)) = Lettuce-Tomato-Plate = Merge(Lettuce, Merge(Chopped(Tomato), Plate))

        Link to original file:
            https://github.com/mgroholski/gym-cooking/blob/529b4cada8392190bac4c7b60c6aaa7471875b87/gym_cooking/recipe_planner/recipe.py#L34
        """
        # should be general enough for any kind of salad / raw plated veggies

        # alphabetical, joined by dashes ex. Ingredient1-Ingredient2-Plate
        # self.full_name = '-'.join(sorted(self.contents + ['Plate']))

        # for any plural number of ingredients

        for i in range(1, len(self.contents) + 1):
            for combo in combinations(self.contents, i):
                for idx, new_item in enumerate(combo):
                    new_item_pred = None

                    if new_item.state_seq == FoodSequence.FRESH_CHOPPED:
                        new_item_pred = recipe.Chopped(new_item.name)
                    elif new_item.state_seq == FoodSequence.FRESH_COOKED:
                        new_item_pred = recipe.Cooked(new_item.name)
                    else:
                        raise Exception(
                            f"Could not find mergable predicate for {new_item.name}."
                        )

                    if len(combo) == 1:
                        self.actions.add(
                            recipe.Merge(
                                new_item.name,
                                "Plate",
                                [new_item_pred, recipe.Fresh("Plate")],
                                None,
                            )
                        )
                    else:
                        other_ingredients = sorted(
                            [
                                ingredient.name
                                for ingredient in combo[0:idx] + combo[idx + 1 :]
                            ]
                            + ["Plate"]
                        )

                        other_ingredients_name = "-".join(other_ingredients)

                        self.actions.add(
                            recipe.Merge(
                                new_item.name,
                                other_ingredients_name,
                                [new_item_pred, recipe.Merged(other_ingredients_name)],
                                None,
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
