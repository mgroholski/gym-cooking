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
