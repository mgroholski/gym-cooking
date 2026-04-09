import os

import numpy as np
import pygame
from misc.game.utils import *
from utils.core import *

graphics_dir = "misc/game/graphics"
_image_library = {}


def get_image(path):
    global _image_library
    image = _image_library.get(path)
    if image == None:
        canonicalized_path = path.replace("/", os.sep).replace("\\", os.sep)
        image = pygame.image.load(canonicalized_path)
        _image_library[path] = image
    return image


class Game:
    def __init__(self, world, sim_agents, play=False):
        self._running = True
        self.world = world
        self.sim_agents = sim_agents
        self.current_agent = self.sim_agents[0]
        self.play = play

        # Visual parameters
        self.scale = 80  # num pixels per tile
        self.holding_scale = 0.5
        self.container_scale = 0.7
        self.world_width = self.scale * self.world.width
        self.sidebar_width = self.scale * 3
        self.width = self.world_width + self.sidebar_width
        self.height = self.scale * self.world.height + self.scale
        self.tile_size = (self.scale, self.scale)
        self.holding_size = tuple(
            (self.holding_scale * np.asarray(self.tile_size)).astype(int)
        )
        self.container_size = tuple(
            (self.container_scale * np.asarray(self.tile_size)).astype(int)
        )
        self.holding_container_size = tuple(
            (self.container_scale * np.asarray(self.holding_size)).astype(int)
        )
        self.comm_font = None
        # self.font = pygame.font.SysFont('arialttf', 10)

    def on_init(self):
        pygame.init()
        self.comm_font = pygame.font.SysFont("arial", 14, bold=True)
        if self.play:
            self.screen = pygame.display.set_mode((self.width, self.height))
        else:
            # Create a hidden surface
            self.screen = pygame.Surface((self.width, self.height))
        self._running = True

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False

    def on_render(self):
        self.screen.fill(Color.FLOOR)
        objs = []

        # Draw gridsquares
        for o_list in self.world.objects.values():
            for o in o_list:
                if isinstance(o, GridSquare):
                    self.draw_gridsquare(o)
                elif o.is_held == False:
                    objs.append(o)

        # Draw objects not held by agents
        for o in objs:
            self.draw_object(o)

        # Draw agents and their holdings
        for agent in self.sim_agents:
            self.draw_agent(agent)

        self.draw_order_queue(self.scale * self.world.height, 0)
        self.draw_comms_sidebar()

        if self.play:
            pygame.display.flip()
            pygame.display.update()

    def draw_order_queue(self, order_row_y, x_offset=0):
        order_queue = list(getattr(self.world, "task_queue", []))
        for idx, order in enumerate(order_queue):
            order_name = order.recipe.full_state_name
            order_x = idx * self.scale + x_offset
            order_location = (order_x, order_row_y)
            fill = pygame.Rect(order_x, order_row_y, self.scale, self.scale)
            bg_color = Color.WHITE
            if order.is_complete:
                bg_color = Color.DELIVERY
            pygame.draw.rect(self.screen, bg_color, fill)
            pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)

            self.draw(order_name, self.tile_size, order_location)

    def draw_gridsquare(self, gs):
        sl = self.scaled_location(gs.location)
        fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)

        if isinstance(gs, Counter):
            pygame.draw.rect(self.screen, Color.COUNTER, fill)
            pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)

        elif isinstance(gs, Delivery):
            pygame.draw.rect(self.screen, Color.DELIVERY, fill)
            self.draw("delivery", self.tile_size, sl)

        elif isinstance(gs, Cutboard):
            pygame.draw.rect(self.screen, Color.COUNTER, fill)
            pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
            self.draw("cutboard", self.tile_size, sl)

        elif isinstance(gs, CookingPan):
            pygame.draw.rect(self.screen, Color.COUNTER, fill)
            pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
            self.draw("cookingpan", self.tile_size, sl)

        if gs.is_dispenser:
            pygame.draw.rect(self.screen, Color.DISPENSER, fill)

        return

    def draw(self, path, size, location):
        image_path = "{}/{}.png".format(graphics_dir, path)
        if os.path.exists(image_path):
            image = pygame.transform.scale(get_image(image_path), size)
            self.screen.blit(image, location)
            return

        for part in path.split("-"):
            if not part:
                continue
            part_path = "{}/{}.png".format(graphics_dir, part)
            if os.path.exists(part_path):
                image = pygame.transform.scale(get_image(part_path), size)
                self.screen.blit(image, location)

    def draw_agent(self, agent):
        self.draw(
            "agent-{}".format(agent.color),
            self.tile_size,
            self.scaled_location(agent.location),
        )
        self.draw_agent_object(agent.holding)

    def draw_comms_sidebar(self):
        if self.comm_font is None:
            return

        sidebar_x = self.world_width
        sidebar_rect = pygame.Rect(sidebar_x, 0, self.sidebar_width, self.height)
        pygame.draw.rect(self.screen, (230, 230, 230), sidebar_rect)
        pygame.draw.line(
            self.screen,
            (0, 0, 0),
            (sidebar_x, 0),
            (sidebar_x, self.height),
            1,
        )

        outer_padding = 6
        inner_padding = 4
        y = outer_padding
        max_text_width = max(
            1, self.sidebar_width - outer_padding * 2 - inner_padding * 2
        )

        for agent in self.sim_agents:
            if agent.comm is None:
                continue

            label = "{}: {}".format(agent.name, agent.comm)

            words = label.split(" ")
            lines = []
            line = ""
            for word in words:
                test_line = word if not line else line + " " + word
                if self.comm_font.size(test_line)[0] <= max_text_width:
                    line = test_line
                else:
                    if line:
                        lines.append(line)
                        line = ""
                    if self.comm_font.size(word)[0] <= max_text_width:
                        line = word
                    else:
                        chunk = ""
                        for ch in word:
                            test_chunk = chunk + ch
                            if self.comm_font.size(test_chunk)[0] <= max_text_width:
                                chunk = test_chunk
                            else:
                                if chunk:
                                    lines.append(chunk)
                                chunk = ch
                        line = chunk
            if line:
                lines.append(line)

            line_height = self.comm_font.get_linesize()
            text_h = len(lines) * line_height + inner_padding * 2
            if y + text_h > self.height - outer_padding:
                break

            bg_rect = pygame.Rect(
                sidebar_x + outer_padding,
                y,
                self.sidebar_width - outer_padding * 2,
                text_h,
            )
            pygame.draw.rect(self.screen, (0, 0, 0), bg_rect)

            for idx, l in enumerate(lines):
                text_surface = self.comm_font.render(l, True, (255, 255, 255))
                self.screen.blit(
                    text_surface,
                    (
                        sidebar_x + outer_padding + inner_padding,
                        y + inner_padding + idx * line_height,
                    ),
                )

            y += text_h + outer_padding

    def draw_agent_object(self, obj):
        # Holding shows up in bottom right corner.
        if obj is None:
            return
        if any([isinstance(c, Plate) for c in obj.contents]):
            self.draw("Plate", self.holding_size, self.holding_location(obj.location))
            if len(obj.contents) > 1:
                plate = obj.unmerge("Plate")
                self.draw(
                    obj.full_name,
                    self.holding_container_size,
                    self.holding_container_location(obj.location),
                )
                obj.merge(plate)
        else:
            self.draw(
                obj.full_name, self.holding_size, self.holding_location(obj.location)
            )

    def draw_object(self, obj):
        if obj is None:
            return
        if any([isinstance(c, Plate) for c in obj.contents]):
            self.draw("Plate", self.tile_size, self.scaled_location(obj.location))
            if len(obj.contents) > 1:
                plate = obj.unmerge("Plate")
                self.draw(
                    obj.full_name,
                    self.container_size,
                    self.container_location(obj.location),
                )
                obj.merge(plate)
        else:
            self.draw(obj.full_name, self.tile_size, self.scaled_location(obj.location))

    def scaled_location(self, loc):
        """Return top-left corner of scaled location given coordinates loc, e.g. (3, 4)"""
        return tuple(self.scale * np.asarray(loc))

    def holding_location(self, loc):
        """Return top-left corner of location where agent holding will be drawn (bottom right corner) given coordinates loc, e.g. (3, 4)"""
        scaled_loc = self.scaled_location(loc)
        return tuple(
            (np.asarray(scaled_loc) + self.scale * (1 - self.holding_scale)).astype(int)
        )

    def container_location(self, loc):
        """Return top-left corner of location where contained (i.e. plated) object will be drawn, given coordinates loc, e.g. (3, 4)"""
        scaled_loc = self.scaled_location(loc)
        return tuple(
            (
                np.asarray(scaled_loc) + self.scale * (1 - self.container_scale) / 2
            ).astype(int)
        )

    def holding_container_location(self, loc):
        """Return top-left corner of location where contained, held object will be drawn given coordinates loc, e.g. (3, 4)"""
        scaled_loc = self.scaled_location(loc)
        factor = (1 - self.holding_scale) + (
            1 - self.container_scale
        ) / 2 * self.holding_scale
        return tuple((np.asarray(scaled_loc) + self.scale * factor).astype(int))

    def on_cleanup(self):
        # pygame.display.quit()
        pygame.quit()
