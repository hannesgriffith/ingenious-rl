from random import randint

import pygame as pg
from pygame.locals import *

from ui.coords import UICoords
from utils.loader import ImageLoader
from game.tiles import flip_tile

# background_colour = [100, 200, 100]
# background_colour = [240, 200, 240]
# background_colour = [randint(0, 255), randint(0, 255), randint(0, 255)]

def generate_background_colour():
    threshold = 10
    range_ = [150, 200]
    i, j, k = randint(*range_), randint(*range_), randint(*range_)
    while abs(i - j) < threshold:
        j = randint(*range_)
    while abs(i - k) < threshold or abs(j - k) < threshold:
        k = randint(*range_)
    return [i, j, k]

background_colour = generate_background_colour()

deck_i_colour = (240, 240, 240)
deck1_i_colour = (240, 240, 240)
deck2_i_colour = (240, 240, 240)
deck_o_colour = (100, 100, 100)
deck1_o_colour = (192, 0, 0)
deck2_o_colour = (0, 112, 192)
choice_i_colour = (220, 220, 240)
choice_o_colour = (170, 170, 190)

class Display:
    def __init__(self, screen, params):
        self.coords = UICoords()
        self.images = ImageLoader()
        self.screen = screen
        self.last_move = None

        self.game_type = params["game_type"]
        self.player1_name = params[1]["name"]
        self.player2_name = params[2]["name"]

        self.player_type = {}
        self.player_type[1] = params[1]["player_type"]
        self.player_type[2] = params[2]["player_type"]

        self.deck = {}
        self.deck[1] = [None] * 6
        self.deck[2] = [None] * 6

        self.show_decks = {}
        self.show_decks[1] = params[1]["show_deck"]
        self.show_decks[2] = params[2]["show_deck"]

        self.colour_image_map = self.images.get_colour_image_map()

        self.hex_map = self.coords.get_hex_coord_map()
        self.tile_map = self.coords.get_tile_coord_map()
        self.choice_map = self.coords.get_choice_map()
        self.choice_to_col_map = self.coords.get_choice_coords_to_col_map()
        self.col_to_choice_coords = self.coords.get_col_to_choice_coords_map()

        self.deck_maps = {}
        self.deck_maps[1] = self.coords.get_deck_map(self.coords.deck1_start_x)
        self.deck_maps[2] = self.coords.get_deck_map(self.coords.deck2_start_x)

        self.screen.fill(background_colour)
        pg.draw.rect(self.screen, deck1_o_colour,self.coords.deck1o_coords)
        pg.draw.rect(self.screen, deck1_i_colour, self.coords.deck1i_coords)
        pg.draw.rect(self.screen, deck2_o_colour, self.coords.deck2o_coords)
        pg.draw.rect(self.screen, deck2_i_colour, self.coords.deck2i_coords)
        pg.draw.rect(self.screen, choice_o_colour, self.coords.choice1o_coords)
        pg.draw.rect(self.screen, choice_i_colour, self.coords.choice1i_coords)

        self.eg_map = self.coords.get_eg_map()
        self.font = pg.font.SysFont("Comic Sans MS", 24)

        self.display_text("1: "+self.player1_name,
            (self.coords.scores_coords[1][0] + 5,
                self.coords.scores_coords[1][1] - 20))
        self.display_text("2: "+self.player2_name,
            (self.coords.scores_coords[2][0] + 5,
                self.coords.scores_coords[2][1] - 20))

        self.draw_new()

        self.confirm_rect = pg.Rect(self.coords.confirm[0],
            self.coords.confirm[1], 140, 30)
        self.cancel_rect = pg.Rect(self.coords.cancel[0],
            self.coords.cancel[1], 140, 30)

        self.hex_rects = {}
        for idx_coords, screen_coords in self.tile_map.items():
            rect = pg.Rect(screen_coords[0], screen_coords[1], 35, 37)
            self.hex_rects[idx_coords] = rect

        self.eg_rects = {}
        for idx_coords, screen_coords in self.choice_map.items():
            rect = pg.Rect(screen_coords[0], screen_coords[1], 35, 37)
            self.eg_rects[idx_coords] = rect

    def draw_new(self):
        self.screen.blit(self.images.ingenious, (10, 10))
        self.screen.blit(self.images.empty_box, (10, 80))
        self.screen.blit(self.images.scores, self.coords.scores_coords[1])
        self.screen.blit(self.images.scores, self.coords.scores_coords[2])

        for coords in self.hex_map.values():
            self.screen.blit(self.images.light_hex, coords)

        for ij in self.coords.get_start_hexes():
            self.screen.blit(self.images.dark_hex, self.hex_map[ij])

        self.screen.blit(self.images.r, self.tile_map[(0, 0)])
        self.screen.blit(self.images.p, self.tile_map[(0, 5)])
        self.screen.blit(self.images.y, self.tile_map[(0, 10)])
        self.screen.blit(self.images.g, self.tile_map[(5, 0)])
        self.screen.blit(self.images.o, self.tile_map[(5, 10)])
        self.screen.blit(self.images.b, self.tile_map[(10, 5)])

        for coords in self.deck_maps[1].values():
            self.screen.blit(self.images.dark_hex, coords)

        for coords in self.deck_maps[2].values():
            self.screen.blit(self.images.dark_hex, coords)

        self.draw_new_choice_map()

        self.screen.blit(self.images.dark_hex, self.eg_map[0])
        self.screen.blit(self.images.dark_hex, self.eg_map[1])

    def draw_new_choice_map(self):
        for coords in self.choice_map.values():
            self.screen.blit(self.images.light_hex, coords)

        self.screen.blit(self.images.r,
            self.coords.add_offset(self.choice_map[(0, 0)]))
        self.screen.blit(self.images.p,
            self.coords.add_offset(self.choice_map[(1, 0)]))
        self.screen.blit(self.images.y,
            self.coords.add_offset(self.choice_map[(0, 1)]))
        self.screen.blit(self.images.g,
            self.coords.add_offset(self.choice_map[(1, 1)]))
        self.screen.blit(self.images.o,
            self.coords.add_offset(self.choice_map[(0, 2)]))
        self.screen.blit(self.images.b,
            self.coords.add_offset(self.choice_map[(1, 2)]))

    def display_text(self, text, coords):
        label = self.font.render(text, 1, (0, 0, 0))
        self.screen.blit(label, coords)

    def clear_message_box(self):
        self.screen.blit(self.images.empty_box, (10, 80))

    def display_message_line1(self, message):
        label = self.font.render(message, 1, (0, 0, 0))
        self.screen.blit(label, (20, 95))

    def display_message_line2(self, message):
        label = self.font.render(message, 1, (0, 0, 0))
        self.screen.blit(label, (20, 120))

    def display_message_line3(self, message):
        label = self.font.render(message, 1, (0, 0, 0))
        self.screen.blit(label, (20, 145))

    def display_messages(self, line1=None, line2=None, line3=None):
        self.clear_message_box()
        if line1 is not None:
            self.display_message_line1(line1)
        if line2 is not None:
            self.display_message_line2(line2)
        if line3 is not None:
            self.display_message_line3(line3)
        pg.display.flip()

    def draw_eg_1(self, colour):
        self.screen.blit(self.images.colours[colour],
            self.coords.add_offset(self.eg_map[0]))

    def draw_eg_2(self, colour):
        self.screen.blit(self.images.colours[colour],
            self.coords.add_offset(self.eg_map[1]))

    def reset_eg(self):
        self.screen.blit(self.images.dark_hex, self.eg_map[0])
        self.screen.blit(self.images.dark_hex, self.eg_map[1])

    def add_tile_to_deck(self, deck_num, tile):
        deck_idx = self.deck[deck_num].index(None)
        self.deck[deck_num][deck_idx] = tile
        if self.show_decks[deck_num]:
            self.screen.blit(self.images.colours[tile[0]],
                self.coords.add_offset(self.deck_maps[deck_num][(deck_idx, 0)]))
            self.screen.blit(self.images.colours[tile[1]],
                self.coords.add_offset(self.deck_maps[deck_num][(deck_idx, 1)]))
            pg.display.flip()

    def remove_tile_from_deck(self, deck_num, tile):
        if self.game_type == "real" and self.player_type[deck_num] == "human":
            return None # When we don't know real player's deck

        if tile in self.deck[deck_num]:
            deck_idx = self.deck[deck_num].index(tile)
        else:
            deck_idx = self.deck[deck_num].index(flip_tile(tile))
        self.deck[deck_num][deck_idx] = None
        if self.show_decks[deck_num]:
            self.screen.blit(self.images.dark_hex,
                self.deck_maps[deck_num][deck_idx, 0])
            self.screen.blit(self.images.dark_hex,
                self.deck_maps[deck_num][deck_idx, 1])
            pg.display.flip()

    def tile_is_in_deck(self, deck_num, tile):
        if self.game_type == "real": # We don't know real player's deck
            return True
        if tile in self.deck[deck_num] or flip_tile(tile) in self.deck[deck_num]:
            return True
        else:
            return False

    def draw_hex_dark(self, coord_idx):
        self.screen.blit(self.images.dark_hex, self.hex_map[coord_idx])

    def draw_hex_light(self, coord_idx):
        self.screen.blit(self.images.light_hex, self.hex_map[coord_idx])

    def draw_hex_select_blue_dark(self, coord_idx):
        self.screen.blit(self.images.select_blue_dark, self.hex_map[coord_idx])

    def draw_hex_select_blue_light(self, coord_idx):
        self.screen.blit(self.images.select_blue_light, self.hex_map[coord_idx])

    def draw_hex_select_red_dark(self, coord_idx):
        self.screen.blit(self.images.select_red_dark, self.hex_map[coord_idx])

    def draw_hex_select_red_light(self, coord_idx):
        self.screen.blit(self.images.select_red_light, self.hex_map[coord_idx])

    def draw_hex_tile(self, coord_idx, colour):
        self.screen.blit(self.images.colours[colour], self.tile_map[coord_idx])

    def draw_confirm_cancel(self):
        self.screen.blit(self.images.confirm, self.coords.confirm)
        self.screen.blit(self.images.cancel, self.coords.cancel)

    def hide_confirm_cancel(self):
        self.screen.blit(self.images.empty_button, self.coords.confirm)
        self.screen.blit(self.images.empty_button, self.coords.cancel)

    def draw_start_game(self):
        self.screen.blit(self.images.start_game, self.coords.confirm)

    def draw_player_choices(self):
        self.screen.blit(self.images.player1, self.coords.confirm)
        self.screen.blit(self.images.player2, self.coords.cancel)

    def draw_score(self, deck_num, scores):
        self.screen.blit(self.images.scores, self.coords.scores_coords[deck_num])
        for idx, score in enumerate(scores):
            self.screen.blit(self.images.numbers[score],
                self.coords.scores[deck_num][idx + 1])

    def clear_move(self, move):
        for hex_ in move.iterator():
            x, y, colour = hex_
            self.draw_hex_dark((x, y))
            self.draw_hex_tile((x, y), colour)

    def clear_last_move(self):
        if self.last_move is not None:
            self.clear_move(self.last_move)

    def set_last_move(self, move):
        self.last_move = move

    def highlight_choice_colour(self, colour):
        self.screen.blit(self.images.select_red_light,
            self.choice_map[self.col_to_choice_coords[colour]])
        self.screen.blit(self.colour_image_map[colour],
            self.coords.add_offset(
                self.choice_map[self.col_to_choice_coords[colour]]))
