import pygame as pg
from pygame.locals import *

class UICoords:
    def __init__(self):
        self.x_start = 350
        self.y_start = 20
        self.x_step = 50
        self.y_step = 43

        self.deck1i_coords = (30, 525, 450, 100)
        self.deck1o_coords = (self.deck1i_coords[0] - 5, 520, 460, 110)
        self.deck2i_coords = (self.deck1i_coords[0] + 475, 525, 450, 100)
        self.deck2o_coords = (self.deck2i_coords[0] - 5, 520, 460, 110)
        self.choice1i_coords = (self.x_start + 450, 30, 150, 300)
        self.choice1o_coords = (self.choice1i_coords[0] - 5, 25, 160, 310)

        self.deck1_start_x = self.deck1i_coords[0] + 27
        self.deck2_start_x = self.deck2i_coords[0] + 27
        self.deck_start_y = 530
        self.deck_step_x = 75

        self.choice_start_x = self.x_start + 470
        self.choice_start_y = 45
        self.choice_step_x = 70
        self.choice_step_y = 55

        self.scores_coords = {}
        self.scores_coords[1] = (self.deck1i_coords[0], self.deck1i_coords[1] - 81)
        self.scores_coords[2] =(self.deck2i_coords[0] + self.deck2i_coords[2] - 200, self.deck2i_coords[1] - 81)

        self.rescaled_tile_side = 35

        self.confirm = (self.choice1o_coords[0] + 10,
            self.choice1o_coords[1] + self.choice1o_coords[3] + 5)
        self.cancel = (self.confirm[0], self.confirm[1] + 35)

        self.setup_scores()

    def get_valid_tiles(self):
        valid_tiles = [(i, j) for i in range(6) for j in range(11)]
        valid_tiles.extend([(6, j) for j in range(1, 10)])
        valid_tiles.extend([(7, j) for j in range(2, 9)])
        valid_tiles.extend([(8, j) for j in range(3, 8)])
        valid_tiles.extend([(9, j) for j in range(4, 7)])
        valid_tiles.append((10, 5))
        return valid_tiles

    def get_hex_coord_map(self):
        """Get the coordinates of the hexes that make up the board grid"""
        valid_tiles = self.get_valid_tiles()
        hex_map = {}
        for hex in valid_tiles:
            if hex[1] < 6:
                i = self.x_start - hex[1] * self.x_step / 2 + hex[0] * self.x_step
                j = self.y_start + hex[1] * self.y_step
            else:
                i = self.x_start + (hex[1] - 10) * self.x_step / 2 + hex[0] * self.x_step
                j = self.y_start + hex[1] * self.y_step
            hex_map[hex] = (int(i) ,int(j))
        return hex_map

    def get_deck_map(self, start_x):
        """Get the coordinates of the tiles showing in the player's deck"""
        deck_map = {}
        for i in range(6):
            deck_map[(i, 0)] = (start_x + i * self.deck_step_x, self.deck_start_y)
            deck_map[(i, 1)] = (start_x - 23 + i * self.deck_step_x, self.deck_start_y + 41)
        return deck_map

    def get_choice_map(self):
        """Get the coordinates of the choice tiles used for selecting new tiles"""
        choice_map = {}
        for i in range(2):
            for j in range(3):
                choice_map[(i, j)] = (self.choice_start_x + i * self.choice_step_x,
                    self.choice_start_y + j * self.choice_step_y)
        return choice_map

    def get_choice_coords_to_col_map(self):
        return {(0, 0): 1, (1, 0): 6, (0, 1): 3, (1, 1): 4, (0, 2): 2, (1, 2): 5}

    def get_col_to_choice_coords_map(self):
        return {v: k for k, v in self.get_choice_coords_to_col_map().items()}

    def add_offset(self, coords):
        """Add offset so that tiles are in middle of hex"""
        new_coords = (coords[0] + 4, coords[1] + 7)
        return new_coords

    def get_tile_coord_map(self):
        """Offset tile coords to align better with centre of hexes"""
        hex_map = self.get_hex_coord_map()
        tile_map = {}
        for hex, coords in hex_map.items():
            tile_map[hex] = self.add_offset(coords)
        return tile_map

    def get_eg_map(self):
        return {0: (self.x_start + 515, 220),
                1: (self.x_start + 490, 263)}

    def get_start_hexes(self):
        return [(0, 0), (0, 5), (0, 10), (5, 0), (5, 10), (10, 5)]

    def setup_scores(self):
        self.scores = {1: {}, 2: {}}
        self.scores[1][1] = (self.deck1i_coords[0] + 40, self.deck1i_coords[1] - 72)
        self.scores[1][2] = (self.scores[1][1][0] + 60, self.scores[1][1][1] + 26)
        self.scores[1][3] = (self.scores[1][1][0] + 120, self.scores[1][1][1])
        self.scores[1][4] = (self.scores[1][1][0] + 60, self.scores[1][1][1])
        self.scores[1][5] = (self.scores[1][1][0] + 120, self.scores[1][1][1] + 26)
        self.scores[1][6] = (self.scores[1][1][0], self.scores[1][1][1] + 26)

        offset = 725
        self.scores[2][1] = (self.scores[1][1][0] + offset, self.scores[1][1][1])
        self.scores[2][2] = (self.scores[1][2][0] + offset, self.scores[1][2][1])
        self.scores[2][3] = (self.scores[1][3][0] + offset, self.scores[1][3][1])
        self.scores[2][4] = (self.scores[1][4][0] + offset, self.scores[1][4][1])
        self.scores[2][5] = (self.scores[1][5][0] + offset, self.scores[1][5][1])
        self.scores[2][6] = (self.scores[1][6][0] + offset, self.scores[1][6][1])
