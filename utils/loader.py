import pygame as pg
from pygame.locals import *

LIGHT_TILE_PATH = 'imgs/tile_light.png'
DARK_TILE_PATH = 'imgs/tile_dark.png'
SELECT_RED_DARK_PATH = 'imgs/select_red_dark.png'
SELECT_RED_LIGHT_PATH = 'imgs/select_red_light.png'
SELECT_BLUE_DARK_PATH = 'imgs/select_blue_dark.png'
SELECT_BLUE_LIGHT_PATH = 'imgs/select_blue_light.png'

ORANGE_TILE_PATH = 'imgs/tile_orange.png'
YELLOW_TILE_PATH = 'imgs/tile_yellow.png'
BLUE_TILE_PATH = 'imgs/tile_blue.png'
PURPLE_TILE_PATH = 'imgs/tile_purple.png'
GREEN_TILE_PATH = 'imgs/tile_green.png'
RED_TILE_PATH = 'imgs/tile_red.png'

SCORES_PATH = 'imgs/scores.png'
SCORE_PATH = 'imgs/score.png'
CONFIRM_PATH = 'imgs/confirm.png'
CANCEL_PATH = 'imgs/cancel.png'
INGENIOUS_PATH = 'imgs/ingenious.png'
START_PATH = 'imgs/start.png'
EMPTY_PATH =  'imgs/empty.png'

PLAYER1 = 'imgs/player1.png'
PLAYER2 = 'imgs/player2.png'

class ImageLoader:
    def __init__(self):
        self.light_hex = pg.image.load(LIGHT_TILE_PATH)
        self.dark_hex = pg.image.load(DARK_TILE_PATH)
        self.select_red_dark = pg.image.load(SELECT_RED_DARK_PATH)
        self.select_red_light = pg.image.load(SELECT_RED_LIGHT_PATH)
        self.select_blue_dark = pg.image.load(SELECT_BLUE_DARK_PATH)
        self.select_blue_light = pg.image.load(SELECT_BLUE_LIGHT_PATH)
        self.o = pg.image.load(ORANGE_TILE_PATH)
        self.y = pg.image.load(YELLOW_TILE_PATH)
        self.b = pg.image.load(BLUE_TILE_PATH)
        self.p = pg.image.load(PURPLE_TILE_PATH)
        self.g = pg.image.load(GREEN_TILE_PATH)
        self.r = pg.image.load(RED_TILE_PATH)
        self.scores = pg.image.load(SCORES_PATH)
        self.score = pg.image.load(SCORE_PATH)
        self.confirm = pg.image.load(CONFIRM_PATH)
        self.cancel = pg.image.load(CANCEL_PATH)
        self.player1 = pg.image.load(PLAYER1)
        self.player2 = pg.image.load(PLAYER2)
        self.ingenious = pg.image.load(INGENIOUS_PATH)
        self.start_game = pg.image.load(START_PATH)
        self.empty_box = pg.image.load(EMPTY_PATH)
        self.empty_button = pg.image.load(EMPTY_PATH)
        self.process_images()
        self.load_numbers()
        self.process_numbers()
        self.colours = {1: self.r, 2: self.o, 3: self.y, 4: self.g, 5: self.b, 6: self.p}

    def process_images(self):
        self.light_hex = pg.transform.scale(self.light_hex, (43, 50))
        self.dark_hex = pg.transform.scale(self.dark_hex, (43, 50))
        self.select_red_dark = pg.transform.scale(self.select_red_dark, (43, 50))
        self.select_red_light = pg.transform.scale(self.select_red_light, (43, 50))
        self.select_blue_dark = pg.transform.scale(self.select_blue_dark, (43, 50))
        self.select_blue_light = pg.transform.scale(self.select_blue_light, (43, 50))
        self.o = pg.transform.scale(self.o, (35, 37))
        self.y = pg.transform.scale(self.y, (35, 35))
        self.b = pg.transform.scale(self.b, (35, 35))
        self.p = pg.transform.scale(self.p, (35, 35))
        self.g = pg.transform.scale(self.g, (35, 35))
        self.r = pg.transform.scale(self.r, (35, 35))
        self.ingenious = pg.transform.scale(self.ingenious, (240, 60))
        self.scores = pg.transform.scale(self.scores, (200, 70))
        self.empty_box = pg.transform.scale(self.empty_box, (240, 95))
        self.empty_button = pg.transform.scale(self.empty_button, (140, 30))
        self.confirm = pg.transform.scale(self.confirm, (140, 30))
        self.cancel = pg.transform.scale(self.cancel, (140, 30))
        self.player1 = pg.transform.scale(self.player1, (140, 30))
        self.player2 = pg.transform.scale(self.player2, (140, 30))
        self.start_game = pg.transform.scale(self.start_game, (140, 30))

    def get_colour_image_map(self):
        return {
            1: self.r,
            2: self.o,
            3: self.y,
            4: self.g,
            5: self.b,
            6: self.p
        }

    def load_numbers(self):
        self.numbers = {}
        for i in range(0, 19):
            filepath = 'imgs/{}.png'.format(i)
            self.numbers[i] = pg.image.load(filepath)

    def process_numbers(self):
        for i in range(0, 10):
            self.numbers[i] = pg.transform.scale(self.numbers[i], (20, 28))
        for i in range(10, 19):
            self.numbers[i] = pg.transform.scale(self.numbers[i], (30, 28))
