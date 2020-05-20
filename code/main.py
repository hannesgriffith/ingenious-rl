import pygame as pg
from pygame.locals import *

from utils.io import load_json
from game.gameplay import get_gameplay
from ui.controller import Controller

INPUT_SETTINGS = 'settings_play.json'

class App:
    def __init__(self):
        self.show_screen = True

    def cleanup(self):
        pg.quit()

    def execute(self):
        self.game_params = load_json(INPUT_SETTINGS)
        self.game_params[1] = self.game_params["1"]
        self.game_params[2] = self.game_params["2"]

        self.controller = Controller(self.game_params)
        self.controller.run()
        while self.show_screen:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.show_screen = False
        self.cleanup()

if __name__ == "__main__" :
    app = App()
    app.execute()
