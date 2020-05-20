import time

import pygame as pg
from pygame.locals import *
from tqdm import tqdm

from utils.io import load_json
from game.gameplay import get_gameplay, get_strategy_types
from game.board import Board
from game.tiles import Tiles
from game.player import get_player
from learn.representation import get_representation

INPUT_SETTINGS = 'settings_speed_test.json'
N_GAMES = 100

def main():
    params = load_json(INPUT_SETTINGS)
    params[1] = params["1"]
    params[2] = params["2"]

    gameplay = get_gameplay(params)

    start_time = time.time()

    for _ in tqdm(range(N_GAMES)):
        board = Board()
        tiles = Tiles()

        strat_1, strat_2 = get_strategy_types(params)
        player_1 = get_player(params[1]["player_type"], 1, params[1]["name"], board, tiles, strat_1)
        player_2 = get_player(params[2]["player_type"],2, params[2]["name"], board, tiles, strat_2)
        # representation = get_representation(params)

        gameplay.play_test_game(player_1, player_2)

    end_time = time.time()
    time_elapsed = end_time - start_time
    time_per_game = time_elapsed / float(N_GAMES)

    print(f"Time taken for {N_GAMES} games: {time_elapsed}")
    print(f"Seconds per game: {time_per_game}")

if __name__ == "__main__" :
    main()
