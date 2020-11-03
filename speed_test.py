import time

from pygame.locals import *
from tqdm import tqdm
import numpy as np

from utils.io import load_json
from game.gameplay import get_gameplay 
from game.player import get_player
from learn.strategy import get_strategy_types

INPUT_SETTINGS = 'settings_speed_test.json'
N_GAMES = 1000

def play_game(gameplay, params):
    strat_1, strat_2 = get_strategy_types(params)
    player_1 = get_player(params[1]["player_type"], None, strat_1, params=params[1])
    player_2 = get_player(params[2]["player_type"], None, strat_2, params=params[2])
    winner, _ = gameplay.play_test_game(player_1, player_2)
    return winner

def main():
    params = load_json(INPUT_SETTINGS)
    params[1] = params["1"]
    params[2] = params["2"]

    start_time_1 = time.time()

    gameplay = get_gameplay(params)
    play_game(gameplay, params)

    start_time_2 = time.time()

    wins = []
    for _ in tqdm(range(N_GAMES)):
        winner = play_game(gameplay, params)
        wins.append(winner)

    end_time = time.time()
    time_elapsed_1 = end_time - start_time_1
    time_elapsed_2 = end_time - start_time_2
    compilation_time = time_elapsed_1 - time_elapsed_2
    time_per_game = time_elapsed_2 / float(N_GAMES)

    print(f"Time taken for {N_GAMES} games with compilation time: {time_elapsed_1}")
    print(f"Time taken for {N_GAMES} games without compilation time: {time_elapsed_2}")
    print(f"Compilation time: {compilation_time}")
    print(f"Seconds per game: {time_per_game}")

    wins = np.array(wins)
    p1_win_rate = np.sum(wins == 1) / float(N_GAMES)
    p2_win_rate = np.sum(wins == 2) / float(N_GAMES)

    print(f"Player 1 win rate: {p1_win_rate}, player 2 win rate: {p2_win_rate}")

if __name__ == "__main__" :
    main()
