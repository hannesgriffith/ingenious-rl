import time

import pygame as pg
from pygame.locals import *
from tqdm import tqdm
import numpy as np

from utils.io import load_json
from game.gameplay import get_gameplay, Move, get_strategy_types
from game.board import Board
from game.tiles import Tiles
from game.player import get_player
from learn.representation import get_representation

N_GAMES = 1000

def play_game(gameplay, params_1, params_2):
    board, tiles = Board(), Tiles()
    player_1 = get_player(params_1["player_type"], board, params_1["strategy_type"], params=params_1)
    player_2 = get_player(params_2["player_type"], board, params_2["strategy_type"], params=params_2)
    winner = gameplay.play_test_game(player_1, player_2)
    return winner

def main():
    test_player_params = {"player_type": "computer", "strategy_type": "mixed_4"}
    other_player_params = {
        "random": {"player_type": "computer", "strategy_type": "random"},
        "increase_min": {"player_type": "computer", "strategy_type": "increase_min"},
        "max": {"player_type": "computer", "strategy_type": "max"},
        "reduce_deficit": {"player_type": "computer", "strategy_type": "reduce_deficit"},
        "mixed_4": {"player_type": "computer", "strategy_type": "mixed_4"}
    }

    gameplay = get_gameplay({"game_type": "training", "representation": "v2", "value_type": "v1"})
    for strat, params2 in other_player_params.items():
        wins = []
        for _ in tqdm(range(N_GAMES)):
            winner = play_game(gameplay, test_player_params, params2)
            wins.append(winner)

        wins = np.array(wins)
        p1_win_rate = np.sum(wins == 1) / float(N_GAMES)
        p2_win_rate = np.sum(wins == 2) / float(N_GAMES)
        print(f"Win rate: {p1_win_rate} / {p2_win_rate} against {strat}")

if __name__ == "__main__" :
    main()

# Max vs Max:                           Player 1 win rate: 0.505, player 2 win rate: 0.495
# Max vs Random:                        Player 1 win rate: 1.000, player 2 win rate: 0.000
# Max vs Increase Min:                  Player 1 win rate: 0.571, player 2 win rate: 0.429
# Max vs Increase Other Min:            Player 1 win rate: 0.788, player 2 win rate: 0.212
# Max vs Reduce Deficit -5:             Player 1 win rate: 0.509, player 2 win rate: 0.491
# Max vs Reduce Deficit -2:             Player 1 win rate: 0.539, player 2 win rate: 0.461
# Max vs Reduce Deficit 0:              Player 1 win rate: 0.536, player 2 win rate: 0.464
# Max vs Reduce Deficit 2:              Player 1 win rate: 0.493, player 2 win rate: 0.507
# Max vs Reduce Deficit 5:              Player 1 win rate: 0.458, player 2 win rate: 0.542
# Max vs Reduce Deficit 8:              Player 1 win rate: 0.462, player 2 win rate: 0.538
# Max vs Reduce Deficit 15:             Player 1 win rate: 0.513, player 2 win rate: 0.487
# Max vs Mixed 1 (-5, 10):              Player 1 win rate: 0.497, player 2 win rate: 0.503
# Max vs Mixed 1 (-3, 8):               Player 1 win rate: 0.507, player 2 win rate: 0.493
# Max vs Mixed 1 (3, 10):               Player 1 win rate: 0.499, player 2 win rate: 0.501
# Max vs Mixed 1 (0, 5):                Player 1 win rate: 0.482, player 2 win rate: 0.518
# Max vs Mixed 2:                       Player 1 win rate: 0.561, player 2 win rate: 0.439
# Reduce Deficit 5 vs Random:           Player 1 win rate: 1.000, player 2 win rate: 0.000
# Reduce Deficit 5 vs Increase Min:     Player 1 win rate: 0.666, player 2 win rate: 0.334
# Reduce Deficit 5 vs Increase Min2:    Player 1 win rate: 0.826, player 2 win rate: 0.174
# Reduce Deficit 5 vs Mixed 1:          Player 1 win rate: 0.472, player 2 win rate: 0.528
# Reduce Deficit 5 vs Mixed 2:          Player 1 win rate: 0.620, player 2 win rate: 0.380
# Mixed 3 (15) vs Max:                  Player 1 win rate: 0.506, player 2 win rate: 0.494
# Mixed 3 (15) vs Reduce Deficit 5:     Player 1 win rate: 0.487, player 2 win rate: 0.513
# Mixed 3 (15) vs Mixed 1:              Player 1 win rate: 0.503, player 2 win rate: 0.497
# Mixed 3 (10) vs Max:                  Player 1 win rate: 0.476, player 2 win rate: 0.524
# Mixed 3 (10) vs Reduce Deficit 5:     Player 1 win rate: 0.476, player 2 win rate: 0.524
# Mixed 3 (10) vs Mixed 1:              Player 1 win rate: 0.474, player 2 win rate: 0.526
# Mixed 4 vs Max:                       Player 1 win rate: 0.543, player 2 win rate: 0.457
# Mixed 4 vs Reduce Deficit 5:          Player 1 win rate: 0.519, player 2 win rate: 0.481
# Mixed 4 vs Mixed 1:                   Player 1 win rate: 0.508, player 2 win rate: 0.492
# Mixed 4 vs Mixed 3:                   Player 1 win rate: 0.545, player 2 win rate: 0.455
