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
    winner, _ = gameplay.play_test_game(player_1, player_2)
    return winner

def main():
    test_player_params = {"player_type": "computer", "strategy_type": "rl", "network_type": "mlp_v1", "ckpt_path": "best_ckpts/mlp_v1_best_rule_0106_2231.pth"}
    other_player_params = {
        "random": {"player_type": "computer", "strategy_type": "random"},
        "increase_min": {"player_type": "computer", "strategy_type": "increase_min"},
        "max": {"player_type": "computer", "strategy_type": "max"},
        "reduce_deficit": {"player_type": "computer", "strategy_type": "reduce_deficit"},
        "mixed_4": {"player_type": "computer", "strategy_type": "mixed_4"},
        "mlp2_only": {"player_type": "computer", "strategy_type": "rl", "network_type": "mlp2_only", "ckpt_path": "best_ckpts/mlp2_only_best_0205_2252.pth"},
        "mlp_v1_self": {"player_type": "computer", "strategy_type": "rl", "network_type": "mlp_v1", "ckpt_path": "best_ckpts/mlp_v1_best_self_3005_2225.pth"},
        # "mlp_v1_rule": {"player_type": "computer", "strategy_type": "rl", "network_type": "mlp_v1", "ckpt_path": "best_ckpts/mlp_v1_best_rule_3005_2225.pth"},
        "mlp_v1_self_v2": {"player_type": "computer", "strategy_type": "rl", "network_type": "mlp_v1", "ckpt_path": "best_ckpts/mlp_v1_best_self_0106_2231.pth"},
        # "mlp_v1_rule_v2": {"player_type": "computer", "strategy_type": "rl", "network_type": "mlp_v1", "ckpt_path": "best_ckpts/mlp_v1_best_rule_0106_2231.pth"},
    }

    gameplay = get_gameplay({"game_type": "training", "representation": "v1", "value_type": "v1"})
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
