from numba import njit
import numpy as np

@njit
def get_other_player(player):
    if player == 1:
        return 2
    else:
        return 1

@njit
def find_winner_fast(p1_score, p2_score):
    p1_score.sort()
    p2_score.sort()

    for i in range(6):
        p1_colour_score = p1_score[i]
        p2_colour_score = p2_score[i]

        if p1_colour_score > p2_colour_score:
            return 1
        if p2_colour_score > p1_colour_score:
            return 2

    return 0