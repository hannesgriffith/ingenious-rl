import copy

import numpy as np
from scipy.ndimage import label

from game import board

def col_to_num(col):
    map = {
        "red": 1,
        "orange": 2,
        "yellow": 3,
        "green": 4,
        "blue": 5,
        "purple": 6
    }
    return map[col]

def num_to_col(num):
    map = {
        1: "red",
        2: "orange",
        3: "yellow",
        4: "green",
        5: "blue",
        6: "purple"
    }
    return map[num]

def define_available_masks():
    return {"top":    np.array([[1, 1, 0],
                                [1, 0, 1],
                                [0, 1, 1]], dtype='int32'),
            "middle": np.array([[1, 1, 0],
                                [1, 0, 1],
                                [1, 1, 0]], dtype='int32'),
            "bottom": np.array([[0, 1, 1],
                                [1, 0, 1],
                                [1, 1, 0]], dtype='int32')
    }

def define_direction_masks():
    return {"top":    np.array([[5, 0, -1],
                                [4, -1, 1],
                                [-1, 3, 2]], dtype='int32'),
            "middle": np.array([[5, 0, -1],
                                [4, -1, 1],
                                [3, 2, -1]], dtype='int32'),
            "bottom": np.array([[-1, 5, 0],
                                [4, -1, 1],
                                [3, 2, -1]], dtype='int32')
    }

def combine_moves_and_deck(possible_moves, deck):
    all_moves = []
    for possible_move in possible_moves:
        for tile in deck.iterator():
            move = copy.deepcopy(possible_move)
            move.colour1 = tile[0]
            move.colour2 = tile[1]
            all_moves.append(move)
    return all_moves

def flip_tile(tile):
    return (tile[1], tile[0])

def game_to_display_coords(coords):
    return (coords[1] - 1, coords[0] - 1)

def display_to_game_coords(coords):
    return (coords[1] + 1, coords[0] + 1)

def display_to_game_move(input_move):
    coord1 = (input_move.coord1_i, input_move.coord1_j)
    coord1 = display_to_game_coords(coord1)
    coord2 = (input_move.coord2_i, input_move.coord2_j)
    coord2 = display_to_game_coords(coord2)
    colour1 = input_move.colour1
    colour2 = input_move.colour2
    output_move = board.Move(coord1, coord2, colour1, colour2)
    return output_move

def game_to_display_move(input_move):
    coord1 = (input_move.coord1_i, input_move.coord1_j)
    coord1 = game_to_display_coords(coord1)
    coord2 = (input_move.coord2_i, input_move.coord2_j)
    coord2 = game_to_display_coords(coord2)
    colour1 = input_move.colour1
    colour2 = input_move.colour2
    output_move = board.Move(coord1, coord2, colour1, colour2)
    return output_move

def remove_cluster(available_mask, first_move):
    kernel = np.ones((3, 3), dtype=np.uint8)
    labeled_mask, num_features = label(available_mask, structure=kernel)
    first_coord = np.array([first_move.coord1_i, first_move.coord1_j])
    permutations = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    for permutation in permutations:
        permutation = np.array(permutation)
        coord_permuted = (first_coord + permutation).tolist()
        if available_mask[coord_permuted[0], coord_permuted[1]] == 1:
            label_num = labeled_mask[coord_permuted[0], coord_permuted[1]]
            available_mask[labeled_mask == label_num] = 0
            return available_mask
    assert False
