import random as rn
from copy import deepcopy

import numpy as np

from game.misc import num_to_col, col_to_num, remove_cluster
from game.misc import define_available_masks, define_direction_masks

# Note, moves are represented as a 6 element array: [i1, j1, i2, j2, c1, c2]

class Board:
    def __init__(self, n=6):
        # NOTE: add padding for processing. Remove for learning.
        self.n = n + 1
        self.side = 2 * self.n - 1
        self.available_masks = define_available_masks()
        self.direction_masks = define_direction_masks()

        self.valid_hexes = None
        self.state = None
        self.occupied = None
        self.available = None
        self.first_move = None
        self.move_num = 1

        self.initialise_valid_hexes()
        self.initialise_state()
        self.intialise_occupied()
        self.initialise_available()

    def initialise_valid_hexes(self):
        self.valid_hexes = np.ones((self.side, self.side), dtype='int32')
        for i in range(1, self.n):
            self.valid_hexes[self.n-1-i, self.side-i-1:] = 0
            self.valid_hexes[self.n-1+i, self.side-i-1:] = 0
        self.valid_hexes[0, :] = 0
        self.valid_hexes[:, 0] = 0
        self.valid_hexes[-1, :] = 0
        self.valid_hexes[:, -1] = 0

    def initialise_state(self):
        """Colour numbers as per col_to_num. 0 is unoccupied. -1 not valid."""
        self.state = np.zeros((self.side, self.side))
        self.state[self.valid_hexes == 0] = -1
        self.state[1, 1] = col_to_num("red")
        self.state[1, 6] = col_to_num("green")
        self.state[11, 1] = col_to_num("yellow")
        self.state[6, 1] = col_to_num("purple")
        self.state[6, 11] = col_to_num("blue")
        self.state[11, 6] = col_to_num("orange")

    def intialise_occupied(self):
        self.occupied = np.ones((self.side, self.side), dtype='int32')
        self.occupied[self.state == 0] = 0

    def initialise_available(self):
        self.available = np.zeros((self.side, self.side), dtype='int32')
        self.available = self.update_available(self.available, [1, 1],
                                        self.available_masks, self.occupied)
        self.available = self.update_available(self.available, [1, 6],
                                        self.available_masks, self.occupied)
        self.available = self.update_available(self.available, [11, 1],
                                        self.available_masks, self.occupied)
        self.available = self.update_available(self.available, [6, 1],
                                        self.available_masks, self.occupied)
        self.available = self.update_available(self.available, [6, 11],
                                        self.available_masks, self.occupied)
        self.available = self.update_available(self.available, [11, 6],
                                        self.available_masks, self.occupied)

    def update_state(self, move):
        self.state[move.coord1_i, move.coord1_j] = move.colour1
        self.state[move.coord2_i, move.coord2_j] = move.colour2

    def peak_board_update(self, move):
        nextState = np.copy(self.state)
        nextState[move.coord2_i, move.coord2_j] = move.colour1
        nextState[move.coord2_i, move.coord2_j] = move.colour2

        nextOccupied = np.copy(self.occupied)
        nextOccupied[move.coord2_i, move.coord2_j] = 1
        nextOccupied[move.coord2_i, move.coord2_j] = 1

        coords = [move.coord2_i, move.coord2_j]
        nextAvailable = np.copy(self.available)
        nextAvailable = self.update_available(nextAvailable,
                                              coords,
                                              self.available_masks,
                                              nextOccupied)

        return (nextState, nextOccupied, nextAvailable)

    def update_occupied(self, move):
        self.occupied[move.coord1_i, move.coord1_j] = 1
        self.occupied[move.coord2_i, move.coord2_j] = 1

    def get_area_of_board(self, i):
        if i < self.n - 1:
            return "top"
        if i == self.n - 1:
            return "middle"
        if i > self.n - 1:
            return "bottom"

    def update_available(self, grid, c, available_masks, occupied):
        # c is coord of newly placed tile hex
        board_area = self.get_area_of_board(c[0])
        grid[c[0]-1:c[0]+2, c[1]-1:c[1]+2] += available_masks[board_area]
        grid[grid > 0] = 1
        grid[grid < 1] = 0
        grid[occupied == 1] = 0
        return grid

    def update_board(self, move):
        self.update_state(move)
        self.update_occupied(move)
        self.available = self.update_available(self.available,
                                               [move.coord1_i, move.coord1_j],
                                               self.available_masks,
                                               self.occupied)
        self.available = self.update_available(self.available,
                                               [move.coord2_i, move.coord2_j],
                                               self.available_masks,
                                               self.occupied)
        self.filter_occupied_and_available()
        if self.move_num == 1:
            self.first_move = deepcopy(move)
        self.move_num += 1

    def filter_occupied_and_available(self):
        # filter single spaces that can't connected to any other ones
        idxs = np.where(self.available == 1)
        positions = np.stack([idxs[0], idxs[1]], axis=1)
        for position_idx in range(positions.shape[0]):
            i, j = positions[position_idx, :]
            board_area = self.get_area_of_board(i)
            patch = self.available[i-1:i+2, j-1:j+2] * \
                    self.available_masks[board_area]
            if np.sum(patch) == 0:
                self.occupied[i, j] = 1
                self.available[i, j] = 0

    def get_all_possible_moves(self):
        # NOTE: The possible moves generated here includes "reflections". So they
        # don't need to be accounted for later when adding colours from the tiles.

        available_copy = np.copy(self.available)
        if self.move_num == 2:
            # Deals with first move case where you cannot move to same cluster
            available_copy = remove_cluster(available_copy, self.first_move)

        possible_moves = set()
        idxs = np.where(available_copy == 1)
        positions = np.stack([idxs[0], idxs[1]], axis=1)

        for position_idx in range(positions.shape[0]):
            i, j = positions[position_idx, :]
            expanded_available = self.update_available(available_copy, [i, j],
                                        self.available_masks, self.occupied)
            board_area = self.get_area_of_board(i)
            check = expanded_available[i-1:i+2, j-1:j+2] * self.available_masks[board_area]
            check[1, 1] = 0
            if np.sum(check) > 0:
                idxs2 = np.where(check == 1)
                next_positions = np.stack([idxs2[0], idxs2[1]], axis=1)
                next_positions -= np.ones_like(next_positions)
                for next_position_idx in range(next_positions.shape[0]):
                    i2, j2 = next_positions[next_position_idx, :]
                    coords1 = (i, j)
                    coords2 = (i + i2, j + j2)
                    possible_moves.add((coords1, coords2))
                    possible_moves.add((coords2, coords1))
        return [Move(c[0], c[1], None, None) for c in possible_moves]

    def check_move_is_legal(self, move):
        allowed_moves = self.get_all_possible_moves()
        coord1 = (move.coord1_i, move.coord1_j)
        coord2 = (move.coord2_i, move.coord2_j)
        for allowed_move in allowed_moves:
            allowed_coord1 = (allowed_move.coord1_i, allowed_move.coord1_j)
            allowed_coord2 = (allowed_move.coord2_i, allowed_move.coord2_j)
            if coord1 == allowed_coord1 and coord2 == allowed_coord2:
                return True
        return False

    def step_in_direction(self, d, i, j):
        """Step directions clockwise from up-right are 0-5, ending up-left."""
        board_area = self.get_area_of_board(i)
        idxs = np.where(self.direction_masks[board_area] == d)
        offsets = np.stack([idxs[0], idxs[1]], axis=1)[0]
        offsets -= np.ones_like(offsets)
        i += offsets[0]
        j += offsets[1]
        return i, j

    def calculate_move_score(self, move):
        move_score = np.zeros((6,), dtype='int32')
        for i, j, c in move.iterator():
            board_area = self.get_area_of_board(i)

            patch = self.state[i-1:i+2, j-1:j+2] * self.available_masks[board_area]
            check = (patch == c)
            if np.sum(check) > 0:
                idxs = np.where(check == True)
                coords = np.stack([idxs[0], idxs[1]], axis=1).tolist()
                directions = [self.direction_masks[board_area][x, y]
                    for x, y in coords]
                for direction in directions:
                    i1, j1 = i, j
                    following_direction = True
                    while following_direction:
                        i1, j1 = self.step_in_direction(direction, i1, j1)
                        if self.state[i1, j1] == c:
                            move_score[c-1] += 1
                        else:
                            following_direction = False
        return move_score.tolist()

    def peak_move_score(self, board_state, move):
        move_score = np.zeros((6,), dtype='int32')
        for i, j, c in move.iterator():
            board_area = self.get_area_of_board(i)

            patch = board_state[i-1:i+2, j-1:j+2] * self.available_masks[board_area]
            check = (patch == c)
            if np.sum(check) > 0:
                idxs = np.where(check == True)
                coords = np.stack([idxs[0], idxs[1]], axis=1).tolist()
                directions = [self.direction_masks[board_area][x, y]
                    for x, y in coords]
                for direction in directions:
                    i1, j1 = i, j
                    following_direction = True
                    while following_direction:
                        i1, j1 = self.step_in_direction(direction, i1, j1)
                        if board_state[i1, j1] == c:
                            move_score[c-1] += 1
                        else:
                            following_direction = False
        return move_score.tolist()

    def peak_hex_score(self, i, j, c, board_state):
        score = 0
        board_area = self.get_area_of_board(i)
        patch = board_state[i-1:i+2, j-1:j+2] * self.available_masks[board_area]
        check = (patch == c)
        if np.sum(check) > 0:
            idxs = np.where(check == True)
            coords = np.stack([idxs[0], idxs[1]], axis=1).tolist()
            directions = [self.direction_masks[board_area][x, y]
                          for x, y in coords]
            for direction in directions:
                i1, j1 = i, j
                following_direction = True
                while following_direction:
                    i1, j1 = self.step_in_direction(direction, i1, j1)
                    if board_state[i1, j1] == c:
                        score += 1
                    else:
                        following_direction = False
        return score

    def get_hex_scores(next_board_state, next_board_available):
        hex_scores = np.zeros((next_board_available.shape[0],
                               next_board_available.shape[1], 6, 10))
        idxs = np.where(next_board_available == 1)
        positions = np.stack([idxs[0], idxs[1]], axis=1)

        for position_idx in range(positions.shape[0]):
            i, j = positions[position_idx, :]
            for c in range(1, 7):
                hex_colour_score = peak_hex_score(i, j, c, next_board_state)
                hex_colour_score = np.clip(0, 10) #*** check 10 or 9 ***
                hex_scores[i, j, c-1, hex_colour_score] += 1

        return hex_scores

class Move:
    """Store grid coordinates and colours (by their numbers)"""
    def __init__(self, coord1, coord2, colour1, colour2):
        self.coord1_i = coord1[0]
        self.coord1_j = coord1[1]
        self.colour1 = colour1
        self.coord2_i = coord2[0]
        self.coord2_j = coord2[1]
        self.colour2 = colour2

    def iterator(self):
        hex1 = (self.coord1_i, self.coord1_j, self.colour1)
        hex2 = (self.coord2_i, self.coord2_j, self.colour2)
        for hex in [hex1, hex2]:
            yield hex

    def __str__(self):
        hex1 = " ".join(["(", str(self.coord1_i), str(self.coord1_j), str(num_to_col(self.colour1)), ")"])
        hex2 = " ".join(["(", str(self.coord2_i), str(self.coord2_j), str(num_to_col(self.colour2)), ")"])
        return hex1 + " " + hex2
