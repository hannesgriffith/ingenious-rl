from numba import njit, jitclass, uint8, types, int64
import numpy as np

# NOTE: Move convention in board: [i1, j1, k1, i2, j2, k2, colour1, colour2]

# NOTE: Colour conventions:
# In game:      0: red, 1: orange, 2: yellow, 3: green, 4: blue, 5: purple
# In display:   1: red, 2: orange, 3: yellow, 4: green, 5: blue, 6: purple

@njit
def combine_moves_and_deck(p, d):
    # NOTE: possible moves already includes 'reflections' so no need to account for here
    num_possible_moves = p.shape[0]
    num_tiles_in_deck = d.shape[0]
    num_combinations = num_possible_moves * num_tiles_in_deck

    combined = np.zeros((num_combinations, 8), dtype=np.uint8)
    for i in range(num_possible_moves):
        for j in range(num_tiles_in_deck):
            idx = num_tiles_in_deck * i + j
            combined[idx, 0:6] = p[i]
            combined[idx, 6:8] = d[j]

    return combined

board_spec = [
    ('n', uint8),
    ('side', uint8),
    ('move_num', uint8),
    ('first_move', uint8[:]),
    ('occupied', uint8[:, :, :]),
    ('available', uint8[:, :, :]),
    ('last_available', uint8[:, :, :]),
    ('possible_moves', uint8[:, :]),
    ('playable_coords', uint8[:, :]),
    ('first_new_possible_moves', uint8[:, :]),
    ('k', uint8[:, :, :]),
    ('f', uint8[:, :]),
    ('c', uint8[:, :, :, :, :]),
    ('s', uint8[:, :, :, :, :]),
    ('state', uint8[:, :, :]),
    ('conversion', types.DictType(
        types.UniTuple(int64, 3),
        types.UniTuple(int64, 2)
        ))
]

@jitclass(board_spec)
class Board:
    def __init__(self):
        self.n = 6
        self.side = 11
        self.move_num = 1

        self.first_move = np.zeros((6,), dtype=np.uint8)
        self.occupied = np.ones((self.side + 2, self.side + 2, self.side + 2), dtype=np.uint8)
        self.available = np.zeros((self.side + 2, self.side + 2, self.side + 2), dtype=np.uint8)
        self.last_available = np.zeros((self.side + 2, self.side + 2, self.side + 2), dtype=np.uint8)
        self.possible_moves = np.zeros((1, 6), dtype=np.uint8)
        self.playable_coords = np.zeros((91, 3), dtype=np.uint8)
        self.first_new_possible_moves = np.zeros((1, 6), dtype=np.uint8)
        # self.possible_tiles = np.zeros((21, 2), dtype=np.uint8)

        self.k = np.zeros((3, 2, 3), dtype=np.uint8)
        self.f = np.zeros((6, 3), dtype=np.uint8)
        self.c = np.zeros((3, self.side + 2, self.side + 2, self.side + 2, 6), dtype=np.uint8)
        self.s = np.zeros((3, self.side + 2, self.side + 2, self.side + 2, 6), dtype=np.uint8)
        self.state = np.zeros((self.side, self.side, 8), dtype=np.uint8) # 6 colours, occupied, available

        self.initialise_conversion()
        self.set_playable_coords()
        self.initialise_occupied()
        self.define_connectivity_kernels()
        self.define_cluster_arrays()
        self.initialise_available()
        self.initialise_possible_moves()
        self.initialise_state()
        # self.set_possible_tiles()

    def set_playable_coords(self):
        self.playable_coords = np.array([
            [ 0,  5, 10], [ 0,  6,  9], [ 0,  7,  8], [ 0,  8,  7], [ 0,  9,  6], [ 0, 10,  5], [ 1,  4, 10],
            [ 1,  5,  9], [ 1,  6,  8], [ 1,  7,  7], [ 1,  8,  6], [ 1,  9,  5], [ 1, 10,  4], [ 2,  3, 10],
            [ 2,  4,  9], [ 2,  5,  8], [ 2,  6,  7], [ 2,  7,  6], [ 2,  8,  5], [ 2,  9,  4], [ 2, 10,  3],
            [ 3,  2, 10], [ 3,  3,  9], [ 3,  4,  8], [ 3,  5,  7], [ 3,  6,  6], [ 3,  7,  5], [ 3,  8,  4],
            [ 3,  9,  3], [ 3, 10,  2], [ 4,  1, 10], [ 4,  2,  9], [ 4,  3,  8], [ 4,  4,  7], [ 4,  5,  6],
            [ 4,  6,  5], [ 4,  7,  4], [ 4,  8,  3], [ 4,  9,  2], [ 4, 10,  1], [ 5,  0, 10], [ 5,  1,  9],
            [ 5,  2,  8], [ 5,  3,  7], [ 5,  4,  6], [ 5,  5,  5], [ 5,  6,  4], [ 5,  7,  3], [ 5,  8,  2],
            [ 5,  9,  1], [ 5, 10,  0], [ 6,  0,  9], [ 6,  1,  8], [ 6,  2,  7], [ 6,  3,  6], [ 6,  4,  5],
            [ 6,  5,  4], [ 6,  6,  3], [ 6,  7,  2], [ 6,  8,  1], [ 6,  9,  0], [ 7,  0,  8], [ 7,  1,  7],
            [ 7,  2,  6], [ 7,  3,  5], [ 7,  4,  4], [ 7,  5,  3], [ 7,  6,  2], [ 7,  7,  1], [ 7,  8,  0],
            [ 8,  0,  7], [ 8,  1,  6], [ 8,  2,  5], [ 8,  3,  4], [ 8,  4,  3], [ 8,  5,  2], [ 8,  6,  1],
            [ 8,  7,  0], [ 9,  0,  6], [ 9,  1,  5], [ 9,  2,  4], [ 9,  3,  3], [ 9,  4,  2], [ 9,  5,  1],
            [ 9,  6,  0], [10,  0,  5], [10,  1,  4], [10,  2,  3], [10,  3,  2], [10,  4,  1], [10,  5,  0]], dtype=np.uint8).reshape(-1, 3)
        self.playable_coords += np.ones_like(self.playable_coords)

    # def set_possible_tiles(self):
    #     self.possible_tiles = np.array([
    #         [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
    #         [1, 1], [1, 2], [1, 3], [1, 4], [1, 5],
    #         [2, 2], [2, 3], [2, 4], [2, 5],
    #         [3, 3], [3, 4], [3, 5],
    #         [4, 4], [4, 5],
    #         [5, 5]
    #     ], dtype=np.uint8)

    def initialise_occupied(self):
        self.state[:, :, 6] = 1
        for idx in range(self.playable_coords.shape[0]):
            i, j, k = self.playable_coords[idx]
            self.occupied[i, j, k] = 0
            self.state = self.update_state_occupied_for_hex(self.state, (i, j, k), val=0, padded=True)

        self.occupied[6, 11, 1] = 1
        self.occupied[11, 6, 1] = 1
        self.occupied[1, 6, 11] = 1
        self.occupied[1, 11, 6] = 1
        self.occupied[11, 1, 6] = 1
        self.occupied[6, 1, 11] = 1

        self.state = self.update_state_occupied_for_hex(self.state, (6, 11, 1), val=1, padded=True)
        self.state = self.update_state_occupied_for_hex(self.state, (11, 6, 1), val=1, padded=True)
        self.state = self.update_state_occupied_for_hex(self.state, (1, 6, 11), val=1, padded=True)
        self.state = self.update_state_occupied_for_hex(self.state, (1, 11, 6), val=1, padded=True)
        self.state = self.update_state_occupied_for_hex(self.state, (11, 1, 6), val=1, padded=True)
        self.state = self.update_state_occupied_for_hex(self.state, (6, 1, 11), val=1, padded=True)

    def define_connectivity_kernels(self):
        self.k[0, 0, :] = np.array([1, 2, 0])
        self.k[0, 1, :] = np.array([1, 0, 2])
        self.k[1, 0, :] = np.array([2, 1, 0])
        self.k[1, 1, :] = np.array([0, 1, 2])
        self.k[2, 0, :] = np.array([2, 0, 1])
        self.k[2, 1, :] = np.array([0, 2, 1])

        self.f[0, :] = np.array([1, 2, 0])
        self.f[1, :] = np.array([1, 0, 2])
        self.f[2, :] = np.array([2, 1, 0])
        self.f[3, :] = np.array([0, 1, 2])
        self.f[4, :] = np.array([2, 0, 1])
        self.f[5, :] = np.array([0, 2, 1])

    def define_cluster_arrays(self):
        self.c, self.s = self.get_updated_clusters(np.array([5, 10, 0], dtype=np.uint8), 0, self.c, self.s)
        self.c, self.s = self.get_updated_clusters(np.array([10, 5, 0], dtype=np.uint8), 3, self.c, self.s)
        self.c, self.s = self.get_updated_clusters(np.array([0, 5, 10], dtype=np.uint8), 2, self.c, self.s)
        self.c, self.s = self.get_updated_clusters(np.array([0, 10, 5], dtype=np.uint8), 5, self.c, self.s)
        self.c, self.s = self.get_updated_clusters(np.array([10, 0, 5], dtype=np.uint8), 4, self.c, self.s)
        self.c, self.s = self.get_updated_clusters(np.array([5, 0, 10], dtype=np.uint8), 1, self.c, self.s)

    def initialise_available(self):
        self.available, _ = self.update_available_for_hex(np.array([5, 10, 0], dtype=np.uint8), self.available, self.occupied)
        self.available, _ = self.update_available_for_hex(np.array([10, 5, 0], dtype=np.uint8), self.available, self.occupied)
        self.available, _ = self.update_available_for_hex(np.array([0, 5, 10], dtype=np.uint8), self.available, self.occupied)
        self.available, _ = self.update_available_for_hex(np.array([0, 10, 5], dtype=np.uint8), self.available, self.occupied)
        self.available, _ = self.update_available_for_hex(np.array([10, 0, 5], dtype=np.uint8), self.available, self.occupied)
        self.available, _ = self.update_available_for_hex(np.array([5, 0, 10], dtype=np.uint8), self.available, self.occupied)

    def initialise_possible_moves(self):
        new_possible_moves, _ = self.get_new_possible_moves()
        self.possible_moves = new_possible_moves

    def initialise_state(self):
        self.state = self.update_state_colours_for_hex(self.state, (5, 10, 0), 0)
        self.state = self.update_state_colours_for_hex(self.state, (10, 5, 0), 3)
        self.state = self.update_state_colours_for_hex(self.state, (0, 5, 10), 2)
        self.state = self.update_state_colours_for_hex(self.state, (0, 10, 5), 5)
        self.state = self.update_state_colours_for_hex(self.state, (10, 0, 5), 4)
        self.state = self.update_state_colours_for_hex(self.state, (5, 0, 10), 1)

    def update_state_colours_for_hex(self, state, coords, colour, padded=False):
        i1, j1, k1 = coords
        if padded:
            i1, j1, k1 = i1 - 1, j1 - 1, k1 - 1

        i2, j2 = self._3d_to_2d_r1(i1, j1, k1)
        state[i2, j2, colour] = 1

        return state

    def update_state_occupied_for_hex(self, state, coords, val=1, padded=False):
        i1, j1, k1 = coords
        if padded:
            i1, j1, k1 = i1 - 1, j1 - 1, k1 - 1

        i2, j2 = self._3d_to_2d_r1(i1, j1, k1)
        state[i2, j2, 6] = val

        return state

    def update_state_available_for_hex(self, state, coords, val, padded=False):
        i1, j1, k1 = coords
        if padded:
            i1, j1, k1 = i1 - 1, j1 - 1, k1 - 1

        i2, j2 = self._3d_to_2d_r1(i1, j1, k1)
        state[i2, j2, 7] = val

        return state

    def update_state_for_move(self, state, move, padded=False):
        i1, j1, k1, i2, j2, k2, c1, c2 = move.flatten()
        state = self.update_state_colours_for_hex(state, (i1, j1, k1), c1, padded=padded)
        state = self.update_state_colours_for_hex(state, (i2, j2, k2), c2, padded=padded)
        return state

    def where_idxs_3d(self, array):
        idxs_tuple = np.where(array)

        idxs_list = []
        for idx in range(3):
            idx_expanded = np.expand_dims(idxs_tuple[idx], 1)
            idxs_list.append(idx_expanded)

        idxs_stacked = np.hstack((idxs_list[0], idxs_list[1], idxs_list[2]))
        return idxs_stacked

    def update_where(self, array, old_value, new_value):
        idxs = self.where_idxs_3d(array == old_value)

        for idx in range(idxs.shape[0]):
            i, j, k = idxs[idx]
            array[i, j, k] = new_value

        return array

    def mask_update(self, ref_array, ref_value, update_array, update_val):
        idxs = self.where_idxs_3d(ref_array == ref_value)

        for idx in range(idxs.shape[0]):
            i, j, k = idxs[idx]
            update_array[i, j, k] = update_val

        return update_array

    def get_updated_clusters(self, coords, col, c, s):
        padded_coords = coords + np.ones_like(coords)
        i, j, k = padded_coords.flatten()

        for d in range(self.k.shape[0]):
            di1, dj1, dk1 = self.k[d, 0, :]
            di2, dj2, dk2 = self.k[d, 1, :]
            cluster_1 = c[d, i + di1 - 1, j + dj1 - 1, k + dk1 - 1, col]
            cluster_2 = c[d, i + di2 - 1, j + dj2 - 1, k + dk2 - 1, col]

            if cluster_1 == 0 and cluster_2 == 0:
                new_cluster_number = np.max(c) + 1
                c[d, i, j, k, col] = new_cluster_number
                s[d, i, j, k, col] = 1

            elif cluster_1 != 0 and cluster_2 != 0:
                if cluster_1 < cluster_2:
                    new_cluster_number = cluster_1
                    other_cluster_number = cluster_2
                else:
                    new_cluster_number = cluster_2
                    other_cluster_number = cluster_1

                c[d, i, j, k, col] = new_cluster_number
                c[d, :, :, :, col] = self.update_where(c[d, :, :, :, col], other_cluster_number, new_cluster_number)
                new_cluster_size = np.sum(c[d, :, :, :, col] == new_cluster_number)
                s[d, :, :, :, col] = self.mask_update(c[d, :, :, :, col], new_cluster_number, s[d, :, :, :, col], new_cluster_size)

            else:
                if cluster_1 != 0:
                    new_cluster_number = cluster_1
                else:
                    new_cluster_number = cluster_2
                c[d, i, j, k, col] = new_cluster_number
                new_cluster_size = np.sum(c[d, :, :, :, col] == new_cluster_number)
                s[d, :, :, :, col] = self.mask_update(c[d, :, :, :, col], new_cluster_number, s[d, :, :, :, col], new_cluster_size)

        return c, s

    def get_connected_start_hex(self, move):
        padded_move = move + np.ones_like(move)
        padded_move = padded_move.flatten().astype(np.uint8)
        start_hexes = np.array(
            [
                [5, 10, 0],
                [10, 5, 0],
                [0, 5, 10],
                [0, 10, 5],
                [10, 0, 5],
                [5, 0, 10]
            ], dtype=np.uint8).reshape(6, 3)
        padded_start_hexes = start_hexes + np.ones_like(start_hexes)

        move_hex_1 = padded_move[0:3]
        move_hex_2 = padded_move[3:6]

        for start_hex_idx in range(padded_start_hexes.shape[0]):
            padded_start_hex = padded_start_hexes[start_hex_idx]
            for offset_idx in range(self.f.shape[0]):
                offset = self.f[offset_idx]
                offset_hex_1 = move_hex_1 + offset - np.ones_like(offset)
                offset_hex_2 = move_hex_2 + offset - np.ones_like(offset)
                match_1 = np.all(offset_hex_1 == padded_start_hex)
                match_2 = np.all(offset_hex_2 == padded_start_hex)
                if match_1 or match_2:
                    return start_hexes[start_hex_idx]

    def game_is_finished(self):
        # return np.sum(self.available) < 2
        return self.get_all_possible_moves().shape[0] == 0

    def update_available_for_hex(self, single_hex_coords, available_mask, occupied_mask):
        single_hex_coords_padded = single_hex_coords + np.ones_like(single_hex_coords)
        i, j, k = single_hex_coords_padded.flatten()
        available_mask[i, j, k] = 0
        self.state = self.update_state_available_for_hex(self.state, (i, j, k), 0, padded=True)

        for offset_idx_1 in range(self.f.shape[0]):
            di1, dj1, dk1 = self.f[offset_idx_1]
            occupied_1 = (occupied_mask[i + di1 - 1, j + dj1 - 1, k + dk1 - 1] == 1)
            if not occupied_1:
                all_neighbours_occupied = True
                for offset_idx_2 in range(self.f.shape[0]):
                    di2, dj2, dk2 = self.f[offset_idx_2]
                    occupied_2 = (occupied_mask[i + di1 + di2 - 2, j + dj1 + dj2 - 2, k + dk1 + dk2 - 2] == 1)
                    if not occupied_2:
                        all_neighbours_occupied = False
                        break

                if all_neighbours_occupied:
                    available_mask[i + di1 - 1, j + dj1 - 1, k + dk1 - 1] = 0
                    self.state = self.update_state_available_for_hex(self.state, (i + di1 - 1, j + dj1 - 1, k + dk1 - 1), 0, padded=True)
                    occupied_mask[i + di1 - 1, j + dj1 - 1, k + dk1 - 1] = 1
                    self.state = self.update_state_occupied_for_hex(self.state, (i + di1 - 1, j + dj1 - 1, k + dk1 - 1), val=1, padded=True)
                else:
                    available_mask[i + di1 - 1, j + dj1 - 1, k + dk1 - 1] = 1
                    self.state = self.update_state_available_for_hex(self.state, (i + di1 - 1, j + dj1 - 1, k + dk1 - 1), 1, padded=True)

        available_mask[i, j, k] = 0
        self.state = self.update_state_available_for_hex(self.state, (i, j, k), 0, padded=True)
        return available_mask, occupied_mask

    def get_updated_occupied(self, move, occupied):
        padded_move = move + np.ones_like(move)
        occupied[padded_move[0], padded_move[1], padded_move[2]] = 1
        occupied[padded_move[3], padded_move[4], padded_move[5]] = 1
        self.state = self.update_state_occupied_for_hex(self.state, (move[0], move[1], move[2]), val=1, padded=False)
        self.state = self.update_state_occupied_for_hex(self.state, (move[3], move[4], move[5]), val=1, padded=False)
        return occupied

    def get_updated_available(self, move, available, occupied):
        last_available = np.copy(available)
        available, occupied = self.update_available_for_hex(move[0:3], available, occupied)
        available, occupied = self.update_available_for_hex(move[3:6], available, occupied)
        return available, last_available, occupied

    def get_updated_scoring_clusters(self, move, c, s):
        c, s = self.get_updated_clusters(move[0:3], int(move[6]), c, s)
        c, s = self.get_updated_clusters(move[3:6], int(move[7]), c, s)
        return c, s

    def get_updated_possible_moves(self, possible_moves, move_num, move):
        if move_num == 1:
            self.first_new_possible_moves, possible_moves = self.first_move_update(move, possible_moves)
        elif move_num == 2:
            possible_moves = self.second_move_update(move, possible_moves)
        else:
            possible_moves = self.get_standard_updated_possible_moves(move, possible_moves)
        return possible_moves

    def update_board(self, move):
        self.occupied = self.get_updated_occupied(move, self.occupied)
        self.available, self.last_available, self.occupied = self.get_updated_available(move, self.available, self.occupied)
        self.c, self.s = self.get_updated_scoring_clusters(move, self.c, self.s)
        self.possible_moves = self.get_updated_possible_moves(self.possible_moves, self.move_num, move)
        self.state = self.update_state_for_move(self.state, move)
        self.move_num += 1

    def first_move_update(self, move, possible_moves):
        start_hex = self.get_connected_start_hex(move)
        connected_hexes = self.get_connected_hexes(start_hex.astype(np.uint8))

        possible_moves, r1 = self.remove_coords_from_possible_moves(connected_hexes[0], possible_moves)
        possible_moves, r2 = self.remove_coords_from_possible_moves(connected_hexes[1], possible_moves)
        possible_moves, r3 = self.remove_coords_from_possible_moves(connected_hexes[2], possible_moves)
        r1, r2, r3 = r1.reshape(-1, 6), r2.reshape(-1, 6), r3.reshape(-1, 6)

        coords_moves = np.vstack((r1, r2, r3))
        coords_moves = self.remove_coords_from_specific_moves(move[0:3], coords_moves)
        coords_moves = self.remove_coords_from_specific_moves(move[3:6], coords_moves)

        possible_moves, r4 = self.remove_coords_from_possible_moves(move[0:3], possible_moves)
        possible_moves, r5 = self.remove_coords_from_possible_moves(move[3:6], possible_moves)
        r4, r5 = r4.reshape(-1, 6), r5.reshape(-1, 6)
        r6, _ = self.get_new_possible_moves()
        r6 = r6.reshape(-1, 6)
        move_moves = np.vstack((r4, r5, r6))

        combined_moves = np.vstack((coords_moves, move_moves))
        combined_moves_no_duplicates = self.remove_duplicates(combined_moves)
        return combined_moves_no_duplicates, possible_moves

    def second_move_update(self, move, possible_moves):
        new_possible_moves, success = self.get_new_possible_moves()
        if success:
            possible_moves = np.vstack((possible_moves, new_possible_moves))

        possible_moves = np.vstack((possible_moves, self.first_new_possible_moves))
        possible_moves, _ = self.remove_coords_from_possible_moves(move[0:3], possible_moves)
        possible_moves, _ = self.remove_coords_from_possible_moves(move[3:6], possible_moves)
        possible_moves = self.remove_duplicates(possible_moves)
        return possible_moves

    def remove_coords_from_possible_moves(self, coords, possible_moves):
        coords_to_remove = np.copy(coords).reshape(1, 3)
        col_1_match = self.all_axis(possible_moves[:, 0:3] == coords_to_remove, 1)
        col_2_match = self.all_axis(possible_moves[:, 3:6] == coords_to_remove, 1)
        matches = (col_1_match | col_2_match)
        others = (matches == 0)
        match_indices = np.where(matches)[0]
        other_indices = np.where(others)[0]
        moves_to_remove = possible_moves[match_indices]
        possible_moves = possible_moves[other_indices]
        return possible_moves, moves_to_remove

    def remove_coords_from_specific_moves(self, coords, specific_moves):
        coords_to_remove = np.copy(coords).reshape(1, 3)
        col_1_match = self.all_axis(specific_moves[:, 0:3] == coords_to_remove, 1)
        col_2_match = self.all_axis(specific_moves[:, 3:6] == coords_to_remove, 1)
        indices = np.where((col_1_match | col_2_match) == 0)[0]
        specific_moves_filtered = specific_moves[indices]
        return specific_moves_filtered

    def remove_duplicates(self, array):
        input_array_size = array.shape[0]
        output_array = np.zeros(array.shape, dtype=np.uint8)[0:1, :]

        for idx in range(input_array_size - 1):
            entry = array[idx:idx+1, :]
            match = np.any(self.all_axis(array[idx+1:input_array_size, :] == entry, 0))
            if not match:
                output_array = np.vstack((output_array, entry))

        output_array = np.vstack((output_array, array[-1:input_array_size, :]))
        output_array_size = output_array.shape[0]
        return np.copy(output_array[1:output_array_size, :])

    def get_new_possible_moves(self):
        newly_available = (self.available == 1) & (self.last_available == 0)
        if np.sum(newly_available) == 0:
            return np.zeros((1, 6), dtype=np.uint8), False

        idxs = self.where_idxs_3d(newly_available)

        all_new_possible_moves_tmp = np.zeros((1, 6), dtype=np.uint8)
        for idx in range(idxs.shape[0]):
            coords_1 = idxs[idx]

            for offset_idx in range(self.f.shape[0]):
                offset = self.f[offset_idx]
                coords_2 = coords_1 + offset - np.ones_like(offset)

                i2, j2, k2 = coords_2.flatten()
                occupied = (self.occupied[i2, j2, k2] == 1)

                if not occupied:
                    possible_moves = np.expand_dims(np.hstack((coords_1, coords_2)), 0).astype(np.uint8)
                    possible_moves_flipped = np.expand_dims(np.hstack((coords_2, coords_1)), 0).astype(np.uint8)
                    all_new_possible_moves_tmp = np.vstack((
                        all_new_possible_moves_tmp,
                        possible_moves - np.ones_like(possible_moves),
                        possible_moves_flipped - np.ones_like(possible_moves_flipped)
                    )).astype(np.uint8)

        all_new_possible_moves = self.remove_duplicates(all_new_possible_moves_tmp[1:, :])
        all_new_possible_moves = all_new_possible_moves.astype(np.uint8)
        return all_new_possible_moves.reshape(-1, 6), True

    def get_standard_updated_possible_moves(self, move, possible_moves):
        move = move.flatten().astype(np.uint8)
        new_possible_moves, success = self.get_new_possible_moves()
        if success:
            possible_moves = np.vstack((possible_moves, new_possible_moves))
        possible_moves, _ = self.remove_coords_from_possible_moves(move[0:3], possible_moves)
        possible_moves, _ = self.remove_coords_from_possible_moves(move[3:6], possible_moves)
        return possible_moves

    def get_state(self):
        return self.state

    def get_state_copy(self):
        return np.copy(self.get_state())

    def get_all_possible_moves(self):
        return self.possible_moves.astype(np.uint8)

    def check_move_is_legal(self, move):
        return np.any(self.all_axis(np.copy(move[0:6]).reshape(1, 6) == np.copy(self.possible_moves).reshape(-1, 6), 1))

    def any_axis(self, array, axis):
        return np.sum(array, axis=axis) > 0

    def all_axis(self, array, axis):
        array_shape = array.shape
        count = np.sum(array, axis=axis)
        return count == array_shape[axis]

    def get_connected_hexes(self, hex_coords):
        padded_hex_coords = hex_coords + np.ones_like(hex_coords)
        connected_hexes = np.zeros((1, 3), dtype=np.uint8)

        for offset_idx in range(self.f.shape[0]):
            offset = self.f[offset_idx].flatten()
            connected_hex = padded_hex_coords + offset - np.ones_like(offset)
            playable = np.any(self.all_axis(connected_hex.reshape(1, 3) == self.playable_coords, 1))
            if playable:
                connected_hexes = np.vstack((connected_hexes, connected_hex.reshape(1, 3)))

        connected_hexes -= np.ones_like(connected_hexes)
        connected_hexes = connected_hexes[1:, :]
        return connected_hexes

    def calculate_hex_score(self, coords, c):
        padded_coords = coords + np.ones_like(coords)
        i, j, k = padded_coords.flatten()
        colour_sum = 0
 
        for direction_idx in range(self.k.shape[0]):
            for offset_idx in range(self.k.shape[1]):
                di, dj, dk = self.k[direction_idx, offset_idx, :]
                colour_sum += self.s[direction_idx, i + di - 1, j + dj - 1, k + dk - 1, c]

        move_score = np.zeros((6,), dtype=np.uint8)
        move_score[c] = colour_sum
        return move_score

    def calculate_move_score(self, move):
        move = move.flatten()
        move_score_1 = self.calculate_hex_score(move[0:3], int(move[6]))
        move_score_2 = self.calculate_hex_score(move[3:6], int(move[7]))
        return move_score_1 + move_score_2

    def batch_calculate_move_scores(self, moves):
        if moves.ndim == 1 or moves.shape[0] == 1:
            return self.calculate_move_score(moves).reshape(-1, 6)

        move_scores = np.zeros((moves.shape[0], 6), dtype=np.uint8)

        for idx in range(moves.shape[0]):
            move = moves[idx:idx+1]
            move_score = self.calculate_move_score(move)
            move_scores[idx] = move_score

        return move_scores

    def batch_get_updated_states(self, moves):
        batch_size = moves.shape[0]
        updated_states = np.zeros((batch_size, 11, 11, 8))

        for idx in range(batch_size):
            move = moves[idx]
            updated_states[idx] = self.state

            for coords, c in [(move[0:3], move[6]), (move[3:6], move[7])]:
                i, j, k = coords
                i2, j2 = self._3d_to_2d_r1(i, j, k)
                updated_states[idx, i2, j2, c] = 1
                updated_states[idx, i2, j2, 6] = 1
                updated_states[idx, i2, j2, 7] = 0

                for offset_idx_1 in range(self.f.shape[0]):
                    di1, dj1, dk1 = self.f[offset_idx_1]
                    occupied_1 = (self.occupied[i + di1 - 1, j + dj1 - 1, k + dk1 - 1] == 1)
                    if not occupied_1:
                        all_neighbours_occupied = True
                        for offset_idx_2 in range(self.f.shape[0]):
                            di2, dj2, dk2 = self.f[offset_idx_2]
                            occupied_2 = (self.occupied[i + di1 + di2 - 2, j + dj1 + dj2 - 2, k + dk1 + dk2 - 2] == 1)
                            if not occupied_2:
                                all_neighbours_occupied = False
                                break

                        i3, j3 = self._3d_to_2d_r1(i + di1 - 1, j + dj1 - 1, k + dk1 - 1)
                        if all_neighbours_occupied:
                            updated_states[idx, i3, j3, 6] = 1
                            updated_states[idx, i3, j3, 7] = 0
                        else:
                            updated_states[idx, i3, j3, 7] = 1

                updated_states[idx, i2, j2, 7] = 0

        return updated_states

    def _3d_to_2d_r1(self, i1, j1, k1):
        tuple_3d = (i1, j1, k1)
        tuple_2d = self.conversion[tuple_3d]
        i2, j2 = tuple_2d
        return i2, j2

    def initialise_conversion(self):
        self.conversion = {
            (0, 9, 6):  (0, 4),
            (0, 10, 5): (0, 5),
            (1, 10, 4): (0, 6),
            (0, 7, 8):  (1, 2),
            (0, 8, 7):  (1, 3),
            (1, 8, 6):  (1, 4),
            (1, 9, 5):  (1, 5),
            (2, 9, 4):  (1, 6),
            (2, 10, 3): (1, 7),
            (3, 10, 2): (1, 8),
            (0, 5, 10): (2, 0),
            (0, 6, 9):  (2, 1),
            (1, 6, 8):  (2, 2),
            (1, 7, 7):  (2, 3),
            (2, 7, 6):  (2, 4),
            (2, 8, 5):  (2, 5),
            (3, 8, 4):  (2, 6),
            (3, 9, 3):  (2, 7),
            (4, 9, 2):  (2, 8),
            (4, 10, 1): (2, 9),
            (5, 10, 0): (2, 10),
            (1, 4, 10): (3, 0),
            (1, 5, 9):  (3, 1),
            (2, 5, 8):  (3, 2),
            (2, 6, 7):  (3, 3),
            (3, 6, 6):  (3, 4),
            (3, 7, 5):  (3, 5),
            (4, 7, 4):  (3, 6),
            (4, 8, 3):  (3, 7),
            (5, 8, 2):  (3, 8),
            (5, 9, 1):  (3, 9),
            (6, 9, 0):  (3, 10),
            (2, 3, 10): (4, 0),
            (2, 4, 9):  (4, 1),
            (3, 4, 8):  (4, 2),
            (3, 5, 7):  (4, 3),
            (4, 5, 6):  (4, 4),
            (4, 6, 5):  (4, 5),
            (5, 6, 4):  (4, 6),
            (5, 7, 3):  (4, 7),
            (6, 7, 2):  (4, 8),
            (6, 8, 1):  (4, 9),
            (7, 8, 0):  (4, 10),
            (3, 2, 10): (5, 0),
            (3, 3, 9):  (5, 1),
            (4, 3, 8):  (5, 2),
            (4, 4, 7):  (5, 3),
            (5, 4, 6):  (5, 4),
            (5, 5, 5):  (5, 5),
            (6, 5, 4):  (5, 6),
            (6, 6, 3):  (5, 7),
            (7, 6, 2):  (5, 8),
            (7, 7, 1):  (5, 9),
            (8, 7, 0):  (5, 10),
            (4, 1, 10): (6, 0),
            (4, 2, 9):  (6, 1),
            (5, 2, 8):  (6, 2),
            (5, 3, 7):  (6, 3),
            (6, 3, 6):  (6, 4),
            (6, 4, 5):  (6, 5),
            (7, 4, 4):  (6, 6),
            (7, 5, 3):  (6, 7),
            (8, 5, 2):  (6, 8),
            (8, 6, 1):  (6, 9),
            (9, 6, 0):  (6, 10),
            (5, 0, 10): (7, 0),
            (5, 1, 9):  (7, 1),
            (6, 1, 8):  (7, 2),
            (6, 2, 7):  (7, 3),
            (7, 2, 6):  (7, 4),
            (7, 3, 5):  (7, 5),
            (8, 3, 4):  (7, 6),
            (8, 4, 3):  (7, 7),
            (9, 4, 2):  (7, 8),
            (9, 5, 1):  (7, 9),
            (10, 5, 0): (7, 10),
            (6, 0, 9):  (8, 1),
            (7, 0, 8):  (8, 2),
            (7, 1, 7):  (8, 3),
            (8, 1, 6):  (8, 4),
            (8, 2, 5):  (8, 5),
            (9, 2, 4):  (8, 6),
            (9, 3, 3):  (8, 7),
            (10, 3, 2): (8, 8),
            (10, 4, 1): (8, 9),
            (8, 0, 7):  (9, 3),
            (9, 0, 6):  (9, 4),
            (9, 1, 5):  (9, 5),
            (10, 1, 4): (9, 6),
            (10, 2, 3): (9, 7),
            (10, 0, 5): (10, 5)
        }