import numpy as np

# NOTE: Move convention in board: [i1, j1, k1, i2, j2, k2, colour1, colour2]

# NOTE: Colour conventions:
# In game:      0: red, 1: orange, 2: yellow, 3: green, 4: blue, 5: purple
# In display:   1: red, 2: orange, 3: yellow, 4: green, 5: blue, 6: purple

def combine_moves_and_deck(possible_moves, deck):
    d = deck.get_deck().astype(np.uint8)
    p = possible_moves.astype(np.uint8)
    idxs = np.array(np.meshgrid(np.arange(p.shape[0]), np.arange(d.shape[0]))).T.reshape(-1, 2)
    combined = np.hstack([p[idxs[:, 0], :], d[idxs[:, 1], :]]).reshape(-1, 8)
    return combined

class Board:
    def __init__(self, n=6):
        self.n = n
        self.side = 2 * self.n - 1
        self.move_num = 1
        self.first_new_possible_moves = None

        self.define_connectivity_kernels()
        self.define_cluster_arrays()

        self.initialise_valid_hexes()
        self.intialise_occupied()
        self.initialise_available()
        self.initialise_possible_moves()

    def initialise_valid_hexes(self):
        self.valid_hexes = np.zeros([self.side] * 3, dtype=np.uint8)
        idxs_1 = np.arange(self.side).reshape(-1, 1, 1)
        idxs_2 = np.arange(self.side).reshape(1, -1, 1)
        idxs_3 = np.arange(self.side).reshape(1, 1, -1)
        sum_idxs = idxs_1 + idxs_2 + idxs_3
        valid_mask = sum_idxs == (3 * (self.n - 1)) # cube constraint
        self.valid_hexes[valid_mask] = 1
        self.valid_hexes = np.pad(self.valid_hexes, 1, mode='constant', constant_values=0)

    def intialise_occupied(self):
        self.occupied = np.ones([self.side + 2] * 3, dtype=np.uint8)
        self.occupied[self.valid_hexes == 1] = 0
        self.occupied[
            [6, 11, 1, 1, 11, 6],
            [11, 6, 6, 11, 1, 1],
            [1, 1, 11, 6, 6, 11]
        ] = 1

    def initialise_available(self):
        self.available = np.zeros([self.side + 2] * 3, dtype=np.uint8) # padded size
        self.last_available = np.zeros_like(self.available)
        self.available = self.update_available_for_hex(np.array([5, 10, 0], dtype=np.uint8), self.available, self.occupied)
        self.available = self.update_available_for_hex(np.array([10, 5, 0], dtype=np.uint8), self.available, self.occupied)
        self.available = self.update_available_for_hex(np.array([0, 5, 10], dtype=np.uint8), self.available, self.occupied)
        self.available = self.update_available_for_hex(np.array([0, 10, 5], dtype=np.uint8), self.available, self.occupied)
        self.available = self.update_available_for_hex(np.array([10, 0, 5], dtype=np.uint8), self.available, self.occupied)
        self.available = self.update_available_for_hex(np.array([5, 0, 10], dtype=np.uint8), self.available, self.occupied)

    def initialise_possible_moves(self):
        self.possible_moves = self.get_new_possible_moves()

    def define_connectivity_kernels(self):
        # kernels: 3 directions x 3 x 3 x 3
        self.k = np.zeros((3, 3, 3, 3), dtype=np.uint8)
        self.k[
                [0, 0, 1, 1, 2, 2], 
                [1, 1, 0, 2, 0, 2], 
                [0, 2, 1, 1, 2, 0], 
                [2, 0, 2, 0, 1, 1]
            ] = 1

        # full kernel: 3 x 3 x 3
        self.f = np.sum(self.k, axis=0).astype(np.uint8)

    def define_cluster_arrays(self):
        # cluster ids: 3 directions x s x s x s x 6 colours (padded to do dot with kernel)
        self.c = np.zeros((3, self.side + 2, self.side + 2, self.side + 2, 6), dtype=np.uint8)

        # sizes of clusters: 3 directions x s x s x s x 6 colours (padded to do dot with kernel)
        self.s = np.zeros((3, self.side + 2, self.side + 2, self.side + 2, 6), dtype=np.uint8)

        self.update_clusters(np.array([5, 10, 0], dtype=np.uint8), 0)
        self.update_clusters(np.array([10, 5, 0], dtype=np.uint8), 3)
        self.update_clusters(np.array([0, 5, 10], dtype=np.uint8), 2)
        self.update_clusters(np.array([0, 10, 5], dtype=np.uint8), 5)
        self.update_clusters(np.array([10, 0, 5], dtype=np.uint8), 4)
        self.update_clusters(np.array([5, 0, 10], dtype=np.uint8), 1)

    def update_clusters_for_direction(self, d, valid_neighbours, i, j, k, c):
        idxs = np.where(valid_neighbours != 0)
        num_neighbours = idxs[0].shape[0]
        if num_neighbours == 0:
            cluster_number = np.max(self.c[d, :, :, :, c]) + 1
            self.c[d, i, j, k, c] = cluster_number
            self.s[d, i, j, k, c] = 1
        elif num_neighbours == 1:
            cluster_number = valid_neighbours[idxs[0][0], idxs[1][0], idxs[2][0]]
            self.c[d, i, j, k, c] = cluster_number
            cluster_mask = (self.c[d, :, :, :, c] == cluster_number)
            self.s[d, :, :, :, c][cluster_mask] = np.sum(cluster_mask)
        elif num_neighbours == 2:
            cluster_number_1 = valid_neighbours[idxs[0][0], idxs[1][0], idxs[2][0]]
            cluster_number_2 = valid_neighbours[idxs[0][1], idxs[1][1], idxs[2][1]]
            cluster_number = np.min([cluster_number_1, cluster_number_2])
            self.c[d, i, j, k, c] = cluster_number
            cluster_mask = ((self.c[d, :, :, :, c] == cluster_number_1) | (self.c[d, :, :, :, c] == cluster_number_2))
            self.c[d, :, :, :, c][cluster_mask] = cluster_number
            self.s[d, :, :, :, c][cluster_mask] = np.sum(cluster_mask)
        else:
            assert False

    def update_clusters(self, coords, c):
        # given coords & colour of a single hex
        padded_coords = coords + 1 # convert to padded coords
        i, j, k = padded_coords.flatten()
        neighbourhood = self.c[:, i-1:i+2, j-1:j+2, k-1:k+2, c]
        valid_neighbours = neighbourhood * self.k

        self.update_clusters_for_direction(0, valid_neighbours[0], i, j, k, c)
        self.update_clusters_for_direction(1, valid_neighbours[1], i, j, k, c)
        self.update_clusters_for_direction(2, valid_neighbours[2], i, j, k, c)

    def get_connected_hexes(self, hex_coords):
        padded_hex_coords = hex_coords + 1 # convert to padded coords
        i, j, k = padded_hex_coords.flatten()
        neighbourhood = self.f * self.valid_hexes[i-1:i+2, j-1:j+2, k-1:k+2]
        connected_hexes = np.stack(np.where(neighbourhood), axis=1).reshape(3, 3)
        return hex_coords + connected_hexes - 1

    def get_connected_start_hex(self, move):
        start_hexes = np.array(
            [
                [5, 10, 0],
                [10, 5, 0],
                [0, 5, 10],
                [0, 10, 5],
                [10, 0, 5],
                [5, 0, 10]
            ]).reshape(6, 3).astype(np.float)
        move_hex_1 = move[0:3].reshape(1, 3).astype(np.float)
        distances = np.sum(np.power((start_hexes - move_hex_1).astype(np.float32), 2), 1)
        min_idx = np.argmin(distances, axis=0)
        return start_hexes[min_idx, ...].astype(np.uint8)

    def update_occupied(self, move):
        padded_move = move + 1 # convert to padded coords
        self.occupied[padded_move[0], padded_move[1], padded_move[2]] = 1
        self.occupied[padded_move[3], padded_move[4], padded_move[5]] = 1

    def game_is_finished(self):
        return np.sum(self.available) < 2

    def update_available_for_hex(self, single_hex_coords, available_mask, occupied_mask):
        available_mask = np.copy(available_mask)
        single_hex_coords_padded = single_hex_coords + 1 # convert to padded coords
        i, j, k = single_hex_coords_padded.flatten()
        available_mask[i-1:i+2, j-1:j+2, k-1:k+2] = self.f
        available_mask[occupied_mask == 1] = 0

        idxs = np.where(available_mask[i-1:i+2, j-1:j+2, k-1:k+2] == 1)
        num_neighbours = idxs[0].shape[0]
        idxs = np.stack(idxs, axis=1).reshape(num_neighbours, 3)
        idxs = idxs -1 + single_hex_coords_padded.reshape(1, 3)
        for idx in range(num_neighbours):
            i2, j2, k2 = idxs[idx, ...]
            neighbours_patch = available_mask[i2-1:i2+2, j2-1:j2+2, k2-1:k2+2]
            available_neighbours = np.sum(neighbours_patch * self.f)
            if available_neighbours == 0:
                available_mask[i2, j2, k2] = 0

        return available_mask

    def update_available(self, move):
        self.last_available = np.copy(self.available)
        self.available = self.update_available_for_hex(move[0:3], self.available, self.occupied)
        self.available = self.update_available_for_hex(move[3:6], self.available, self.occupied)
        self.available[self.occupied == 1] = 0

    def update_scoring_clusters(self, move):
        self.update_clusters(move[0:3], move[6])
        self.update_clusters(move[3:6], move[7])

    def update_board(self, move):
        self.update_occupied(move)
        self.update_available(move)
        self.update_scoring_clusters(move)

        if self.move_num == 1:
            self.first_move_update(move)
        elif self.move_num == 2:
            self.second_move_update(move)
        else:
            self.update_possible_moves(move)

        self.move_num += 1

    def first_move_update(self, move):
        start_hex = self.get_connected_start_hex(move)
        connected_hexes = self.get_connected_hexes(start_hex)

        r1 = self.remove_coords_from_possible_moves(connected_hexes[0]).reshape(-1, 6)
        r2 = self.remove_coords_from_possible_moves(connected_hexes[1]).reshape(-1, 6)
        r3 = self.remove_coords_from_possible_moves(connected_hexes[2]).reshape(-1, 6)

        coords_moves = np.vstack([r1, r2, r3])
        coords_moves = self.remove_coords_from_specific_moves(move[0:3], coords_moves)
        coords_moves = self.remove_coords_from_specific_moves(move[3:6], coords_moves)

        r4 = self.remove_coords_from_possible_moves(move[0:3]).reshape(-1, 6)
        r5 = self.remove_coords_from_possible_moves(move[3:6]).reshape(-1, 6)
        r6 = self.get_new_possible_moves().reshape(-1, 6)
        move_moves = np.vstack([r4, r5, r6])

        combined_moves = np.vstack([coords_moves, move_moves])
        combined_moves = np.unique(combined_moves, axis=0)
        self.first_new_possible_moves = combined_moves

    def second_move_update(self, move):
        self.remove_coords_from_possible_moves(move[0:3])
        self.remove_coords_from_possible_moves(move[3:6])

        new_possible_moves = self.get_new_possible_moves()
        if new_possible_moves is not None:
            self.possible_moves = np.vstack([self.possible_moves, new_possible_moves])

        self.possible_moves = np.vstack([self.possible_moves, self.first_new_possible_moves])
        self.possible_moves = np.unique(self.possible_moves, axis=0)

    def remove_coords_from_possible_moves(self, coords):
        coords = coords.reshape(1, 3)
        col_1_match = np.all(self.possible_moves[:, 0:3] == coords, axis=1)
        col_2_match = np.all(self.possible_moves[:, 3:6] == coords, axis=1)
        match_indices = np.where(col_1_match | col_2_match)[0]
        moves_to_remove = self.possible_moves[match_indices, ...]
        self.possible_moves = np.delete(self.possible_moves, match_indices, axis=0)
        return moves_to_remove

    def remove_coords_from_specific_moves(self, coords, specific_moves):
        coords = coords.reshape(1, 3)
        col_1_match = np.all(specific_moves[:, 0:3] == coords, axis=1)
        col_2_match = np.all(specific_moves[:, 3:6] == coords, axis=1)
        match_indices = np.where(col_1_match | col_2_match)[0]
        specific_moves = np.delete(specific_moves, match_indices, axis=0)
        return specific_moves

    def get_new_possible_moves(self):
        newly_available = (self.available == 1) & (self.last_available == 0)
        idxs = np.where(newly_available)
        num_neighbours = idxs[0].shape[0]
        if num_neighbours == 0:
            return None

        all_new_possible_moves = np.zeros((1, 6), dtype=np.uint8)
        idxs = np.stack(idxs, axis=1).reshape(num_neighbours, 3)
        for idx in range(num_neighbours):
            coords_1 = idxs[idx, ...].reshape(1, 3)
            i, j, k = coords_1.flatten()
            available_patch_2 = (self.occupied[i-1:i+2, j-1:j+2, k-1:k+2] == 0) * self.f
            neighbours_2 = np.where(available_patch_2)
            num_neighbours_2 = neighbours_2[0].shape[0]
            if num_neighbours_2 > 0:
                coords_2 = coords_1 + np.stack(neighbours_2, axis=1).reshape(-1, 3) - 1
                coords_1_tiled = np.tile(coords_1, (num_neighbours_2, 1))
                possible_moves = np.hstack([coords_1_tiled, coords_2]).reshape(-1, 6)
                possible_moves -= 1 # convert to unpadded coords
                possible_moves_flipped = np.hstack([coords_2, coords_1_tiled]).reshape(-1, 6)
                possible_moves_flipped -= 1 # convert to unpadded coords
                all_new_possible_moves = np.vstack([all_new_possible_moves, possible_moves, possible_moves_flipped])

        return all_new_possible_moves[1:, :].reshape(-1, 6)

    def update_possible_moves(self, move):
        self.remove_coords_from_possible_moves(move[0:3])
        self.remove_coords_from_possible_moves(move[3:6])
        new_possible_moves = self.get_new_possible_moves()
        if new_possible_moves is not None:
            self.possible_moves = np.vstack([self.possible_moves, new_possible_moves])
        if self.possible_moves.ndim > 1 and self.possible_moves.shape[0] > 0:
            self.possible_moves = np.unique(self.possible_moves, axis=0)

    def get_all_possible_moves(self):
        return self.possible_moves

    def check_move_is_legal(self, move):
        return np.any(np.all(move[0:6].reshape(1, 6) == self.possible_moves.reshape(-1, 6), axis=1))

    def calculate_hex_score(self, coords, c):
        move_score = np.zeros((6,), dtype=np.uint8)
        padded_coords = coords + 1 # offset to padded coords
        i, j, k = padded_coords.flatten()
        cluster_scores = self.s[:, i-1:i+2, j-1:j+2, k-1:k+2, c] * self.k
        move_score[c] = np.sum(cluster_scores)
        return move_score

    def calculate_move_score(self, move):
        move_score_1 = self.calculate_hex_score(move[0:3], move[6])
        move_score_2 = self.calculate_hex_score(move[3:6], move[7])
        return move_score_1 + move_score_2

    def batch_calculate_hex_scores(self, coords, colours):
        b = coords.shape[0]
        move_scores = np.zeros((b, 6), dtype=np.uint8)
        c = colours.reshape(b, 1, 1, 1)
        padded_coords = coords.reshape(b, 3) + 1 # b x 3 (offset to padded coords)
        padded_coords = np.expand_dims(padded_coords, [1, 2, 3]) # b x 1 x 1 x 1 x 3

        idxs = np.meshgrid(np.arange(3), np.arange(3), np.arange(3), indexing='ij') # 3 x (3 x 3 x 3)
        offsets = np.stack(idxs, axis=3) - 1 # 3 x 3 x 3 x 3
        offsets = np.expand_dims(offsets, 0) # 1 x 3 x 3 x 3 x 3

        i  = padded_coords + offsets # b x 3 x 3 x 3 x 3
        cluster_scores = self.s[:, i[..., 0], i[..., 1], i[..., 2], c] * np.expand_dims(self.k, 1) # 3 x b x 3 x 3 x 3

        num_clusters = np.sum(cluster_scores > 0, axis=(0, 2, 3, 4)) # b
        move_scores[np.arange(b), np.squeeze(c)] = np.sum(cluster_scores, axis=(0, 2, 3, 4)) + num_clusters # b x 6
        return move_scores # b x 6

    def batch_calculate_move_scores(self, moves):
        if moves.ndim == 1 or moves.shape[0] == 1:
            return self.calculate_move_score(moves)

        num_moves = moves.shape[0]
        stacked_hex_coords = np.vstack([moves[:, 0:3], moves[:, 3:6]])
        stacked_hex_colours = np.vstack([moves[:, 6:7], moves[:, 7:8]])
        move_scores = self.batch_calculate_hex_scores(stacked_hex_coords, stacked_hex_colours)
        return move_scores[:num_moves, :] + move_scores[num_moves:, :]
