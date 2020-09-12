from numba import njit, prange
import numpy as np

from game.utils import fast_initialise_playable, fast_initialise_start_playable, fast_initialise_areas

class Board:
    def __init__(self):
        self.move_num = 1
        self.height, self.width = 11, 21 # all arrays are padded on top of these dims

        self.first_move = np.zeros((6,), dtype=np.uint8)
        self.offsets = np.array(((-1, -1), (-1, 1), (0, -2), (0, 2), (1, -1), (1, 1)), dtype=np.int8)
        self.directions = np.array((((-1, -1), (1, 1)), ((0, -2), (0, 2)), ((1, -1), (-1, 1))), dtype=np.int8)
        # Colour convention 0: red, 1: orange, 2: yellow, 3: green, 4: blue, 5: purple
        self.start_hexes = (((1, 7), 0), ((11, 7), 2), ((1, 17), 3), ((11, 17), 1), ((6, 2), 5), ((6, 22), 4))
        self.start_hexes_only = np.array(((1, 7), (11, 7), (1, 17), (11, 17), (6, 2), (6, 22)), dtype=np.uint8)

        self.initialise_playable()
        self.initialise_available()
        self.update_possible_moves()
        self.initialise_scoring_arrays()
        self.initialise_state()

    def initialise_playable(self):
        self.playable = fast_initialise_playable()

    def initialise_available(self):
        self.available = np.zeros((self.height + 2, self.width + 4), dtype=np.uint8)
        self.last_available = np.zeros((self.height + 2, self.width + 4), dtype=np.uint8)
        self.update_available_for_hexes(self.start_hexes)

    def update_available_for_hexes(self, hexes):
        for coords, _ in hexes:
            self.available = fast_update_available_for_hex(self.available, coords, self.offsets, self.playable)

    def update_possible_moves(self):
        # Move convention: [i1, j1, i2, j2, colour1, colour2]
        self.possible_moves = fast_get_all_possible_moves(self.playable, self.available, self.offsets, self.height, self.width)

    def get_possible_moves(self):
        return self.possible_moves

    def game_is_finished(self):
        return self.get_possible_moves().shape[0] == 0

    def check_move_is_legal(self, move):
        return fast_check_if_legal(move[0:4], self.get_possible_moves())

    def initialise_scoring_arrays(self):
        self.clusters = np.zeros((self.height + 2, self.width + 4, 3, 6), dtype=np.uint8)
        self.sizes = np.zeros((self.height + 2, self.width + 4, 3, 6), dtype=np.uint8)
        self.scores = np.zeros((self.height + 2, self.width + 4, 4, 6), dtype=np.uint8)
        self.update_scoring_arrays_for_hexes(self.start_hexes)

    def update_scoring_arrays_for_hexes(self, hexes):
        for coords, colour in hexes:
            self.clusters, self.sizes, self.scores = fast_update_scoring_arrays_for_hex(coords, colour, 
                self.clusters, self.sizes, self.scores, self.playable, self.directions, self.height, self.width)

    def initialise_state(self):
        self.state = np.zeros((self.height + 2, self.width + 4, 9), dtype=np.uint8)
        self.update_state_for_hexes(self.start_hexes)
        self.state[:, :, 7] = fast_initialise_start_playable()
        self.state[:, :, 8] = fast_initialise_areas()

    def update_state_for_hexes(self, hexes):
        for coords, colour in hexes:
            self.state = fast_update_state_for_hex(self.state, coords, colour)

    def update_playable_for_hex(self, coords):
        self.playable = fast_update_playable_for_hex(self.playable, coords, self.offsets)

    def update_playable_for_move(self, move):
        self.update_playable_for_hex(move[0:2])
        self.update_playable_for_hex(move[2:4])

    def update_available_for_move(self, move):
        hexes = ((move[0:2], move[4]), (move[2:4], move[5]))
        self.update_available_for_hexes(hexes)

    def update_scoring_arrays_for_move(self, move):
        hexes = ((move[0:2], move[4]), (move[2:4], move[5]))
        self.update_scoring_arrays_for_hexes(hexes)

    def update_state_for_move(self, move):
        hexes = ((move[0:2], move[4]), (move[2:4], move[5]))
        self.update_state_for_hexes(hexes)

    def update_available_for_move_given_move_num(self, move):
        if self.move_num == 1:
            self.update_available_for_first_move(move)
        elif self.move_num == 2:
            self.update_available_for_second_move(move)
        else:
            self.update_available_for_move(move)

    def update_board(self, move):
        self.update_playable_for_move(move)
        self.update_scoring_arrays_for_move(move)
        self.update_available_for_move_given_move_num(move)
        self.update_possible_moves()
        self.update_state_for_move(move)
        self.move_num += 1

    def update_available_for_first_move(self, move):
        hexes = ((move[0:2], move[4]), (move[2:4], move[5]))
        self.update_available_for_hexes(hexes)

        self.available, self.tmp_removed_available = fast_remove_first_available(
            self.available, move[0:2], move[2:4], self.start_hexes_only, self.offsets)

    def update_available_for_second_move(self, move):
        hexes = ((move[0:2], move[4]), (move[2:4], move[5]))
        self.update_available_for_hexes(hexes)
        self.available = fast_add_first_available_back(self.available, self.tmp_removed_available)

    def calculate_move_score(self, move):
        return fast_calculate_move_score(move, self.scores)

    def generate_state_representation(self):
        return np.dstack((self.state, self.playable, self.available)), self.scores

    def generate_vector_representation(self):
        return fast_generate_vector_representation(self.playable, self.available, self.state, self.scores, self.height, self.width)

    def batch_calculate_move_scores(self, moves):
        return fast_batch_calculate_move_score(moves, self.scores)

    def batch_generate_vector_representation(self, updated_playable, updated_available, updated_states, updated_scores):
        return fast_batch_generate_vector_representation(updated_playable, updated_available, updated_states, updated_scores, self.height, self.width)

    def batch_get_updated_playable(self, moves):
        return fast_batch_get_updated_playable(moves, self.height, self.width, self.playable, self.offsets)

    def batch_get_updated_available(self, moves, updated_playable):
        return fast_batch_get_updated_available(moves, updated_playable, self.height, self.width, self.available, self.offsets)

    def batch_get_updated_states(self, moves, updated_playable, updated_available):
        return fast_batch_get_updated_states(moves, self.state, self.height, self.width, updated_playable, updated_available)

    def batch_get_updated_scoring_arrays(self, moves, updated_playable):
        return fast_batch_get_updated_scoring_arrays(moves, updated_playable, self.height, self.width, self.directions, self.clusters, self.sizes, self.scores)

    def batch_generate_state_representations(self, moves):
        updated_playable = self.batch_get_updated_playable(moves)
        updated_available = self.batch_get_updated_available(moves, updated_playable)
        updated_states = self.batch_get_updated_states(moves, updated_playable, updated_available)
        _, _, updated_scores = self.batch_get_updated_scoring_arrays(moves, updated_playable)
        updated_board_vecs = self.batch_generate_vector_representation(updated_playable, updated_available, updated_states, updated_scores)
        return updated_states, updated_scores, updated_board_vecs

@njit(parallel=True, fastmath=True, cache=True)
def combine_moves_and_deck(possible_moves, deck):
    num_possible_moves = possible_moves.shape[0]
    num_tiles_in_deck = deck.shape[0]
    num_combinations = num_possible_moves * num_tiles_in_deck

    combined = np.zeros((num_combinations, 6), dtype=np.uint8)
    for i in prange(num_possible_moves):
        for j in range(num_tiles_in_deck):
            idx = num_tiles_in_deck * i + j
            combined[idx, 0:4] = possible_moves[i]
            combined[idx, 4:6] = deck[j]

    combined_filtered = fast_remove_duplicates(combined)

    return combined_filtered

@njit(parallel=True, fastmath=True, cache=True)
def fast_remove_duplicates(moves):
    num_moves = moves.shape[0]
    match = np.zeros((num_moves,), dtype=np.uint8)

    for i in prange(num_moves - 1):
        move = moves[i, :]
        other_moves = moves[i+1:num_moves, :]

        for j in range(other_moves.shape[0]):
            if other_moves[j, 0] != move[0]:
                continue
            if other_moves[j, 1] != move[1]:
                continue
            if other_moves[j, 2] != move[2]:
                continue
            if other_moves[j, 3] != move[3]:
                continue
            if other_moves[j, 4] != move[4]:
                continue
            if other_moves[j, 5] != move[5]:
                continue

            match[i] = 1
            break

    return moves[match == 0]

@njit(cache=True)
def fast_update_playable_for_hex(playable, coords, neighbour_offsets):
    i1, j1 = coords
    playable[i1, j1] = 0

    for idx_1 in range(6):
        di1, dj1 = neighbour_offsets[idx_1]
        i2 = i1 + di1
        j2 = j1 + dj1
        if playable[i2, j2] == 0:
            continue

        has_neighbours = False
        for idx_2 in range(6):
            di2, dj2 = neighbour_offsets[idx_2]
            i3 = i2 + di2
            j3 = j2 + dj2
            if playable[i3, j3] == 1:
                has_neighbours = True
                break

        if not has_neighbours:
            playable[i2, j2] = 0

    return playable

@njit(cache=True)
def fast_update_available_for_hex(available, coords, neighbour_offsets, playable):
    i1, j1 = coords
    available[i1, j1] = 0

    for o_idx in range(6):
        di, dj = neighbour_offsets[o_idx]
        i2 = i1 + di
        j2 = j1 + dj
        if playable[i2, j2] == 1:
            available[i2, j2] = 1

    return available

@njit(cache=True)
def fast_get_all_possible_moves(playable, available, neighbour_offsets, height, width):
    max_idxs = height * width * 6 * 2
    possible_moves = np.zeros((max_idxs, 5), dtype=np.uint8)

    for i1 in range(1, height + 1):
        for j1 in range(2, width + 2):
            if available[i1, j1] == 1:
                for o in range(6):
                    di, dj = neighbour_offsets[o]
                    i2 = i1 + di
                    j2 = j1 + dj
                    if playable[i2, j2] == 1:
                        idx = i1 * height + j1 * width + o * 6

                        possible_moves[idx + 0, 0] = i1
                        possible_moves[idx + 0, 1] = j1
                        possible_moves[idx + 0, 2] = i2
                        possible_moves[idx + 0, 3] = j2
                        possible_moves[idx + 0, 4] = 1

                        possible_moves[idx + 1, 0] = i2
                        possible_moves[idx + 1, 1] = j2
                        possible_moves[idx + 1, 2] = i1
                        possible_moves[idx + 1, 3] = j1
                        possible_moves[idx + 1, 4] = 1

    had_possible_moves = possible_moves[:, 4] == 1
    possible_moves = possible_moves[had_possible_moves]
    possible_moves = possible_moves[:, :4]

    return possible_moves # filter duplicates after combining with colours

@njit(cache=True)
def fast_calculate_move_score(move, scores):
    move_score = np.zeros(6, dtype=np.uint8)
    i1, j1, i2, j2, c1, c2 = move
    move_score[c1] += scores[i1, j1, 3, c1]
    move_score[c2] += scores[i2, j2, 3, c2]
    return move_score

@njit(parallel=True, fastmath=True, cache=True)
def fast_batch_calculate_move_score(moves, scores):
    num_moves = moves.shape[0]
    move_scores = np.zeros((num_moves, 6), dtype=np.uint8)

    for idx in prange(num_moves):
        move_scores[idx] = fast_calculate_move_score(moves[idx], scores)

    return move_scores

@njit(parallel=True, fastmath=True, cache=True)
def fast_batch_get_updated_playable(moves, height, width, playable, neighbour_offsets):
    num_moves = moves.shape[0]
    updated_playables = np.zeros((num_moves, height + 2, width + 4), dtype=np.uint8)

    for move_idx in prange(num_moves):
        coords1 = moves[move_idx, 0:2]
        coords2 = moves[move_idx, 2:4]
        updated_playables[move_idx] = playable
        updated_playables[move_idx] = fast_update_playable_for_hex(updated_playables[move_idx], coords1, neighbour_offsets)
        updated_playables[move_idx] = fast_update_playable_for_hex(updated_playables[move_idx], coords2, neighbour_offsets)

    return updated_playables

@njit(parallel=True, fastmath=True, cache=True)
def fast_batch_get_updated_available(moves, updated_playables, height, width, available, neighbour_offsets):
    num_moves = moves.shape[0]
    updated_availables = np.zeros((num_moves, height + 2, width + 4), dtype=np.uint8)

    for move_idx in prange(num_moves):
        coords1 = moves[move_idx, 0:2]
        coords2 = moves[move_idx, 2:4]
        updated_availables[move_idx] = available
        updated_playable = updated_playables[move_idx]
        updated_availables[move_idx] = fast_update_available_for_hex(updated_availables[move_idx], coords1, neighbour_offsets, updated_playable)
        updated_availables[move_idx] = fast_update_available_for_hex(updated_availables[move_idx], coords2, neighbour_offsets, updated_playable)

    return updated_availables

@njit(parallel=True, fastmath=True, cache=True)
def fast_batch_get_updated_states(moves, state, height, width, updated_playables, updated_available):
    num_moves = moves.shape[0]
    updated_states = np.zeros((num_moves, height + 2, width + 4, 11), dtype=np.uint8)

    for move_idx in prange(num_moves):
        updated_states[move_idx, :, :, :9] = state

        coords1 = moves[move_idx, 0:2]
        colour1 = moves[move_idx, 4]
        updated_states[move_idx, :, :, :9] = fast_update_state_for_hex(updated_states[move_idx, :, :, :9], coords1, colour1)

        coords2 = moves[move_idx, 2:4]
        colour2 = moves[move_idx, 5]
        updated_states[move_idx, :, :, :9] = fast_update_state_for_hex(updated_states[move_idx, :, :, :9], coords2, colour2)

        updated_states[move_idx, :, :, 9] = updated_playables[move_idx]
        updated_states[move_idx, :, :, 10] = updated_available[move_idx]

    return updated_states

@njit(cache=True)
def fast_check_if_legal(move, possible_moves):
    for i in range(possible_moves.shape[0]):
        for j in range(4):
            if move[j] != possible_moves[i, j]:
                continue
            return True
    return False

@njit(cache=True)
def fast_remove_available_around_hex(coords, available, offsets):
    i0, j0 = coords
    to_remove = np.zeros((6, 3), dtype=np.uint8)

    for idx in range(6):
        di, dj = offsets[idx]
        i1 = i0 + di
        j1 = j0 + dj

        if available[i1, j1] == 1:
            to_remove[idx, 0] = i1
            to_remove[idx, 1] = j1
            to_remove[idx, 2] = 1

    return to_remove

@njit(cache=True)
def fast_remove_first_available(available, start_coords1, start_coords2, start_hexes, offsets):
    max_available = 3 * 6
    removed_available = np.zeros((max_available, 3), dtype=np.uint8)

    third_hex_coords = None

    i1, j1 = start_coords1
    for o in range(6):
        di, dj = offsets[o]
        i2 = i1 + di
        j2 = j1 + dj

        for s in range(6):
            start_hex = start_hexes[s]
            if i2 == start_hex[0] and j2 == start_hex[1]:
                third_hex_coords = start_hex

    if third_hex_coords is None:
        i1, j1 = start_coords2
        for o in range(6):
            di, dj = offsets[o]
            i2 = i1 + di
            j2 = j1 + dj

            for s in range(6):
                start_hex = start_hexes[s]
                if i2 == start_hex[0] and j2 == start_hex[1]:
                    third_hex_coords = start_hex

    removed_available[0:6, :] = fast_remove_available_around_hex(start_coords1, available, offsets)
    removed_available[6:12, :] = fast_remove_available_around_hex(start_coords2, available, offsets)
    removed_available[12:18, :] = fast_remove_available_around_hex(third_hex_coords, available, offsets)

    for idx in range(max_available):
        if removed_available[idx, 2] == 1:
            i, j = removed_available[idx, :2]
            available[i, j] = 0

    return available, removed_available

@njit(cache=True)
def fast_add_first_available_back(available, removed_available):
    for idx in range(3 * 6):
        if removed_available[idx, 2] == 1:
            i, j = removed_available[idx, :2]
            available[i, j] = 1

    return available

@njit(cache=True)
def fast_update_state_for_hex(state, coords, colour):
    i, j = coords
    state[i, j, colour] = 1
    state[i, j, 6] = 1
    return state

@njit(cache=True)
def fast_count_scores(colour_channel, top_k):
    out = np.zeros((top_k), dtype=np.uint8)

    colour_channel_flat = colour_channel.flatten()
    colour_channel_flat.sort()

    for k in range(top_k):
        out[k] = colour_channel_flat[-1 - k]

    return out

@njit(cache=True)
def fast_generate_vector_representation(playable, available, state, scores, height, width):
    colour_counts = np.zeros((6,), dtype=np.uint8)
    score_counts = np.zeros((3 + 8, 6), dtype=np.uint8)
    vector_repr = np.zeros((6 + (3 + 8) * 6 + 2))

    for c in prange(6):
        colour_counts_sum = 0
        score_counts_sum_0 = 0
        score_counts_sum_1 = 0
        score_counts_sum_2 = 0

        for i in range(1, height + 1):
            for j in range(2, width + 2):
                colour_counts_sum += state[i, j, c]
                score_counts_sum_1 += scores[i, j, 3, c]

                if scores[i, j, 3, c] > 0:
                    score_counts_sum_0 += 1

                if available[i, j] == 1 and scores[i, j, 3, c] == 0:
                    score_counts_sum_2 += 1

        colour_counts[c] = colour_counts_sum
        score_counts[0, c] = score_counts_sum_0
        score_counts[1, c] = score_counts_sum_1
        score_counts[2, c] = score_counts_sum_2

        score_counts[3:, c] = fast_count_scores(scores[:, :, 3, c], 8)

    num_playable = 0
    num_available = 0
    for i in range(1, height + 1):
        for j in range(2, width + 2):
            num_playable += playable[i, j]
            num_available += available[i, j]

    vector_repr[0] = num_playable
    vector_repr[1] = num_available
    vector_repr[2:2+6] = colour_counts
    vector_repr[2+6:] = score_counts.flatten()

    return vector_repr

@njit(parallel=True, fastmath=True, cache=True)
def fast_batch_generate_vector_representation(updated_playable, updated_available, updated_states, updated_scores, height, width):
    batch_size = updated_playable.shape[0]
    updated_board_vecs = np.zeros((batch_size, 74), dtype=np.uint8)

    for idx in prange(batch_size):
        updated_board_vecs[idx] = fast_generate_vector_representation(
            updated_playable[idx],
            updated_available[idx],
            updated_states[idx],
            updated_scores[idx],
            height,
            width
            )

    return updated_board_vecs # b x 74

@njit(cache=True)
def fast_update_scoring_arrays_for_hex(coords, colour, clusters, sizes, scores, playable, all_directions, height, width):
    i0, j0 = coords

    for d in range(3):
        directions = all_directions[d]
        di1, dj1 = directions[0]
        di2, dj2 = directions[1]

        i1 = i0 + di1
        j1 = j0 + dj1
        i2 = i0 + di2
        j2 = j0 + dj2

        cluster1 = clusters[i1, j1, d, colour]
        cluster2 = clusters[i2, j2, d, colour]

        if cluster1 == 0 and cluster2 == 0:
            max_cluster_num = 0
            for i in range(1, height + 1):
                for j in range(2, width + 2):
                    if clusters[i, j, d, colour] > max_cluster_num:
                        max_cluster_num = clusters[i, j, d, colour]

            new_cluster_num = max_cluster_num + 1
            clusters[i0, j0, d, colour] = new_cluster_num
            sizes[i0, j0, d, colour] = 1

        elif cluster1 > 0 and cluster2 == 0:
            new_cluster_num = cluster1
            clusters[i0, j0, d, colour] = new_cluster_num
            new_cluster_size = sizes[i1, j1, d, colour] + 1

            for i in range(1, height + 1):
                for j in range(2, width + 2):
                    if clusters[i, j, d, colour] == new_cluster_num:
                        sizes[i, j, d, colour] = new_cluster_size

        elif cluster1 == 0 and cluster2 > 0:
            new_cluster_num = cluster2
            clusters[i0, j0, d, colour] = new_cluster_num
            new_cluster_size = sizes[i2, j2, d, colour] + 1

            for i in range(1, height + 1):
                for j in range(2, width + 2):
                    if clusters[i, j, d, colour] == new_cluster_num:
                        sizes[i, j, d, colour] = new_cluster_size

        else:
            if cluster1 < cluster2:
                new_cluster_num = cluster1
                other_cluster_num = cluster2
            else:
                new_cluster_num = cluster2
                other_cluster_num = cluster1

            clusters[i0, j0, d, colour] = new_cluster_num

            for i in range(1, height + 1):
                for j in range(2, width + 2):
                    if clusters[i, j, d, colour] == other_cluster_num:
                        clusters[i, j, d, colour] = new_cluster_num

            new_cluster_size = sizes[i1, j1, d, colour] + sizes[i2, j2, d, colour] + 1

            for i in range(1, height + 1):
                for j in range(2, width + 2):
                    if clusters[i, j, d, colour] == new_cluster_num:
                        sizes[i, j, d, colour] = new_cluster_size

    for d in range(3):
        directions = all_directions[d]

        for d2 in range(2):
            di, dj = directions[d2]
            i1 = i0 + di
            j1 = j0 + dj

            while sizes[i1, j1, d, colour] > 0:
                i1 += di
                j1 += dj

            if playable[i1, j1] == 1:
                i2 = i1 + di
                j2 = j1 + dj

                scores[i1, j1, d, colour] = sizes[i0, j0, d, colour] + sizes[i2, j2, d, colour]

    for i in range(1, height + 1):
        for j in range(2, width + 2):
            if playable[i, j] == 1:
                for c in range(6):
                    scores[i, j, 3, c] = scores[i, j, 0, c] + scores[i, j, 1, c] + scores[i, j, 2, c]
            else:
                scores[i, j, :, :] = 0

    return clusters, sizes, scores

# @njit(parallel=True, fastmath=True, cache=True)
# def fast_batch_get_updated_scoring_arrays(moves, updated_playables, height, width, all_directions, clusters, sizes, scores):
#     num_moves = moves.shape[0]
#     updated_clusters = np.zeros((num_moves, height + 2, width + 4, 3, 6), dtype=np.uint8)
#     updated_sizes = np.zeros((num_moves, height + 2, width + 4, 3, 6), dtype=np.uint8)
#     updated_scores = np.zeros((num_moves, height + 2, width + 4, 4, 6), dtype=np.uint8)

#     for move_idx in prange(num_moves):
#         coords1 = moves[move_idx, 0:2]
#         coords2 = moves[move_idx, 2:4]
#         colour1 = moves[move_idx, 4]
#         colour2 = moves[move_idx, 5]
#         updated_playable = updated_playables[move_idx]
#         updated_clusters[move_idx] = clusters
#         updated_sizes[move_idx] = sizes
#         updated_scores[move_idx] = scores
#         updated_clusters[move_idx], updated_sizes[move_idx], updated_scores[move_idx] = fast_update_scoring_arrays_for_hex(
#             coords1, colour1, updated_clusters[move_idx], updated_sizes[move_idx], updated_scores[move_idx], updated_playable, all_directions, height, width)
#         updated_clusters[move_idx], updated_sizes[move_idx], updated_scores[move_idx] = fast_update_scoring_arrays_for_hex(
#             coords2, colour2, updated_clusters[move_idx], updated_sizes[move_idx], updated_scores[move_idx], updated_playable, all_directions, height, width)

#     return updated_clusters, updated_sizes, updated_scores

@njit(parallel=True, fastmath=True, cache=True)
def fast_batch_get_updated_scoring_arrays(moves, updated_playables, height, width, all_directions, clusters_original, sizes_original, scores_original):
    num_moves = moves.shape[0]
    updated_clusters = np.zeros((num_moves, height + 2, width + 4, 3, 6), dtype=np.uint8)
    updated_sizes = np.zeros((num_moves, height + 2, width + 4, 3, 6), dtype=np.uint8)
    updated_scores = np.zeros((num_moves, height + 2, width + 4, 4, 6), dtype=np.uint8)

    for move_idx in prange(num_moves):
        coords1 = moves[move_idx, 0:2]
        coords2 = moves[move_idx, 2:4]
        colour1 = moves[move_idx, 4]
        colour2 = moves[move_idx, 5]

        playable = updated_playables[move_idx]
        updated_clusters[move_idx] = clusters_original
        updated_sizes[move_idx] = sizes_original
        updated_scores[move_idx] = scores_original

        clusters = updated_clusters[move_idx]
        sizes = updated_sizes[move_idx]
        scores = updated_scores[move_idx]

        for coords, colour in zip((coords1, coords2), (colour1, colour2)):
            i0, j0 = coords

            for d in range(3):
                directions = all_directions[d]
                di1, dj1 = directions[0]
                di2, dj2 = directions[1]

                i1 = i0 + di1
                j1 = j0 + dj1
                i2 = i0 + di2
                j2 = j0 + dj2

                cluster1 = clusters[i1, j1, d, colour]
                cluster2 = clusters[i2, j2, d, colour]

                if cluster1 == 0 and cluster2 == 0:
                    max_cluster_num = 0
                    for i in range(1, height + 1):
                        for j in range(2, width + 2):
                            if clusters[i, j, d, colour] > max_cluster_num:
                                max_cluster_num = clusters[i, j, d, colour]

                    new_cluster_num = max_cluster_num + 1
                    clusters[i0, j0, d, colour] = new_cluster_num
                    sizes[i0, j0, d, colour] = 1

                elif cluster1 > 0 and cluster2 == 0:
                    new_cluster_num = cluster1
                    clusters[i0, j0, d, colour] = new_cluster_num
                    new_cluster_size = sizes[i1, j1, d, colour] + 1

                    for i in range(1, height + 1):
                        for j in range(2, width + 2):
                            if clusters[i, j, d, colour] == new_cluster_num:
                                sizes[i, j, d, colour] = new_cluster_size

                elif cluster1 == 0 and cluster2 > 0:
                    new_cluster_num = cluster2
                    clusters[i0, j0, d, colour] = new_cluster_num
                    new_cluster_size = sizes[i2, j2, d, colour] + 1

                    for i in range(1, height + 1):
                        for j in range(2, width + 2):
                            if clusters[i, j, d, colour] == new_cluster_num:
                                sizes[i, j, d, colour] = new_cluster_size

                else:
                    if cluster1 < cluster2:
                        new_cluster_num = cluster1
                        other_cluster_num = cluster2
                    else:
                        new_cluster_num = cluster2
                        other_cluster_num = cluster1

                    clusters[i0, j0, d, colour] = new_cluster_num

                    for i in range(1, height + 1):
                        for j in range(2, width + 2):
                            if clusters[i, j, d, colour] == other_cluster_num:
                                clusters[i, j, d, colour] = new_cluster_num

                    new_cluster_size = sizes[i1, j1, d, colour] + sizes[i2, j2, d, colour] + 1

                    for i in range(1, height + 1):
                        for j in range(2, width + 2):
                            if clusters[i, j, d, colour] == new_cluster_num:
                                sizes[i, j, d, colour] = new_cluster_size

            for d in range(3):
                directions = all_directions[d]

                for d2 in range(2):
                    di, dj = directions[d2]
                    i1 = i0 + di
                    j1 = j0 + dj

                    while sizes[i1, j1, d, colour] > 0:
                        i1 += di
                        j1 += dj

                    if playable[i1, j1] == 1:
                        i2 = i1 + di
                        j2 = j1 + dj

                        scores[i1, j1, d, colour] = sizes[i0, j0, d, colour] + sizes[i2, j2, d, colour]

            for i in range(1, height + 1):
                for j in range(2, width + 2):
                    if playable[i, j] == 1:
                        for c in range(6):
                            scores[i, j, 3, c] = scores[i, j, 0, c] + scores[i, j, 1, c] + scores[i, j, 2, c]
                    else:
                        scores[i, j, :, :] = 0

        updated_clusters[move_idx] = clusters
        updated_sizes[move_idx] = sizes
        updated_scores[move_idx] = scores

    return updated_clusters, updated_sizes, updated_scores