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
        self.possible_moves = fast_get_all_possible_moves(self.playable, self.available, self.offsets)

    def get_possible_moves(self):
        return self.possible_moves

    def game_is_finished(self):
        return self.get_possible_moves().shape[0] == 0

    def check_move_is_legal(self, move):
        return np.any(fast_all_axis(np.expand_dims(move[0:4], 0) == self.get_possible_moves(), 1))

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
            self.available, move[0:2], move[2:4], self.offsets)

    def update_available_for_second_move(self, move):
        hexes = ((move[0:2], move[4]), (move[2:4], move[5]))
        self.update_available_for_hexes(hexes)
        self.available = fast_add_first_available_back(self.available, self.tmp_removed_available)

    def calculate_move_score(self, move):
        return fast_calculate_move_score(move, self.scores)

    def generate_state_representation(self):
        return fast_generate_state_representation(self.playable, self.available, self.scores, self.state)

    def generate_vector_representation(self):
        return fast_generate_vector_representation(self.playable, self.available, self.state, self.scores)

    def batch_calculate_move_scores(self, moves):
        return fast_batch_calculate_move_score(moves, self.scores)

    def batch_generate_vector_representation(self, updated_playable, updated_available, updated_states, updated_scores):
        return fast_batch_generate_vector_representation(updated_playable, updated_available, updated_states, updated_scores)

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

@njit(cache=True)
def combine_moves_and_deck(possible_moves, deck):
    num_possible_moves = possible_moves.shape[0]
    num_tiles_in_deck = deck.shape[0]
    num_combinations = num_possible_moves * num_tiles_in_deck

    combined = np.zeros((num_combinations, 6), dtype=np.uint8)
    for i in range(num_possible_moves):
        for j in range(num_tiles_in_deck):
            idx = num_tiles_in_deck * i + j
            combined[idx, 0:4] = possible_moves[i]
            combined[idx, 4:6] = deck[j]

    combined_filtered = fast_remove_duplicates(combined)

    return combined_filtered

@njit(cache=True)
def fast_update_playable_for_hex(playable, coords, neighbour_offsets):
    i1, j1 = coords
    playable[i1, j1] = 0

    for idx_1 in range(6):
        di1, dj1 = neighbour_offsets[idx_1].flatten()
        i2 = i1 + di1
        j2 = j1 + dj1
        if playable[i2, j2] == 0:
            continue

        has_neighbours = False
        for idx_2 in range(6):
            di2, dj2 = neighbour_offsets[idx_2].flatten()
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
        di, dj = neighbour_offsets[o_idx].flatten()
        i2 = i1 + di
        j2 = j1 + dj
        if playable[i2, j2] == 1:
            available[i2, j2] = 1

    return available

@njit(cache=True)
def fast_get_all_possible_moves(playable, available, neighbour_offsets):
    possible_moves = np.zeros((0, 4), dtype=np.uint8)
    available_coords = np.where(available == 1)
    num_available = available_coords[0].shape[0]

    for idx in range(num_available):
        i1 = available_coords[0][idx]
        j1 = available_coords[1][idx]

        for o_idx in range(6):
            di, dj = neighbour_offsets[o_idx].flatten()
            i2 = i1 + di
            j2 = j1 + dj
            if playable[i2, j2] == 1:
                possible_move_1 = np.array((i1, j1, i2, j2), dtype=np.uint8).reshape(1, 4)
                possible_move_2 = np.array((i2, j2, i1, j1), dtype=np.uint8).reshape(1, 4)
                possible_moves = np.concatenate((possible_moves, possible_move_1, possible_move_2))

    return possible_moves # filter duplicates after combining with colours

@njit(cache=True)
def fast_all_axis(array, axis):
    array_shape = array.shape
    count = np.sum(array, axis=axis)
    return count == array_shape[axis]

@njit(cache=True)
def fast_remove_duplicates(moves):
    filtered_moves = np.zeros((0, moves.shape[1]), dtype=np.uint8)
    num_moves = moves.shape[0]

    for idx in range(num_moves - 1):
        entry = moves[idx:idx+1, :]
        match = np.any(fast_all_axis(moves[idx+1:num_moves, :] == entry, 1))
        if not match:
            filtered_moves = np.concatenate((filtered_moves, entry))
    
    filtered_moves = np.concatenate((filtered_moves, np.expand_dims(moves[-1, :], 0)))

    return filtered_moves

@njit(cache=True)
def fast_where_idxs_2d(array):
    idxs_tuple = np.where(array)

    idxs_list = []
    for idx in range(2):
        idx_expanded = np.expand_dims(idxs_tuple[idx], 1)
        idxs_list.append(idx_expanded)

    idxs_stacked = np.hstack((idxs_list[0], idxs_list[1]))
    return idxs_stacked

@njit(cache=True)
def fast_mask_update(ref_array, ref_value, update_array, update_val):
    idxs = fast_where_idxs_2d(ref_array == ref_value)

    for idx in range(idxs.shape[0]):
        i, j = idxs[idx]
        update_array[i, j] = update_val

    return update_array

@njit(cache=True)
def fast_update_scoring_arrays_for_hex(coords, colour, clusters, sizes, scores, playable, all_directions, height, width):
    i0, j0 = coords

    for d_idx in range(3):
        directions = all_directions[d_idx]
        di1, dj1 = directions[0].flatten()
        di2, dj2 = directions[1].flatten()

        i1 = i0 + di1
        j1 = j0 + dj1
        i2 = i0 + di2
        j2 = j0 + dj2

        cluster1 = clusters[i1, j1, d_idx, colour]
        cluster2 = clusters[i2, j2, d_idx, colour]

        if cluster1 == 0 and cluster2 == 0:
            new_cluster_num = np.max(clusters) + 1
            clusters[i0, j0, d_idx, colour] = new_cluster_num
            sizes[i0, j0, d_idx, colour] = 1

        elif cluster1 > 0 and cluster2 == 0:
            new_cluster_num = cluster1
            clusters[i0, j0, d_idx, colour] = new_cluster_num
            new_cluster_size = sizes[i1, j1, d_idx, colour] + 1
            fast_mask_update(clusters[..., d_idx, colour], new_cluster_num, sizes[..., d_idx, colour], new_cluster_size)

        elif cluster1 == 0 and cluster2 > 0:
            new_cluster_num = cluster2
            clusters[i0, j0, d_idx, colour] = new_cluster_num
            new_cluster_size = sizes[i2, j2, d_idx, colour] + 1
            fast_mask_update(clusters[..., d_idx, colour], new_cluster_num, sizes[..., d_idx, colour], new_cluster_size)

        else:
            new_cluster_num = np.minimum(cluster1, cluster2)
            other_cluster_num = np.maximum(cluster1, cluster2)
            clusters[i0, j0, d_idx, colour] = new_cluster_num
            fast_mask_update(clusters[..., d_idx, colour], other_cluster_num, clusters[..., d_idx, colour], new_cluster_num)
            new_cluster_size = sizes[i1, j1, d_idx, colour] + sizes[i2, j2, d_idx, colour] + 1
            fast_mask_update(clusters[..., d_idx, colour], new_cluster_num, sizes[..., d_idx, colour], new_cluster_size)

    for d_idx in range(3):
        directions = all_directions[d_idx]

        for d_idx_2 in range(2):
            di, dj = directions[d_idx_2].flatten()
            i1 = i0 + di
            j1 = j0 + dj

            while sizes[i1, j1, d_idx, colour] > 0:
                i1 += di
                j1 += dj

            if playable[i1, j1] == 1:
                scores[i1, j1, d_idx, colour] = sizes[i0, j0, d_idx, colour]
                scores[i1, j1, 3, colour] = np.sum(scores[i1, j1, :3, colour])

    playable_reshaped = playable.reshape(height + 2, width + 4, 1, 1)
    scores *= playable_reshaped

    return clusters, sizes, scores

@njit(cache=True)
def fast_calculate_move_score(move, scores):
    move_score = np.zeros(6, dtype=np.uint8)
    i1, j1, i2, j2, c1, c2 = move.flatten()
    move_score[c1] += scores[i1, j1, 3, c1]
    move_score[c2] += scores[i2, j2, 3, c2]
    return move_score

@njit(parallel=True, fastmath=True, cache=True)
def fast_batch_calculate_move_score(moves, scores):
    num_moves = moves.shape[0]
    move_scores = np.zeros((num_moves, 6), dtype=np.uint8)

    for move_idx in prange(num_moves):
        move = moves[move_idx]
        move_score = fast_calculate_move_score(move, scores)
        move_scores[move_idx] = move_score

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
def fast_batch_get_updated_scoring_arrays(moves, updated_playables, height, width, all_directions, clusters, sizes, scores):
    num_moves = moves.shape[0]
    updated_clusters = np.zeros((num_moves, height + 2, width + 4, 3, 6), dtype=np.uint8)
    updated_sizes = np.zeros((num_moves, height + 2, width + 4, 3, 6), dtype=np.uint8)
    updated_scores = np.zeros((num_moves, height + 2, width + 4, 4, 6), dtype=np.uint8)

    for move_idx in prange(num_moves):
        coords1 = moves[move_idx, 0:2]
        coords2 = moves[move_idx, 2:4]
        colour1 = moves[move_idx, 4]
        colour2 = moves[move_idx, 5]
        updated_playable = updated_playables[move_idx]
        updated_clusters[move_idx] = clusters
        updated_sizes[move_idx] = sizes
        updated_scores[move_idx] = scores
        updated_clusters[move_idx], updated_sizes[move_idx], updated_scores[move_idx] = fast_update_scoring_arrays_for_hex(
            coords1, colour1, updated_clusters[move_idx], updated_sizes[move_idx], updated_scores[move_idx], updated_playable, all_directions, height, width)
        updated_clusters[move_idx], updated_sizes[move_idx], updated_scores[move_idx] = fast_update_scoring_arrays_for_hex(
            coords2, colour2, updated_clusters[move_idx], updated_sizes[move_idx], updated_scores[move_idx], updated_playable, all_directions, height, width)

    return updated_clusters, updated_sizes, updated_scores

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
def fast_concat_channels_3d(array1, array2):
    shape = array1.shape
    output = np.zeros((0, shape[0], shape[1]))

    array1_T = np.transpose(array1, (2, 0, 1))
    array2_T = np.transpose(array2, (2, 0, 1))
    output = np.concatenate((array1_T, array2_T))

    return np.transpose(output, (1, 2, 0))

@njit(cache=True)
def fast_in(item, array):
    return np.any(fast_all_axis(item == array, 1))

@njit(cache=True)
def fast_remove_available_around_hex(coords, available, tmp_removed_available, neighbour_offsets):
    i0, j0 = coords

    for idx_2 in range(6):
        neighbour_offset = neighbour_offsets[idx_2]
        di, dj = neighbour_offset.flatten()
        i1 = i0 + di
        j1 = j0 + dj

        if available[i1, j1] == 1:
            available[i1, j1] = 0
            removed_hex = np.array((i1, j1), dtype=np.uint8).reshape(1, 2)
            tmp_removed_available = np.concatenate((tmp_removed_available, removed_hex))

    return available, tmp_removed_available

@njit(cache=True)
def fast_remove_first_available(available, start_coords1, start_coords2, neighbour_offsets):
    tmp_removed_available = np.zeros((0, 2), dtype=np.uint8)
    start_hexes = np.array(((1, 7), (11, 7), (1, 17), (11, 17), (6, 2), (6, 22)), dtype=np.int32).reshape(6, 2)
    start_coords1 = start_coords1.astype(np.int32)
    start_coords2 = start_coords2.astype(np.int32)

    min_distance = 100
    closest_hex = None
    for idx in range(6):
        start_hex = start_hexes[idx]
        diff = np.sum(np.abs(start_coords1 - start_hex))
        if diff < min_distance:
            min_distance = diff
            closest_hex = start_hex

    available, tmp_removed_available = fast_remove_available_around_hex(start_coords1, available, tmp_removed_available, neighbour_offsets)
    available, tmp_removed_available = fast_remove_available_around_hex(start_coords2, available, tmp_removed_available, neighbour_offsets)
    available, tmp_removed_available = fast_remove_available_around_hex(closest_hex, available, tmp_removed_available, neighbour_offsets)

    return available, tmp_removed_available

@njit(cache=True)
def fast_add_first_available_back(available, tmp_removed_available):
    num_removed = tmp_removed_available.shape[0]
    for idx in prange(num_removed):
        i, j = tmp_removed_available[idx].flatten()
        available[i, j] = 1

    return available

@njit(cache=True)
def fast_update_state_for_hex(state, coords, colour):
    i, j = coords
    state[i, j, colour] = 1
    state[i, j, 6] = 1
    return state

# @njit(cache=True)
def fast_generate_state_representation(playable, available, scores, state):
    combined = fast_concat_channels_3d(np.expand_dims(playable, 2), np.expand_dims(available, 2))
    state_combined = fast_concat_channels_3d(state, combined)
    return state_combined, scores # h x w x 11, h x w x 4 x 6

@njit(cache=True)
def fast_count_scores(colour_channel, c, top_k):
    out = np.zeros((top_k), dtype=np.uint8)

    colour_channel_flat = colour_channel.flatten()
    colour_channel_flat.sort()

    for k in range(top_k):
        out[k] = colour_channel_flat[-1 - k]

    return out

@njit(cache=True)
def fast_generate_vector_representation(playable, available, state, scores):
    num_playable = np.sum(playable)
    num_available = np.sum(available)

    colour_counts = np.zeros((6,), dtype=np.uint8)
    score_counts = np.zeros((3 + 8, 6), dtype=np.uint8)

    for c in prange(6):
        colour_counts[c] = np.sum(state[..., c])

        score_counts[0, c] = np.sum(scores[:, :, 3, c] > 0)
        score_counts[1, c] = np.sum(scores[:, :, 3, c])
        score_counts[2, c] = int(np.sum(scores[:, :, 3, c]) == 0)
        score_counts[3:, c] = fast_count_scores(scores[:, :, 3, c], c, 8)

    vector_repr = np.concatenate((
        np.array((num_playable)).reshape(1,),
        np.array((num_available)).reshape(1,),
        colour_counts,
        score_counts.flatten()
    ))

    return vector_repr # 74

@njit(parallel=True, fastmath=True, cache=True)
def fast_batch_generate_vector_representation(updated_playable, updated_available, updated_states, updated_scores):
    batch_size = updated_playable.shape[0]
    updated_board_vecs = np.zeros((batch_size, 74), dtype=np.uint8)

    for idx in prange(batch_size):
        updated_board_vecs[idx] = fast_generate_vector_representation(
            updated_playable[idx],
            updated_available[idx],
            updated_states[idx],
            updated_scores[idx]
            )

    return updated_board_vecs # b x 74