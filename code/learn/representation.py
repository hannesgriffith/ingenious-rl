from numba import njit, prange
import numpy as np

from game.player import fast_batch_peek_can_exchange_tiles

class RepresentationsBuffer():
    def __init__(self):
        self.size = 0
        self.empty = 1

    def set_single_reprs_from_scratch(self, board_repr1, board_repr2, board_vec, deck_repr, scores_repr, general_repr, turn_of_repr, values_repr):
        self.board_repr1 = np.expand_dims(board_repr1, 0)
        self.board_repr2 = np.expand_dims(board_repr2, 0)
        self.board_vec = np.expand_dims(board_vec, 0)
        self.deck_repr = np.expand_dims(deck_repr, 0)
        self.scores_repr = np.expand_dims(scores_repr, 0)
        self.general_repr = np.expand_dims(general_repr, 0)
        self.values_repr = np.expand_dims(values_repr, 0)
        self.turn_of_repr = turn_of_repr
        self.size += 1
        self.empty = 0

    def set_batched_reprs_from_scratch(self, board_repr1, board_repr2, board_vec, deck_repr, scores_repr, general_repr, turn_of_repr, values_repr):
        self.board_repr1 = board_repr1
        self.board_repr2 = board_repr2
        self.board_vec = board_vec
        self.deck_repr = deck_repr
        self.scores_repr = scores_repr
        self.general_repr = general_repr
        self.values_repr = values_repr
        self.turn_of_repr = turn_of_repr
        self.size += board_repr1.shape[0]
        self.empty = 0

    def set_reprs_from_reprs(self, reprs):
        self.board_repr1 = reprs.board_repr1
        self.board_repr2 = reprs.board_repr2
        self.board_vec = reprs.board_vec
        self.deck_repr = reprs.deck_repr
        self.scores_repr = reprs.scores_repr
        self.general_repr = reprs.general_repr
        self.turn_of_repr = reprs.turn_of_repr
        self.values_repr = reprs.values_repr
        self.size += reprs.size
        self.empty = 0

    def combine_reprs(self, reprs):
        if self.empty == 1:
            self.set_reprs_from_reprs(reprs)
        else:
            self.board_repr1 = np.concatenate((reprs.board_repr1, self.board_repr1))
            self.board_repr2 = np.concatenate((reprs.board_repr2, self.board_repr2))
            self.board_vec = np.concatenate((reprs.board_vec, self.board_vec))
            self.deck_repr = np.concatenate((reprs.deck_repr, self.deck_repr))
            self.scores_repr = np.concatenate((reprs.scores_repr, self.scores_repr))
            self.general_repr = np.concatenate((reprs.general_repr, self.general_repr))
            self.turn_of_repr = np.concatenate((reprs.turn_of_repr, self.turn_of_repr))
            self.values_repr = np.concatenate((reprs.values_repr, self.values_repr))
            self.size += reprs.size

    def clip_to_size(self, required_size):
        self.board_repr1 = self.board_repr1[:required_size]
        self.board_repr2 = self.board_repr2[:required_size]
        self.board_vec = self.board_vec[:required_size]
        self.deck_repr = self.deck_repr[:required_size]
        self.scores_repr = self.scores_repr[:required_size]
        self.general_repr = self.general_repr[:required_size]
        self.turn_of_repr = self.turn_of_repr[:required_size]
        self.values_repr = self.values_repr[:required_size]
        self.size = required_size

    def get_examples_by_idxs(self, idxs):
        x = (self.board_repr1[idxs], self.board_repr2[idxs], self.board_vec[idxs], self.deck_repr[idxs], self.scores_repr[idxs], self.general_repr[idxs])
        y = self.values_repr[idxs, 0]
        return x, y

    @staticmethod
    def preprocess(inputs):
        return fast_preprocess(inputs)

class RepresentationGenerator:

    @staticmethod
    def generate(board, deck, score, other_score, ingenious, num_ingenious, can_exchange, should_exchange, turn_of):
        board_repr1, board_repr2 = board.generate_state_representation() # h x w x 11, h x w x 4 x 6
        board_vec = board.generate_vector_representation() # 11
        deck_repr = deck.get_state_copy() # 2 x 6 ([single tiles, double tiles] x [6 colours])
        your_score_repr = score.get_score_copy()
        other_score_repr = other_score.get_score_copy()
        board_repr1, board_repr2, board_vec, deck_repr, scores_repr, general_repr, values_repr = fast_generate(board_repr1, board_repr2, board_vec, deck_repr, 
            your_score_repr, other_score_repr, ingenious, num_ingenious, can_exchange, should_exchange, board.move_num)

        new_reprs_buffer = RepresentationsBuffer()
        new_reprs_buffer.set_single_reprs_from_scratch(board_repr1, board_repr2, board_vec, deck_repr, scores_repr, general_repr, turn_of, values_repr)

        return new_reprs_buffer

    @staticmethod
    def generate_batched(board, deck, score, other_score, turn_of, possible_moves):
        b = possible_moves.shape[0] # possible moves shape: b x 8

        board_repr1, board_repr2, board_vec = board.batch_generate_state_representations(possible_moves) # b x h x w x 11, b x h x w x 4 x 6, b x ?

        deck_repr = deck.batch_peek_next_states(possible_moves[:, 4:6]) # b x 2 x 6
        move_scores = board.batch_calculate_move_scores(possible_moves) # b x 6
        updated_scores, ingenious, num_ingenious = score.batch_peek_next_scores(move_scores) # b x 6, b, b

        next_decks = deck.batch_peek_next_decks(possible_moves[:, 4:6]) # b x 6 x 2
        can_exchange = fast_batch_peek_can_exchange_tiles(next_decks, updated_scores) # b

        other_score = other_score.get_score().flatten()
        scores_repr = np.zeros((b, 2, 6), dtype=np.uint8) # b x 2 x 6

        results = fast_generate_batched(board_repr1, board_repr2, board_vec, deck_repr, scores_repr, other_score, turn_of, possible_moves, ingenious, num_ingenious, can_exchange, updated_scores, board.move_num)
        board_repr1_subset, board_repr2_subset, board_vec_subset, deck_repr_subset, scores_repr_subset, general_repr_subset, turn_of_repr, values_repr, possible_moves_subset = results

        new_reprs_buffer = RepresentationsBuffer()
        new_reprs_buffer.set_batched_reprs_from_scratch(board_repr1_subset, board_repr2_subset, board_vec_subset, deck_repr_subset, scores_repr_subset, general_repr_subset, turn_of_repr, values_repr)

        return new_reprs_buffer, possible_moves_subset

def fast_generate(board_repr1, board_repr2, board_vec, deck_repr, your_score_repr, other_score_repr, ingenious, num_ingenious, can_exchange, should_exchange, move_num):
    scores_repr = np.vstack((
        np.expand_dims(your_score_repr, 0),
        np.expand_dims(other_score_repr, 0)
        )) # 2 x 6

    general_repr = np.array((
        ingenious,
        num_ingenious,
        can_exchange,
        should_exchange * can_exchange, # always 0 if can't exchange
        move_num
        ), dtype=np.uint8) # 5

    values_repr = np.zeros(2, dtype=np.uint8)

    return board_repr1, board_repr2, board_vec, deck_repr, scores_repr, general_repr, values_repr

@njit(parallel=True, fastmath=True, cache=True)
def fast_update_scores_repr(b, scores_repr, updated_scores, other_score):
    for idx in prange(b):
        scores_repr[idx, 0, :] = updated_scores[idx]
        scores_repr[idx, 1, :] = other_score
    return scores_repr

def fast_generate_batched(board_repr1, board_repr2, board_vec, deck_repr, scores_repr, other_score, turn_of, possible_moves, ingenious, num_ingenious, can_exchange, updated_scores, move_num):
    b = possible_moves.shape[0] # possible moves shape: b x 8

    scores_repr = fast_update_scores_repr(b, scores_repr, updated_scores, other_score)

    general_repr_dont_exchange = np.hstack((
        np.expand_dims(ingenious, 1),
        np.expand_dims(num_ingenious, 1),
        np.expand_dims(can_exchange, 1),
        np.zeros((b, 1), dtype=np.uint8),
        np.ones((b, 1), dtype=np.uint8) * move_num
        ))

    general_repr_do_exchange = np.hstack((
        np.expand_dims(ingenious, 1),
        np.expand_dims(num_ingenious, 1),
        np.expand_dims(can_exchange, 1),
        np.ones((b, 1), dtype=np.uint8),
        np.ones((b, 1), dtype=np.uint8) * move_num
        ))

    possible_moves_stacked = np.concatenate((possible_moves, possible_moves))
    board_repr1_stacked = np.concatenate((board_repr1, board_repr1))
    board_repr2_stacked = np.concatenate((board_repr2, board_repr2))
    board_vec_stacked = np.concatenate((board_vec, board_vec))
    deck_repr_stacked = np.concatenate((deck_repr, deck_repr))
    scores_repr_stacked = np.concatenate((scores_repr, scores_repr))
    general_repr_stacked = np.concatenate((general_repr_dont_exchange, general_repr_do_exchange))

    can_exchange_stacked = np.concatenate((can_exchange, can_exchange))
    should_exchange_stacked = np.concatenate((np.zeros(b, dtype=np.uint8), np.ones(b, dtype=np.uint8)))
    valid_idxs = np.where(((can_exchange_stacked == 0) & (should_exchange_stacked == 0)) | (can_exchange_stacked == 1))[0]

    possible_moves_subset = possible_moves_stacked[valid_idxs].astype(np.uint8)
    board_repr1_subset = board_repr1_stacked[valid_idxs].astype(np.uint8)
    board_repr2_subset = board_repr2_stacked[valid_idxs].astype(np.uint8)
    board_vec_subset = board_vec_stacked[valid_idxs].astype(np.uint8)
    deck_repr_subset = deck_repr_stacked[valid_idxs].astype(np.uint8)
    scores_repr_subset = scores_repr_stacked[valid_idxs].astype(np.int32)
    general_repr_subset = general_repr_stacked[valid_idxs].astype(np.uint8)

    turn_of_repr = np.full(valid_idxs.shape[0], turn_of, dtype=np.uint8)
    values_repr = np.zeros((valid_idxs.shape[0], 2), dtype=np.uint8)

    return (board_repr1_subset,
            board_repr2_subset,
            board_vec_subset,
            deck_repr_subset,
            scores_repr_subset,
            general_repr_subset,
            turn_of_repr,
            values_repr,
            possible_moves_subset)

@njit(parallel=True, fastmath=True, cache=True)
def fast_augment(board_repr1, board_repr2, board_vec, deck_repr, scores_repr, general_repr):
    n = board_repr1.shape[0]
    ordering = np.array((0, 1, 2, 3, 4, 5)).astype(np.uint8)

    for i in prange(n):
        ordering_copy = np.copy(ordering)
        np.random.shuffle(ordering_copy)

        states = board_repr1[i, :, :, :6]
        board_repr1[i, :, :, :6] = states[:, :, ordering_copy]

        scores = board_repr2[i]
        board_repr2[i] = scores[:, :, :, ordering_copy]

        colour_counts = board_vec[i, 2:8]
        board_vec[i, 2:8] = colour_counts[ordering_copy]

        scores_count = board_vec[i, 8:].reshape(3 + 8, 6)
        board_vec[i, 8:] = scores_count[:, ordering_copy].flatten()

        deck_repr_example = deck_repr[i]
        deck_repr[i] = deck_repr_example[:, ordering_copy]

        scores_repr_example = scores_repr[i]
        scores_repr[i] = scores_repr_example[:, ordering_copy]

    return (board_repr1, board_repr2, board_vec, deck_repr, scores_repr, general_repr)

def fast_random_transformations(board_repr1, board_repr2, board_vec, deck_repr, scores_repr, general_repr):
    flip = np.random.randint(0, high=2, size=3)

    if flip[0]:
        board_repr1[:, ::-1, :, :]
        board_repr2[:, ::-1, :, :, :]

    if flip[1]:
        board_repr1[:, :, ::-1, :]
        board_repr2[:, :, ::-1, :, :]

    if flip[2]:
        board_repr1 = np.rot90(board_repr1, k=2, axes=(1, 2))
        board_repr2 = np.rot90(board_repr2, k=2, axes=(1, 2))

    return (board_repr1, board_repr2, board_vec, deck_repr, scores_repr, general_repr)

@njit(parallel=True, fastmath=True, cache=True)
def fast_normalise(board_repr1, board_repr2, board_vec, deck_repr, scores_repr, general_repr):
    for i in prange(board_repr1.shape[0]):
        board_repr1[i, :, :, 8] /= 6.

        board_repr2[i, :, :, 0, :] /= 5.
        board_repr2[i, :, :, 1, :] /= 5.
        board_repr2[i, :, :, 2, :] /= 5.
        board_repr2[i, :, :, 3, :] /= 9.

        board_vec[i, 0] /= 85. # num playable
        board_vec[i, 1] /= 21. # num available
        board_vec[i, 2:8] /= 21. # colour counts
        board_vec[i, 8:8+6] /= 21. # playable colours
        board_vec[i, 8+6:8+12] /= 45. # total scores
        board_vec[i, 8+18:] /= 9. # colour scores

        deck_repr[i] /= 4.
        scores_repr[i] /= 18.
        general_repr[i, 1] /= 2.
        general_repr[i, 4] /= 40.

    return (board_repr1, board_repr2, board_vec, deck_repr, scores_repr, general_repr)

def fast_prepare_vector_input(board_vec_flat, deck_repr, scores_repr, general_repr_flat):
    b = board_vec_flat.shape[0]
    deck_repr_flat = deck_repr.reshape(b, -1)
    scores_repr_flat = scores_repr.reshape(b, -1)
    vector_for_grid = np.hstack((deck_repr_flat, scores_repr_flat, general_repr_flat))
    vector_input = np.hstack((board_vec_flat, vector_for_grid))
    return vector_for_grid, vector_input

def fast_prepare_grid_input(board_repr1, board_repr2):
    b = board_repr1.shape[0]
    board_repr2_reshaped = board_repr2.reshape(b, 11 + 2, 21 + 4, 4 * 6)
    combined = np.concatenate((board_repr1, board_repr2_reshaped), axis=3)
    combined = np.transpose(combined, (0, 3, 1, 2))
    return combined

def fast_add_score_features(scores_repr):
    b = scores_repr.shape[0]
    scores_repr_updated = np.zeros((b, 3, 6))
    scores_repr_updated[:, :2, :] = scores_repr
    scores_repr_updated[:, 2, :] = scores_repr[:, 0, :] - scores_repr[:, 1, :]
    return scores_repr_updated

def fast_prepare(board_repr1, board_repr2, board_vec, deck_repr, scores_repr, general_repr):
    scores_repr = fast_add_score_features(scores_repr)
    vector_for_grid, vector_input = fast_prepare_vector_input(board_vec, deck_repr, scores_repr, general_repr)
    grid_input = fast_prepare_grid_input(board_repr1, board_repr2)
    return ((grid_input, vector_for_grid), vector_input)

def fast_preprocess(inputs):
    inputs = fast_augment(*inputs)
    # inputs = fast_random_transformations(*inputs)
    inputs = [input_.astype(np.float32) for input_ in inputs]
    inputs = fast_normalise(*inputs)
    inputs = fast_prepare(*inputs)
    return inputs