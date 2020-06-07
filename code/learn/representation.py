from numba import jitclass, njit, uint8, int32
import numpy as np

from game.player import batch_peek_can_exchange_tiles

def get_representation(params):
    type_ = params["representation"]
    if type_ == "v1":
        return RepresentationGenerator()
    else:
        raise ValueError("Incorrect representation name.")

@njit
def get_other(turn):
    if turn == 1:
        return 0
    else:
        return 1

repr_spec = [
    ('version', uint8),
    ]

@jitclass(repr_spec)
class RepresentationGenerator:
    def __init__(self):
        self.version = 1

    def generate(self, board, deck, score, other_score, ingenious, num_ingenious, can_exchange, should_exchange, turn_of):
        board_repr = board.get_state_copy() # 11 x 11 x 8 (i x j x [6 colours, occupied, available])
        deck_repr = deck.get_state_copy() # 2 x 6 ([single tiles, double tiles] x [6 colours])

        scores_repr = np.vstack((
            np.expand_dims(score.get_score_copy(), 0),
            np.expand_dims(other_score.get_score_copy(), 0)
            )) # 2 x 6 ([player_turn_of, player_other x [6 colour counts]])

        general_repr = np.array((
            ingenious,
            num_ingenious,
            can_exchange,
            should_exchange * can_exchange, # always 0 if can't exchange
            board.move_num), dtype=np.uint8) # (5,)

        values_repr = np.zeros(2, dtype=np.uint8)

        new_reprs_buffer = self.get_new_reprs_buffer()
        new_reprs_buffer.set_single_reprs_from_scratch(board_repr, deck_repr, scores_repr, general_repr, turn_of, values_repr)
        return new_reprs_buffer

    def generate_batched(self, board, deck, score, other_score, turn_of, possible_moves):
        b = possible_moves.shape[0] # possible moves shape: b x 8
        board_repr = board.batch_get_updated_states(possible_moves) # b x 11 x 11 x 8
        deck_repr = deck.batch_peek_next_states(possible_moves[:, 6:8]) # b x 2 x 6

        move_scores = board.batch_calculate_move_scores(possible_moves) # b x 6
        updated_scores, ingenious, num_ingenious = score.batch_peek_next_scores(move_scores) # b x 6, b, b

        next_decks = deck.batch_peek_next_decks(possible_moves[:, 6:8]) # b x 6 x 2
        can_exchange = batch_peek_can_exchange_tiles(next_decks, updated_scores) # b

        other_score = other_score.get_score().flatten()
        scores_repr = np.zeros((b, 2, 6)) # b x 2 x 6
        for idx in range(b):
            scores_repr[idx, 0, :] = updated_scores[idx]
            scores_repr[idx, 1, :] = other_score

        general_repr_dont_exchange = np.hstack((
            np.expand_dims(ingenious, 1),
            np.expand_dims(num_ingenious, 1),
            np.expand_dims(can_exchange, 1),
            np.zeros((b, 1), dtype=np.uint8),
            np.ones((b, 1), dtype=np.uint8) * board.move_num
            )) # b x 5

        general_repr_do_exchange = np.hstack((
            np.expand_dims(ingenious, 1),
            np.expand_dims(num_ingenious, 1),
            np.expand_dims(can_exchange, 1),
            np.ones((b, 1), dtype=np.uint8),
            np.ones((b, 1), dtype=np.uint8) * board.move_num
            )) # b x 5

        possible_moves_stacked = np.concatenate((possible_moves, possible_moves))
        board_repr_stacked = np.concatenate((board_repr, board_repr))
        deck_repr_stacked = np.concatenate((deck_repr, deck_repr))
        scores_repr_stacked = np.concatenate((scores_repr, scores_repr))
        general_repr_stacked = np.concatenate((general_repr_dont_exchange, general_repr_do_exchange))

        can_exchange_stacked = np.concatenate((can_exchange, can_exchange))
        should_exchange_stacked = np.concatenate((np.zeros(b, dtype=np.uint8), np.ones(b, dtype=np.uint8)))
        valid_idxs = np.where(((can_exchange_stacked == 0) & (should_exchange_stacked == 0)) | (can_exchange_stacked == 1))[0]

        possible_moves_subset = possible_moves_stacked[valid_idxs].astype(np.uint8)
        board_repr_subset = board_repr_stacked[valid_idxs].astype(np.uint8)
        deck_repr_subset = deck_repr_stacked[valid_idxs].astype(np.uint8)
        scores_repr_subset = scores_repr_stacked[valid_idxs].astype(np.uint8)
        general_repr_subset = general_repr_stacked[valid_idxs].astype(np.uint8)

        turn_of_repr = np.full(valid_idxs.shape[0], turn_of, dtype=np.uint8)
        values_repr = np.zeros((valid_idxs.shape[0], 2), dtype=np.uint8)

        new_reprs_buffer = self.get_new_reprs_buffer()
        new_reprs_buffer.set_batched_reprs_from_scratch(board_repr_subset, deck_repr_subset, scores_repr_subset, general_repr_subset, turn_of_repr, values_repr)
        return new_reprs_buffer, possible_moves_subset

    def get_new_reprs_buffer(self):
        return RepresentationsBuffer()

reprs_buffer_spec = [
    ('version', uint8),
    ('size', int32),
    ('empty', uint8),
    ('board_repr', uint8[:, :, :, :] ),
    ('deck_repr', uint8[:, :, :] ),
    ('scores_repr', uint8[:, :, :] ),
    ('general_repr', uint8[:, :] ),
    ('values_repr', uint8[:, :] ),
    ('turn_of_repr', uint8[:] ),
    ]

@jitclass(reprs_buffer_spec)
class RepresentationsBuffer():
    def __init__(self):
        self.version = 1
        self.size = 0
        self.empty = 1

    def set_single_reprs_from_scratch(self, board_repr, deck_repr, scores_repr, general_repr, turn_of_repr, values_repr):
        self.board_repr = np.expand_dims(board_repr, 0)
        self.deck_repr = np.expand_dims(deck_repr, 0)
        self.scores_repr = np.expand_dims(scores_repr, 0)
        self.general_repr = np.expand_dims(general_repr, 0)
        self.values_repr = np.expand_dims(values_repr, 0)
        self.turn_of_repr = turn_of_repr
        self.size += 1
        self.empty = 0

    def set_batched_reprs_from_scratch(self, board_repr, deck_repr, scores_repr, general_repr, turn_of_repr, values_repr):
        self.board_repr = board_repr
        self.deck_repr = deck_repr
        self.scores_repr = scores_repr
        self.general_repr = general_repr
        self.values_repr = values_repr
        self.turn_of_repr = turn_of_repr
        self.size += board_repr.shape[0]
        self.empty = 0

    def set_reprs_from_reprs(self, reprs):
        self.board_repr = reprs.board_repr
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
            self.board_repr = np.concatenate((reprs.board_repr, self.board_repr))
            self.deck_repr = np.concatenate((reprs.deck_repr, self.deck_repr))
            self.scores_repr = np.concatenate((reprs.scores_repr, self.scores_repr))
            self.general_repr = np.concatenate((reprs.general_repr, self.general_repr))
            self.turn_of_repr = np.concatenate((reprs.turn_of_repr, self.turn_of_repr))
            self.values_repr = np.concatenate((reprs.values_repr, self.values_repr))
            self.size += reprs.size

    def clip_to_size(self, required_size):
        self.board_repr = self.board_repr[:required_size]
        self.deck_repr = self.deck_repr[:required_size]
        self.scores_repr = self.scores_repr[:required_size]
        self.general_repr = self.general_repr[:required_size]
        self.turn_of_repr = self.turn_of_repr[:required_size]
        self.values_repr = self.values_repr[:required_size]
        self.size = required_size

    def get_examples_by_idxs(self, idxs):
        x = (self.board_repr[idxs], self.deck_repr[idxs], self.scores_repr[idxs], self.general_repr[idxs])
        y = self.values_repr[idxs, 0]
        return x, y

    def augment(self, board_repr, deck_repr, scores_repr, general_repr):
        n = board_repr.shape[0]
        ordering = np.array((0, 1, 2, 3, 4, 5)).astype(np.uint8)

        for i in range(n):
            np.random.shuffle(ordering)
            board_ordering = np.concatenate((ordering, np.array((6, 7)))).astype(np.uint8)

            board_repr_example1 = board_repr[i]
            deck_repr_example1 = deck_repr[i]
            scores_repr_example1 = scores_repr[i]

            board_repr_example_augmented1 = board_repr_example1[:, :, board_ordering] # b x 11 x 11 x 8
            deck_repr_example_augmented1 = deck_repr_example1[:, ordering] # b x 2 x 6
            scores_repr_example_augmented1 = scores_repr_example1[:, ordering] # b x 2 x 6

            board_repr[i] = board_repr_example_augmented1
            deck_repr[i] = deck_repr_example_augmented1
            scores_repr[i] = scores_repr_example_augmented1

        return (board_repr, deck_repr, scores_repr, general_repr)

    def normalise(self, board_repr, deck_repr, scores_repr, general_repr):
        board_repr_normalised = board_repr.astype(np.float32)       # b x 11 x 11 x 8
        deck_repr_normalised = deck_repr.astype(np.float32)         # b x 2 x 6
        scores_repr_normalised = scores_repr.astype(np.float32)     # b x 2 x 6
        general_repr_normalised = general_repr.astype(np.float32)   # b x 5

        deck_repr_normalised /= 4.0
        scores_repr_normalised /= 18.0
        general_repr_normalised /= np.array(((1, 2, 1, 1, 40))).astype(np.float32)

        return (board_repr_normalised, deck_repr_normalised, scores_repr_normalised, general_repr_normalised)

    def prepare(self, board_repr, deck_repr, scores_repr, general_repr):
        b = board_repr.shape[0]

        deck_repr_flat = deck_repr.reshape(b, -1)
        scores_repr_flat = scores_repr.reshape(b, -1)
        general_repr_flat = general_repr
        vector_input = np.hstack((deck_repr_flat, scores_repr_flat, general_repr_flat))
        
        grid_input_offset = self.offset_grid(board_repr)
        extra_feature_channels = self.get_extra_grid_feature_channels(grid_input_offset.shape[0])
        grid_input_combined = self.concat_channels(grid_input_offset, extra_feature_channels)
        grid_input_offset = np.transpose(grid_input_combined, (0, 3, 1, 2)) # NHWC -> NCHW

        return (grid_input_offset, vector_input)

    def offset_grid(self, board_repr):
        n, _, _, c = board_repr.shape
        offset_grid = np.zeros((n, 11, 21, c)).astype(board_repr.dtype) # NHWC

        for i in range(11):
            for j in range(11):
                if i % 2 == 0:
                    if j < 10:
                        offset_grid[:, i, 2 * j + 1, :] = board_repr[:, j, i, :]
                else:
                    offset_grid[:, i, 2 * j, :] = board_repr[:, j, i, :]

        return offset_grid

    def get_extra_grid_feature_channels(self, b):
        playable_channels = self.create_playable_channels(b)
        idx_channels = self.create_idx_channels(b)
        extra_channels = self.concat_channels(playable_channels, idx_channels)
        return extra_channels

    def concat_channels(self, array1, array2):
        array1 = np.transpose(array1, (3, 1, 2, 0))
        array2 = np.transpose(array2, (3, 1, 2, 0))
        concatenated = np.concatenate((array1, array2))
        concatenated = np.transpose(concatenated, (3, 1, 2, 0))
        return concatenated

    def create_playable_channels(self, b):
        channels = np.zeros((b, 11, 21, 1), dtype=np.uint8)
        playable_channel = self.get_playable_channel()

        for idx in range(b):
            channels[idx] = playable_channel[0]

        return channels.astype(np.float32)

    def get_playable_channel(self):
        channel = np.zeros((1, 11, 21, 1), dtype=np.uint8)
        playable = np.array([
            [0, 5], [0, 7], [0, 9], [0, 11], [0, 13], [0, 15],
            [1, 4], [1, 6], [1, 8], [1, 10], [1, 12], [1, 14], [1, 16],
            [2, 3], [2, 5], [2, 7], [2, 9],  [2, 11], [2, 13], [2, 15], [2, 17],
            [3, 2], [3, 4], [3, 6], [3, 8],  [3, 10], [3, 12], [3, 14], [3, 16], [3, 18],
            [4, 1], [4, 3], [4, 5], [4, 7],  [4, 9],  [4, 11], [4, 13], [4, 15], [4, 17], [4, 19],
            [5, 0], [5, 2], [5, 4], [5, 6],  [5, 8],  [5, 10], [5, 12], [5, 14], [5, 16], [5, 18], [5, 20],
            [6, 1], [6, 3], [6, 5], [6, 7],  [6, 9],  [6, 11], [6, 13], [6, 15], [6, 17], [6, 19],
            [7, 2], [7, 4], [7, 6], [7, 8],  [7, 10], [7, 12], [7, 14], [7, 16], [7, 18],
            [8, 3], [8, 5], [8, 7], [8, 9],  [8, 11], [8, 13], [8, 15], [8, 17],
            [9, 4], [9, 6], [9, 8], [9, 10], [9, 12], [9, 14], [9, 16],
            [10, 5], [10, 7], [10, 9], [10, 11], [10, 13], [10, 15]
        ], dtype=np.uint8).reshape(-1, 2)

        for idx in range(playable.shape[0]):
            i, j = playable[idx]
            channel[0, i, j, 0] = 1

        return channel

    def create_idx_channels(self, b):
        channels = np.zeros((b, 11, 21, 1), dtype=np.uint8)
        idx_channel = self.get_idx_channel()

        for idx in range(b):
            channels[idx] = idx_channel[0]

        return channels

    # def get_idx_channel(self):
    #     i_channel = np.ones((1, 11, 21, 1), dtype=np.float32)
    #     j_channel = np.ones((1, 11, 21, 1), dtype=np.float32)

    #     i_channel = i_channel * np.arange(11).astype(np.float32).reshape(1, 11, 1, 1)
    #     i_channel = i_channel / 11.

    #     j_channel = j_channel * np.arange(21).astype(np.float32).reshape(1, 1, 21, 1)
    #     j_channel = j_channel / 21.

    #     idx_channels = self.concat_channels(i_channel, j_channel)
    #     return idx_channels.astype(np.float32)

    def get_playable_channel(self):
        channel = np.zeros((1, 11, 21, 1), dtype=np.uint8)
        idx_1 = np.array([
            [0, 5], [0, 7], [0, 9], [0, 11], [0, 13], [0, 15],
            [1, 4], [1, 16],
            [2, 3], [2, 17],
            [3, 2], [3, 18],
            [4, 1], [4, 19],
            [5, 0], [5, 20],
            [6, 1], [6, 19],
            [7, 2], [7, 18],
            [8, 3], [8, 17],
            [9, 4], [9, 16],
            [10, 5], [10, 7], [10, 9], [10, 11], [10, 13], [10, 15]
        ], dtype=np.uint8).reshape(-1, 2)

        idx_2 = np.array([
            [1, 6], [1, 8], [1, 10], [1, 12], [1, 14],
            [2, 5], [2, 15],
            [3, 4], [3, 16],
            [4, 3], [4, 17],
            [5, 2], [5, 18],
            [6, 3], [6, 17],
            [7, 4], [7, 16],
            [8, 5], [8, 15],
            [9, 6], [9, 8], [9, 10], [9, 12], [9, 14]
        ], dtype=np.uint8).reshape(-1, 2)

        idx_3 = np.array([
            [2, 7], [2, 9],  [2, 11], [2, 13],
            [3, 6], [3, 14],
            [4, 5], [4, 15],
            [5, 4], [5, 16],
            [6, 5], [6, 15],
            [7, 6], [7, 14],
            [8, 7], [8, 9],  [8, 11], [8, 13]
        ], dtype=np.uint8).reshape(-1, 2)

        idx_4 = np.array([
            [3, 8],  [3, 10], [3, 12],
            [4, 7],  [4, 13],
            [5, 6],  [5, 14],
            [6, 7],  [6, 13],
            [7, 8],  [7, 10], [7, 12]
        ], dtype=np.uint8).reshape(-1, 2)

        idx_5 = np.array([
            [4, 9],  [4, 11],
            [5, 8],  [5, 10], [5, 12],
            [6, 9],  [6, 11]
        ], dtype=np.uint8).reshape(-1, 2)

        idx_6 = np.array([[5, 10]], dtype=np.uint8).reshape(-1, 2)

        for val, idx_playables in enumerate([idx_1, idx_2, idx_3, idx_4, idx_5, idx_6]):
            for idx in range(idx_playables.shape[0]):
                i, j = idx_playables[idx]
                channel[0, i, j, 0] = val + 1

        return channel