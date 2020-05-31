from numba import njit, jitclass, uint8
import numpy as np

from game.strategy import get_strategy

def get_player(player_type, board, strategy_type, params=None):
    if player_type == "computer":
        return ComputerPlayer(board, strategy_type, params=params)
    elif player_type == "human":
        return HumanPlayer(board)
    else:
        raise ValueError("Invalid player type chosen.")

@njit
def get_other_player(player):
    if player == 1:
        return 2
    else:
        return 1

@njit
def batch_peek_can_exchange_tiles(decks, scores): # b x 6 x 2, b x 6
    b = scores.shape[0]
    batch_can_exchange = np.ones(b, dtype=np.uint8)
    decks_flat = decks.reshape(b, -1)

    for idx in range(b):
        score = scores[idx]
        deck_flat = decks_flat[idx]
        min_score = np.min(score)
        min_idxs = np.where(score == min_score)[0]
        for min_idx in min_idxs:
            if np.sum(deck_flat == min_idx) > 0:
                batch_can_exchange[idx] = 0
                break

    return batch_can_exchange

class ComputerPlayer:
    def __init__(self, board, strategy_type, params=None):
        self.player_type = "computer"
        self.board = board
        self.deck = Deck()
        self.score = Score()
        self.strategy = get_strategy(strategy_type, params=params)

    def choose_strategy_move(self, players, turn_of, repr_fn, inference=False):
        move, predicted_value = self.strategy.choose_move(
            self.board,
            players[turn_of].deck,
            players[turn_of].score,
            players[get_other_player(turn_of)].score,
            turn_of,
            repr_fn,
            inference=inference
        )

        if inference:
            print(round((predicted_value + 1.) / 2., 2))

        return move, predicted_value

    def make_move(self, players, turn_of, repr_fn, inference=False):
        move, _ = self.choose_strategy_move(players, turn_of, repr_fn, inference=inference)
        chosen_move, should_exchange = move
        ingenious, new_score, num_ingenious = self.update_score(chosen_move)
        self.board.update_board(chosen_move)
        tile_to_play = np.array([chosen_move[6], chosen_move[7]], dtype=np.uint8)
        self.deck.play_tile(tile_to_play)
        return ingenious, chosen_move, new_score, should_exchange, num_ingenious

    def update_score(self, move):
        move_score = self.board.calculate_move_score(move)
        ingenious, num_ingenious = self.score.update_score(move_score)
        return ingenious, self.get_score(), num_ingenious

    def get_score(self):
        return self.score.get_score()

    def update_deck(self, tiles):
        self.deck.add_tiles(tiles)

    def exchange_tiles(self, game_tiles):
        new_tiles = game_tiles.pick_n_tiles_from_bag(6)
        game_tiles.add_tiles(self.deck.get_deck())
        self.deck.replace_deck(new_tiles)

    def pick_up(self, game_tiles):
        num = self.deck.get_num_to_pick_up()
        new_tiles = game_tiles.pick_n_tiles_from_bag(num)
        self.deck.add_tiles(new_tiles)

    def can_exchange_tiles(self):
        return can_exchange_tiles_fast(self.get_score(), self.deck.get_deck())

@njit
def can_exchange_tiles_fast(score, deck):
    min_score = np.min(score)
    min_colours = np.where(score == min_score)[0] # n
    deck_flat = deck.flatten() # 12
    matches = np.expand_dims(deck_flat, 1) == np.expand_dims(min_colours, 0) # 12 x n
    return ~np.any(matches)

class HumanPlayer:
    def __init__(self, board):
        self.player_type = "human"
        self.board = board
        self.deck = Deck()
        self.score = Score()

    def get_score(self):
        return self.score.get_score()

    def update_score(self, move):
        move_score = self.board.calculate_move_score(move)
        ingenious, _ = self.score.update_score(move_score)
        return ingenious, self.get_score()

    def update_deck(self, tiles):
        self.deck.add_tiles(tiles)

    def exchange_tiles(self, game_tiles):
        new_tiles = game_tiles.pick_n_tiles_from_bag(6)
        game_tiles.add_tiles(self.deck.get_deck())
        self.deck.replace_deck(new_tiles)

    def pick_up(self, game_tiles):
        num = self.deck.get_num_to_pick_up()
        new_tiles = game_tiles.pick_n_tiles_from_bag(num)
        self.deck.add_tiles(new_tiles)

    def can_exchange_tiles(self):
        score = self.get_score() # 6
        min_score = np.min(score) # 1
        min_colours = np.where(score == min_score)[0] # n
        deck_flat = self.deck.get_deck().flatten() # 12
        matches = deck_flat[:, np.newaxis] == min_colours[np.newaxis, :] # 12 x n
        return not np.any(matches)

score_spec = [
    ('score', uint8[:]),
]

@jitclass(score_spec)
class Score:
    def __init__(self):
        self.score = np.zeros((6,), dtype=np.uint8)

    def get_score(self):
        return self.score

    def get_score_copy(self):
        return np.copy(self.get_score())

    def update_score(self, move_score):
        count_before = np.sum(self.score == 18)
        self.score += move_score
        self.score[self.score > 18] = 18
        count_after = np.sum(self.score == 18)
        ingenious = count_after > count_before
        num_ingenious = count_after - count_before
        return ingenious, num_ingenious

    def batch_peek_next_scores(self, move_scores):
        move_scores = move_scores.reshape(-1, 6)
        score = np.copy(self.score).astype(np.uint8)
        count_before = np.sum(score == 18)

        batch_size = move_scores.shape[0]
        updated_scores = np.zeros((batch_size, 6), dtype=np.uint8)
        counts_diff_batch = np.zeros((batch_size,), dtype=np.uint8)
        ingenious_batch = np.zeros((batch_size,), dtype=np.uint8)

        for idx in range(batch_size):
            move_score = move_scores[idx, :]
            updated_score = move_score + score
            updated_score[updated_score > 18] = 18
            counts_after = np.sum(updated_score == 18)
            counts_diff = counts_after - count_before
            ingenious = (counts_diff > 0) * 1

            updated_scores[idx, :] = updated_score
            counts_diff_batch[idx] = counts_diff
            ingenious_batch[idx] = ingenious

        return updated_scores, ingenious_batch, counts_diff_batch

deck_spec = [
    ('deck_size', uint8),
    ('num_in_deck', uint8),
    ('deck', uint8[:, :]),
    ('state', uint8[:, :]),
]

@jitclass(deck_spec)
class Deck:
    def __init__(self):
        self.deck_size = 6
        self.num_in_deck = 0
        self.deck = np.zeros((6, 2), dtype=np.uint8)
        self.state = np.zeros((2, 6), dtype=np.uint8)

    def replace_deck(self, new_tiles):
        new_tiles = new_tiles.reshape(-1, 2).astype(np.uint8)
        self.deck = new_tiles.astype(np.uint8)
        self.num_in_deck = self.deck.shape[0]
        self.state = np.zeros((2, 6), dtype=np.uint8)
        self.update_state_for_tiles(new_tiles)

    def get_deck(self):
        return self.deck

    def get_deck_copy(self):
        return np.copy(self.get_deck())

    def get_state(self):
        return self.state

    def get_state_copy(self):
        return np.copy(self.get_state())

    def add_tiles(self, tiles):
        tiles = tiles.reshape(-1, 2).astype(np.uint8)

        if self.num_in_deck == 0:
            self.deck = tiles
        else:
            self.deck = np.vstack((self.deck, tiles))

        self.num_in_deck = self.deck.shape[0]
        self.update_state_for_tiles(tiles)

    def get_num_to_pick_up(self):
        return self.deck_size - self.num_in_deck

    def all_axis(self, array, axis):
        array_shape = array.shape
        count = np.sum(array, axis=axis)
        return count == array_shape[axis]

    def get_tile_index_in_deck(self, tile):
        tile = np.copy(tile).reshape(-1, 2)
        stacked_deck = np.vstack((self.deck, self.deck[:, ::-1]))
        match = self.all_axis(stacked_deck == tile, 1)
        idx = np.where(match)[0][0] % self.deck_size # take first if multiple
        return idx

    def play_tile(self, tile_to_play):
        tile_idx = self.get_tile_index_in_deck(tile_to_play)
        mask = (np.arange(self.deck.shape[0]) == tile_idx)
        mask_others = (mask == 0)
        tile_to_remove = self.deck[tile_idx]
        self.deck = self.deck[mask_others]
        self.num_in_deck = self.deck.shape[0]
        self.remove_tile_from_state(tile_to_remove)

    def update_state_for_tile(self, tile):
        tile = tile.flatten()
        c1 = tile[0]
        c2 = tile[1]
        if c1 != c2:
            self.state[0, c1] += 1
            self.state[0, c2] += 1
        else:
            self.state[1, c1] += 1

    def update_state_for_tiles(self, tiles):
        tiles = np.copy(tiles).reshape(-1, 2)
        for i in range(tiles.shape[0]):
            tile = tiles[i]
            self.update_state_for_tile(tile)

    def remove_tile_from_state(self, tile):
        tile = tile.flatten()
        c1 = tile[0]
        c2 = tile[1]
        if c1 != c2:
            self.state[0, c1] -= 1
            self.state[0, c2] -= 1
        else:
            self.state[1, c1] -= 1

    def remove_tiles_from_state(self, tiles):
        tiles = np.copy(tiles).reshape(-1, 2)
        for i in range(tiles.shape[0]):
            tile = tiles[i]
            self.remove_tile_from_state(tile)

    def batch_peek_next_decks(self, tiles_to_play):
        batch_size = tiles_to_play.shape[0]
        current_deck = self.get_deck_copy()
        next_decks = np.zeros((batch_size, current_deck.shape[0] - 1, 2), dtype=np.uint8)

        for idx in range(batch_size):
            tile_to_play = tiles_to_play[idx]
            tile_idx = self.get_tile_index_in_deck(tile_to_play)
            mask = (np.arange(current_deck.shape[0]) == tile_idx)
            mask_others = (mask == 0)
            next_decks[idx] = current_deck[mask_others]

        return next_decks

    def batch_peek_next_states(self, tiles_to_play):
        batch_size = tiles_to_play.shape[0]
        current_state = self.get_state_copy()
        next_states = np.zeros((batch_size, 2, 6))

        for idx in range(batch_size):
            tile_to_play = tiles_to_play[idx].flatten()
            next_states[idx] = current_state

            c1 = tile_to_play[0]
            c2 = tile_to_play[1]
            if c1 != c2:
                next_states[idx, 0, c1] -= 1
                next_states[idx, 0, c2] -= 1
            else:
                next_states[idx, 1, c1] -= 1

        return next_states

    def aslist(self):
        as_list = []
        deck = np.copy(self.deck)
        for i in range(self.deck.shape[0]):
            entry = (deck[i, 0], deck[i, 1])
            as_list.append(entry)
        return as_list

    def iterator(self):
        deck_as_list = self.aslist()
        for i in deck_as_list:
            yield (i[0] + 1, i[1] + 1)
