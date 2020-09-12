from numba import njit, prange
import numpy as np

from game.utils import get_other_player
from learn.strategy import get_strategy

def get_player(player_type, board, strategy_type, params=None):
    if player_type == "computer":
        return ComputerPlayer(board, strategy_type, params=params)
    elif player_type == "human":
        return HumanPlayer(board)
    else:
        raise ValueError("Invalid player type chosen.")

class Player:
    def __init__(self, board):
        self.board = board
        self.deck = Deck()
        self.score = Score()

    def get_score(self):
        return self.score.get_score()

    def pick_up(self, game_tiles):
        num = self.deck.get_num_to_pick_up()
        new_tiles = game_tiles.pick_n_tiles_from_bag(num)
        self.deck.add_tiles(new_tiles)

    def exchange_tiles(self, game_tiles):
        new_tiles = game_tiles.pick_n_tiles_from_bag(6)
        game_tiles.add_tiles(self.deck.get_deck())
        self.deck.replace_deck(new_tiles)

    def update_deck(self, tiles):
        self.deck.add_tiles(tiles)

    def update_score(self, move):
        move_score = self.board.calculate_move_score(move)
        ingenious, _ = self.score.update_score(move_score)
        return ingenious, self.get_score()

    def can_exchange_tiles(self):
        return fast_can_exchange_tiles(self.get_score(), self.deck.get_deck())

class HumanPlayer(Player):
    def __init__(self, board):
        super().__init__(board)
        self.player_type = "human"

class ComputerPlayer(Player):
    def __init__(self, board, strategy_type, params=None):
        super().__init__(board)
        self.player_type = "computer"
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
        tile_to_play = np.array([chosen_move[4], chosen_move[5]], dtype=np.uint8)
        self.deck.play_tile(tile_to_play)
        return ingenious, chosen_move, new_score, should_exchange, num_ingenious

    def update_score(self, move):
        move_score = self.board.calculate_move_score(move)
        ingenious, num_ingenious = self.score.update_score(move_score)
        return ingenious, self.get_score(), num_ingenious

class Score:
    def __init__(self):
        self.score = np.zeros((6,), dtype=np.uint8)

    def get_score(self):
        return self.score

    def get_score_copy(self):
        return np.copy(self.get_score())

    def update_score(self, move_score):
        self.score, ingenious, num_ingenious = fast_update_score(self.score, move_score)
        return ingenious, num_ingenious

    def batch_peek_next_scores(self, move_scores):
        return fast_batch_peek_next_scores(self.score, move_scores)

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

    def get_state(self):
        return self.state

    def get_state_copy(self):
        return np.copy(self.get_state())

    def add_tiles(self, tiles):
        self.deck, self.num_in_deck = fast_add_tiles(tiles, self.deck, self.state, self.num_in_deck)

    def get_num_to_pick_up(self):
        return self.deck_size - self.num_in_deck

    def play_tile(self, tile_to_play):
        self.deck, self.num_in_deck, self.state = fast_play_tile(tile_to_play, self.deck, self.state)

    def update_state_for_tiles(self, tiles):
        self.state = fast_update_state_for_tiles(tiles, self.state)

    def remove_tiles_from_state(self, tiles):
        self.state = fast_remove_tiles_from_state(tiles, self.state)

    def batch_peek_next_decks(self, tiles_to_play):
        return fast_batch_peek_next_decks(tiles_to_play, self.deck)

    def batch_peek_next_states(self, tiles_to_play):
        return fast_batch_peek_next_states(tiles_to_play, self.state)

    def as_list(self):
        as_list = []
        deck = np.copy(self.deck)
        for i in range(self.deck.shape[0]):
            entry = (deck[i, 0], deck[i, 1])
            as_list.append(entry)
        return as_list

    def iterator(self):
        deck_as_list = self.as_list()
        for i in deck_as_list:
            yield (i[0] + 1, i[1] + 1)
    
    @staticmethod
    def create_dummy_deck():
        dummy_deck = Deck()
        dummy_deck.deck = np.array((
            (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
            (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
            (2, 2), (2, 3), (2, 4), (2, 5),
            (3, 3), (3, 4), (3, 5),
            (4, 4), (4, 5),
            (5, 5)
            ), dtype=np.uint8)
        dummy_deck.state = np.array(
            (
                (2, 2, 2, 2, 2, 2),
                (2, 2, 2, 2, 2, 2)
            ), dtype=np.uint8)
        return dummy_deck

@njit(cache=True)
def fast_can_exchange_tiles(score, deck):
    min_score = np.min(score)
    min_colours = np.where(score == min_score)[0] # n
    deck_flat = deck.flatten() # 12
    matches = np.expand_dims(deck_flat, 1) == np.expand_dims(min_colours, 0) # 12 x n
    return ~np.any(matches)

@njit(parallel=True, fastmath=True, cache=True)
def fast_batch_peek_can_exchange_tiles(decks, scores): # b x 6 x 2, b x 6
    b = scores.shape[0]
    batch_can_exchange = np.ones(b, dtype=np.uint8)
    decks_flat = decks.reshape(b, -1)

    for idx in prange(b):
        score = scores[idx]
        deck_flat = decks_flat[idx]
        min_score = np.min(score)
        min_idxs = np.where(score == min_score)[0]
        for min_idx in min_idxs:
            if np.sum(deck_flat == min_idx) > 0:
                batch_can_exchange[idx] = 0
                break

    return batch_can_exchange

@njit(cache=True)
def fast_update_score(score, move_score):
    count_before = np.sum(score == 18)
    score += move_score
    score[score > 18] = 18
    count_after = np.sum(score == 18)
    ingenious = count_after > count_before
    num_ingenious = count_after - count_before
    return score, ingenious, num_ingenious

@njit(parallel=True, fastmath=True, cache=True)
def fast_batch_peek_next_scores(score, move_scores):
    move_scores = move_scores.reshape(-1, 6)
    count_before = np.sum(score == 18)

    batch_size = move_scores.shape[0]
    updated_scores = np.zeros((batch_size, 6), dtype=np.uint8)
    counts_diff_batch = np.zeros((batch_size,), dtype=np.uint8)
    ingenious_batch = np.zeros((batch_size,), dtype=np.uint8)

    for idx in prange(batch_size):
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

@njit(cache=True)
def fast_add_tiles(tiles, deck, state, num_in_deck):
    tiles = tiles.reshape(-1, 2).astype(np.uint8)

    if num_in_deck == 0:
        deck = tiles
    else:
        deck = np.vstack((deck, tiles))

    num_in_deck = deck.shape[0]
    fast_update_state_for_tiles(tiles, state)

    return deck, num_in_deck

@njit(cache=True)
def fast_all_axis(array, axis):
    array_shape = array.shape
    count = np.sum(array, axis=axis)
    return count == array_shape[axis]

@njit(cache=True)
def fast_get_tile_index_in_deck(tile, deck):
    deck_size = deck.shape[0]
    tile = np.copy(tile).reshape(-1, 2)
    stacked_deck = np.vstack((deck, deck[:, ::-1]))
    match = fast_all_axis(stacked_deck == tile, 1)
    idx = np.where(match)[0][0] % deck_size # take first if multiple
    return idx

@njit(cache=True)
def fast_play_tile(tile_to_play, deck, state):
    tile_idx = fast_get_tile_index_in_deck(tile_to_play, deck)
    mask = (np.arange(deck.shape[0]) == tile_idx)
    mask_others = (mask == 0)
    tile_to_remove = deck[tile_idx]
    deck = deck[mask_others]
    num_in_deck = deck.shape[0]
    state = fast_remove_tile_from_state(tile_to_remove, state)
    return deck, num_in_deck, state

@njit(cache=True)
def fast_update_state_for_tile(tile, state):
    tile = tile.flatten()
    c1 = tile[0]
    c2 = tile[1]
    if c1 != c2:
        state[0, c1] += 1
        state[0, c2] += 1
    else:
        state[1, c1] += 1
    return state

@njit(cache=True)
def fast_update_state_for_tiles(tiles, state):
    tiles = np.copy(tiles).reshape(-1, 2)
    for i in range(tiles.shape[0]):
        tile = tiles[i]
        state = fast_update_state_for_tile(tile, state)
    return state

@njit(cache=True)
def fast_remove_tile_from_state(tile, state):
    tile = tile.flatten()
    c1 = tile[0]
    c2 = tile[1]
    if c1 != c2:
        state[0, c1] -= 1
        state[0, c2] -= 1
    else:
        state[1, c1] -= 1
    return state

@njit(cache=True)
def fast_remove_tiles_from_state(tiles, state):
    tiles = np.copy(tiles).reshape(-1, 2)
    for i in range(tiles.shape[0]):
        tile = tiles[i]
        state = fast_remove_tile_from_state(tile, state)
    return state

@njit(parallel=True, fastmath=True, cache=True)
def fast_batch_peek_next_decks(tiles_to_play, deck):
    batch_size = tiles_to_play.shape[0]
    current_deck = np.copy(deck)
    next_decks = np.zeros((batch_size, current_deck.shape[0] - 1, 2), dtype=np.uint8)

    for idx in prange(batch_size):
        tile_to_play = tiles_to_play[idx]
        tile_idx = fast_get_tile_index_in_deck(tile_to_play, current_deck)
        mask = (np.arange(current_deck.shape[0]) == tile_idx)
        mask_others = (mask == 0)
        next_decks[idx] = current_deck[mask_others]

    return next_decks

@njit(parallel=True, fastmath=True, cache=True)
def fast_batch_peek_next_states(tiles_to_play, state):
    batch_size = tiles_to_play.shape[0]
    current_state = np.copy(state)
    next_states = np.zeros((batch_size, 2, 6))

    for idx in prange(batch_size):
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