import numpy as np

from game.strategy import get_strategy

def get_player(player_type, player_num, player_name, board, tiles, strategy_type):
    if player_type == "computer":
        return ComputerPlayer(player_num, player_name, board, tiles, strategy_type)
    elif player_type == "human":
        return HumanPlayer(player_num, player_name, board, tiles)
    else:
        assert False

class ComputerPlayer:
    def __init__(self, player_num, player_name, board, tiles, strategy_type):
        self.player_type = "computer"
        self.player_num = player_num
        self.name = player_name

        self.tiles = tiles
        self.board = board
        self.deck = Deck()
        self.score = Score()
        self.strategy = get_strategy(strategy_type)

    def choose_strategy_move(self, players, player_num, repr_fn):
        move, value = self.strategy.choose_move(
            self.board,
            self.deck,
            self.score,
            players,
            player_num,
            repr_fn
        )
        return move, value

    def make_move(self, players, player_num, repr_fn):
        move, _ = self.choose_strategy_move(players, player_num, repr_fn)
        chosen_move, should_exchange = move
        ingenious, new_score = self.update_score(chosen_move)
        self.board.update_board(chosen_move)
        tile_to_play = np.array([chosen_move[6], chosen_move[7]], dtype=np.uint8)
        self.deck.play_tile(tile_to_play)
        return ingenious, chosen_move, new_score, should_exchange

    def update_score(self, move):
        move_score = self.board.calculate_move_score(move)
        ingenious = self.score.update_score(move_score)
        return ingenious, self.get_score()

    def get_score(self):
        return self.score.get_score()

    def update_deck(self, tiles):
        self.deck.add_tiles(tiles)

    def exchange_tiles(self):
        new_tiles = self.tiles.pick_n_tiles_from_bag(6)
        self.tiles.add_tiles(self.deck.get_deck())
        self.deck.replace_deck(new_tiles)

    def pick_up(self):
        num = self.deck.get_num_to_pick_up()
        new_tiles = self.tiles.pick_n_tiles_from_bag(num)
        self.deck.add_tiles(new_tiles)

    def can_exchange_tiles(self):
        score = self.get_score() # 6
        min_score = np.min(score) # 1
        min_colours = np.where(score == min_score)[0] # n
        deck_flat = self.deck.get_deck().flatten() # 12
        matches = deck_flat[:, np.newaxis] == min_colours[np.newaxis, :] # 12 x n
        return not np.any(matches)

    def batch_peek_can_exchange_tiles(self, deck, scores): # b x 6 x 2, b x 6
        scores = scores.reshape(-1, 6)
        b = score.shape[0]
        min_scores = np.min(scores, axis=1) # b
        min_matches = (scores == min_scores[:, np.newaxis]) # b x 6
        min_colours = np.where(min_matches) # n x 2

        deck_flat = np.copy(deck).flatten() # 12
        matches = min_colours[1][:, np.newaxis] == deck_flat[np.newaxis, :] # n x 12
        matches = np.any(matches, axis=1) # n
        batch_idxs = min_colours[0][matches] # n subset where matches True

        batch_matches = np.full((b,), False) # b
        batch_matches[batch_idxs] = True # n subset set to True

        return ~batch_matches # b

class HumanPlayer:
    def __init__(self, player_num, player_name, board, tiles):
        self.player_type = "human"
        self.player_num = player_num
        self.name = player_name

        self.tiles = tiles
        self.board = board
        self.deck = Deck()
        self.score = Score()

    def get_score(self):
        return self.score.get_score()

    def update_score(self, move):
        move_score = self.board.calculate_move_score(move)
        ingenious = self.score.update_score(move_score)
        return ingenious, self.get_score()

    def update_deck(self, tiles):
        self.deck.add_tiles(tiles)

    def exchange_tiles(self):
        new_tiles = self.tiles.pick_n_tiles_from_bag(6)
        self.tiles.add_tiles(self.deck.get_deck())
        self.deck.replace_deck(new_tiles)

    def pick_up(self):
        num = self.deck.get_num_to_pick_up()
        new_tiles = self.tiles.pick_n_tiles_from_bag(num)
        self.deck.add_tiles(new_tiles)

    def can_exchange_tiles(self):
        score = self.get_score() # 6
        min_score = np.min(score) # 1
        min_colours = np.where(score == min_score)[0] # n
        deck_flat = self.deck.get_deck().flatten() # 12
        matches = deck_flat[:, np.newaxis] == min_colours[np.newaxis, :] # 12 x n
        return not np.any(matches)

class Score:
    def __init__(self):
        self.score = np.zeros((6,), dtype=np.uint8)

    def get_score(self):
        return self.score

    def update_score(self, move_score):
        count_before = np.sum(self.score == 18)
        self.score += move_score
        self.score[self.score > 18] = 18
        count_after = np.sum(self.score == 18)
        ingenious = count_after > count_before
        return ingenious

    def batch_peek_next_scores(self, move_scores):          # b x 6
        score = np.copy(self.score).reshape(1, 6)           # 1 x 6
        move_scores = move_scores.reshape(-1, 6)            # b x 6
        count_before = np.sum(score == 18)                  # 1
        updated_scores = move_scores + score                # b x 6
        updated_scores[updated_scores > 18] = 18            # b x 6
        counts_after = np.sum(updated_scores == 18, axis=1) # b
        counts_diff = counts_after - count_before           # b
        ingenious = counts_diff > 0                         # b
        return updated_scores, ingenious, counts_diff

class Deck:
    def __init__(self):
        self.deck_size = 6
        self.num_in_deck = 0
        self.deck = None

    def replace_deck(self, new_tiles):
        self.deck = new_tiles
        self.num_in_deck = self.deck.shape[0]

    def get_deck(self):
        return self.deck

    def add_tiles(self, tiles):
        tiles = tiles.reshape(-1, 2)
        if self.num_in_deck == 0:
            self.deck = tiles
        else:
            self.deck = np.vstack([self.deck, tiles])
        self.num_in_deck = self.deck.shape[0]

    def get_num_to_pick_up(self):
        return self.deck_size - self.num_in_deck

    def get_tile_index_in_deck(self, tile):
        tile = tile.reshape(1, 2)
        stacked_deck = np.vstack((self.deck, self.deck[:, ::-1]))
        match = np.all(stacked_deck == tile, axis=1)
        idx = np.where(match)[0][0] % self.deck_size # take first if multiple
        return idx

    def play_tile(self, tile_to_play):
        tile_idx = self.get_tile_index_in_deck(tile_to_play)
        self.deck = np.delete(self.deck, tile_idx, axis=0)
        self.num_in_deck = self.deck.shape[0]

    def batch_get_tile_indices_in_deck(self, tiles): # b x 2
        stacked_deck = np.vstack((self.deck, self.deck[:, ::-1])) # 12 x 2
        matches = tiles[:, np.newaxis, :] == stacked_deck[np.newaxis, :, :] # b x 12 x 2
        matches = np.all(matches, axis=2) # b x 12
        idxs = np.argmax(matches, axis=1) % self.deck_size # b
        return idxs

    def batch_peek_next_decks(self, tiles_to_play): # b x 2
        b = tiles_to_play.shape[0] if tiles_to_play.ndim > 1 else 1
        deck = np.copy(self.deck) # 6 x 2
        tile_idxs = self.batch_get_tile_indices_in_deck(tiles_to_play) # b
        next_decks = np.tile(np.expand_dims(deck, 0), (b, 1, 1)) # b x 6 x 2
        next_decks = np.delete(deck, tile_idxs, axis=1) # b x 5 x 2
        return next_decks # b x 5 x 2

    def aslist(self):
        return [tuple(i) for i in np.copy(self.deck).tolist()]

    def iterator(self):
        for i in self.aslist():
            yield (i[0] + 1, i[1] + 1)
