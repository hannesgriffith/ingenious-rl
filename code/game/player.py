import random as rn

import numpy as np

from game.strategy import get_strategy
from game.misc import flip_tile

def get_player(player_type, player_num, player_name, board, tiles,
                strategy_type):
    if player_type == "computer":
        return ComputerPlayer(player_num, player_name, board, tiles,
                                strategy_type)
    elif player_type == "human":
        return HumanPlayer(player_num, player_name, board, tiles)
    else:
        assert False

class ComputerPlayer:
    def __init__(self, player_num, player_name, board, tiles, strategy_type):
        self.player_num = player_num
        self.player_type = "computer"
        self.name = player_name

        self.tiles = tiles
        self.board = board
        self.deck = Deck(tiles=self.tiles)
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
        move_score = self.board.calculate_move_score(chosen_move)
        self.board.update_board(chosen_move)
        ingenious = self.score.update_score(move_score)
        tile_to_play = (chosen_move.colour1, chosen_move.colour2)
        self.deck.play_tile(tile_to_play)
        return ingenious, chosen_move, self.get_score(), should_exchange

    def can_exchange_tiles(self):
        min_score = np.min(self.get_score())
        min_colours = [c + 1 for c, s in enumerate(self.score.get_score()) if s == min_score]
        for tile in self.deck.iterator():
            c1, c2 = tile
            if c1 in min_colours or c2 in min_colours:
                return False
        return True

    def peak_can_exchange_tiles(self, deck, score):
        min_score = np.min(score)
        min_colours = [c + 1 for c, s in enumerate(score) if s == min_score]
        for tile in deck:
            c1, c2 = tile
            if c1 in min_colours or c2 in min_colours:
                return False
        return True

    def update_deck(self, tiles):
        self.deck.add_tiles(tiles)

    def pick_up(self):
        while self.deck.should_pickup_tile():
            new_tile = self.tiles.pick_tile_from_bag()
            self.deck.add_tile(new_tile)

    def exchange_tiles(self):
        new_tiles = [self.tiles.pick_tile_from_bag() for _ in range(6)]
        self.deck.add_deck_to_bag()
        self.deck.replace_deck(new_tiles)

    def get_score(self):
        return self.score.get_score()

class HumanPlayer:
    def __init__(self, player_num, player_name, board, tiles):
        self.player_num = player_num
        self.player_type = "human"
        self.name = player_name

        self.tiles = tiles
        self.board = board
        self.deck = Deck(tiles=self.tiles)
        self.score = Score()

    def can_exchange_tiles(self):
        min_score = np.min(self.get_score())
        min_colours = [c + 1 for c, s in enumerate(self.score.get_score()) if s == min_score]
        for tile in self.deck.iterator():
            c1, c2 = tile
            if c1 in min_colours or c2 in min_colours:
                return False
        return True

    def update_score(self, move):
        move_score = self.board.calculate_move_score(move)
        ingenious = self.score.update_score(move_score)
        return ingenious, self.get_score()

    def update_deck(self, tiles):
        self.deck.add_tiles(tiles)

    def exchange_tiles(self):
        new_tiles = [self.tiles.pick_tile_from_bag() for _ in range(6)]
        self.deck.add_deck_to_bag()
        self.deck.replace_deck(new_tiles)

    def pick_up(self):
        while self.deck.should_pickup_tile():
            new_tile = self.tiles.pick_tile_from_bag()
            self.deck.add_tile(new_tile)

    def get_score(self):
        return self.score.get_score()

class Score:
    def __init__(self):
        self.score = np.zeros((6,), dtype='int32')

    def update_score(self, move_score):
        move_score = np.array(move_score, dtype='int32')
        ingenious = False
        count_before = np.sum(self.score == 18)
        self.score += move_score
        self.score[self.score > 18] = 18
        count_after = np.sum(self.score == 18)
        if count_after > count_before:
            ingenious = True
        return ingenious

    def get_score(self):
        return self.score.tolist()

    def get_score_copy(self):
        return [i for i in self.get_score()]

    def min_score(self):
        return np.min(self.score)

    def peak_next_score(self, move_score):
        score_copy = self.get_score_copy()
        move_score = np.array(move_score, dtype='int32')
        ingenious = False
        count_before = np.sum(score_copy == 18)
        score_copy += move_score
        score_copy[score_copy > 18] = 18
        count_after = np.sum(score_copy == 18)
        if count_after > count_before:
            ingenious = True
        return score_copy, ingenious

class Deck:
    def __init__(self, tiles=None):
        self.deck = []
        self.tiles = tiles

    def should_pickup_tile(self):
        return len(self.deck) < 6

    def play_tile(self, tile_to_play):
        if tile_to_play in self.deck:
            self.deck.remove(tile_to_play)
        elif flip_tile(tile_to_play) in self.deck:
            self.deck.remove(flip_tile(tile_to_play))
        else:
            assert False

    def peak_next_deck(self, tile_to_play):
        deck_copy = self.get_deck_copy()
        if tile_to_play in deck_copy:
            deck_copy.remove(tile_to_play)
        elif flip_tile(tile_to_play) in deck_copy:
            deck_copy.remove(flip_tile(tile_to_play))
        else:
            assert False
        return deck_copy

    def get_deck_copy(self):
        return [i for i in self.deck]

    def add_tile(self, new_tile):
        self.deck.append(new_tile)

    def add_tiles(self, tiles):
        for tile in tiles:
            self.add_tile(tile)

    def add_deck_to_bag(self):
        self.tiles.add_tiles(self.deck)
        assert self.tiles is not None

    def replace_deck(self, new_tiles):
        self.deck = new_tiles
        assert len(self.deck) == 6

    def iterator(self):
        for tile in self.deck:
            yield tile
