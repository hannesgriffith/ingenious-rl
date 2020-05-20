import random as rn

import numpy as np

from game.strategy import get_strategy
from game.misc import flip_tile

def get_player(player_number, board, tiles, params):
    if params[player_number]["player_type"] == "computer":
        return ComputerPlayer(player_number, board, tiles, params)
    elif params[player_number]["player_type"] == "human":
        return HumanPlayer(player_number, board, tiles, params)

class ComputerPlayer:
    def __init__(self, player_number, board, tiles, params):
        self.player_number = player_number
        self.name = params[self.player_number]["name"]
        self.player_type = params[self.player_number]["player_type"]
        self.strategy_type = params[self.player_number]["strategy_type"]

        self.tiles = tiles
        self.board = board
        self.deck = Deck(tiles=self.tiles)
        self.score = Score()
        self.strategy = get_strategy(self.strategy_type, self.board,
                                        self.deck, self.score)

    def make_move(self):
        chosen_move, should_exchange = self.strategy.choose_move()
        self.board.check_move_is_legal(chosen_move)
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
    def __init__(self, player_number, board, tiles, params):
        self.player_number = player_number
        self.name = params[self.player_number]["name"]
        self.player_type = params[self.player_number]["player_type"]

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

    def min_score(self):
        return np.min(self.score)

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
