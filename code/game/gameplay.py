from numba import njit
import numpy as np

from game.board import Board
from game.tiles import Tiles
from game.player import get_player, Deck, Score
from learn.representation import get_representation
from learn.value import get_value_type
from learn.network import get_network

# option to replace tiles in deck
# class to set up configuration, rather than get functions
# add personal tiles tracker for agent

def get_gameplay(params):
    if params["game_type"] == "real":
        return RealGameplay(params)
    elif params["game_type"] == "computer":
        return ComputerGameplay(params)
    elif params["game_type"] == "training":
        return TrainingGameplay(params)
    else:
        raise ValueError("Invalid gameplay type chosen.")

def get_strategy_types(params):
    strategy_type_1 = None
    strategy_type_2 = None

    if params[1]["player_type"] == "computer":
        strategy_type_1 = params[1]["strategy_type"]

    if params[2]["player_type"] == "computer":
        strategy_type_2 = params[2]["strategy_type"]

    return strategy_type_1, strategy_type_2

class RealGameplay:
    def __init__(self, params):
        self.params = params
        self.number_moves = 0
        self.players = {1: None, 2: None}
        self.turn_of = None
        self.other = None
        self.ingenious = False
        self.num_to_pickup = 0

    def initialise_game(self, player_to_start):
        self.board = Board()
        self.tiles = Tiles()

        strat_1, strat_2 = get_strategy_types(self.params)
        self.players[1] = get_player(self.params[1]["player_type"], self.board, strat_1, params=self.params[1])
        self.players[2] = get_player(self.params[2]["player_type"], self.board, strat_2, params=self.params[2])

        self.turn_of = get_other_player(player_to_start)
        self.other = get_other_player(self.turn_of)
        if self.params["representation"]:
            self.representation = get_representation(self.params)

    def switch_player(self):
        tmp = self.other
        self.other = self.turn_of
        self.turn_of = tmp

    def find_winner(self):
        p1_score = self.players[1].get_score()
        p2_score = self.players[2].get_score()
        return find_winner_fast(p1_score, p2_score)

    def get_initial_request(self):
        initial_request = Request()
        initial_request.add_display_message(None,
            "Please pick inital pieces")
        initial_request.add_update_score(1, self.players[1].get_score())
        initial_request.add_update_score(2, self.players[2].get_score())
        for i in [1, 2]:
            if self.players[i].player_type == "computer":
                initial_request.add_request_pickup_tiles(i, 6)
        return initial_request

    def next_(self, response):
        request = Request()

        for item in response.action_iterator():
            if item["type"] == "move_made":
                move_made = item["body"]
                self.ingenious, score = self.players[item["player"]].update_score(move_made.to_game_coords())
                self.board.update_board(move_made.to_game_coords())
                request.add_update_score(item["player"], score)
            elif item["type"] == "tiles_picked_up":
                tiles_picked_up = item["body"]
                self.players[item["player"]].update_deck(display_to_game_tiles(tiles_picked_up))
                raise ValueError('Unrecognised response item.')

        if self.board.game_is_finished():
            request.add_update_score(1, self.players[1].get_score())
            request.add_update_score(2, self.players[2].get_score())
            winner = self.find_winner()
            request.add_game_finished(self.turn_of, winner)
            return request

        if self.ingenious:
            request.add_display_message(self.turn_of, "Ingenious! Go again :)")
            self.ingenious = False
        else:
            self.switch_player()

        if self.players[self.turn_of].player_type == "human":
            request.add_request_move(self.turn_of)

        if self.players[self.turn_of].player_type == "computer":
            move_output = self.players[self.turn_of].make_move(
                        self.players,
                        self.turn_of,
                        self.representation.generate_batched,
                        inference=True
            )
            self.ingenious, chosen_move, score, should_exchange, _ = move_output
            request.add_make_move(self.turn_of, chosen_move)
            self.num_to_pickup += 1
            request.add_update_score(self.turn_of, score)
            if not self.ingenious:
                if should_exchange and self.players[self.turn_of].can_exchange_tiles():
                    request.add_request_exchange_tiles(self.turn_of)
                    self.num_to_pickup = 0
                else:
                    request.add_request_pickup_tiles(self.turn_of, self.num_to_pickup)
                    self.num_to_pickup = 0

        self.number_moves += 1
        return request

class ComputerGameplay:
    def __init__(self, params):
        self.params = params
        self.number_moves = 0
        self.players = {1: None, 2: None}
        self.turn_of = None
        self.other = None
        self.ingenious = False

    def initialise_game(self, _):
        self.board = Board()
        self.tiles = Tiles()

        strat_1, strat_2 = get_strategy_types(self.params)
        self.players[1] = get_player(self.params[1]["player_type"], self.board, strat_1, params=self.params[1])
        self.players[2] = get_player(self.params[2]["player_type"], self.board, strat_2, params=self.params[2])

        self.turn_of = np.random.choice([1, 2])
        self.other = get_other_player(self.turn_of)
        self.players[self.other].pick_up(self.tiles)
        self.players[self.turn_of].pick_up(self.tiles)
        if self.params["representation"]:
            self.representation = get_representation(self.params)

    def switch_player(self):
        tmp = self.other
        self.other = self.turn_of
        self.turn_of = tmp

    def find_winner(self):
        p1_score = self.players[1].get_score()
        p2_score = self.players[2].get_score()
        return find_winner_fast(p1_score, p2_score)

    def get_initial_request(self):
        initial_request = Request()
        initial_request.add_display_message(None, "Picking up initial pieces")
        initial_request.add_update_score(1, self.players[1].get_score())
        initial_request.add_update_score(2, self.players[2].get_score())
        initial_request.add_update_deck(1)
        initial_request.add_update_deck(2)
        initial_request.add_display_message(None, "Player {} starts".format(self.other))
        return initial_request

    def next_(self, response):
        request = Request()

        for item in response.action_iterator():
            if item["type"] == "move_made":
                move_made = item["body"].to_game_coords()
                self.ingenious, score = self.players[item["player"]].update_score(move_made)
                self.board.update_board(move_made)

                request.add_update_score(item["player"], score)
                tile = move_made[6:8]
                self.players[item["player"]].deck.play_tile(tile)

                if not self.ingenious:
                    if self.players[item["player"]].can_exchange_tiles():
                        request.possible_exchange(item["player"])
                    else:
                        self.players[item["player"]].pick_up(self.tiles)
                        request.add_update_deck(item["player"])
            else:
                raise ValueError('Unrecognised response item.')

        if self.board.game_is_finished():
            request.add_update_score(1, self.players[1].get_score())
            request.add_update_score(2, self.players[2].get_score())
            winner = self.find_winner()
            request.add_game_finished(self.turn_of, winner)
            return request

        if self.ingenious:
            request.add_display_message(self.turn_of, "Ingenious! Go again :)")
            self.ingenious = False
        else:
            self.switch_player()

        if self.players[self.turn_of].player_type == "human":
            request.add_request_move(self.turn_of)

        if self.players[self.turn_of].player_type == "computer":
            move_output = self.players[self.turn_of].make_move(
                        self.players,
                        self.turn_of,
                        self.representation.generate_batched,
                        inference=True
            )
            self.ingenious, chosen_move, score, should_exchange, _ = move_output
            request.add_make_move(self.turn_of, chosen_move)
            request.add_update_score(self.turn_of, score)
            if not self.ingenious:
                if should_exchange and self.players[self.turn_of].can_exchange_tiles():
                    self.players[self.turn_of].exchange_tiles(self.tiles)
                    request.add_computer_exchange_tiles(self.turn_of)
                else:
                    self.players[self.turn_of].pick_up(self.tiles)
                    request.add_update_deck(self.turn_of)

        self.number_moves += 1
        return request

@njit
def get_other_player(player):
    if player == 1:
        return 2
    else:
        return 1

@njit
def find_winner_fast(p1_score, p2_score):
    p1_score.sort()
    p2_score.sort()

    for i in range(6):
        p1_colour_score = p1_score[i]
        p2_colour_score = p2_score[i]

        if p1_colour_score > p2_colour_score:
            return 1
        if p2_colour_score > p1_colour_score:
            return 2

    return 0

class TrainingGameplay:
    def __init__(self, params):
        self.params = params
        self.players = {1: None, 2: None}
        self.turn_of = None
        self.other = None
        self.ingenious = False
        self.num_ingenious = 0
        self.should_exchange = {1: False, 2: False}
        self.num_moves = 0

    def initialise_game(self, player_1, player_2):
        self.board = Board()
        self.tiles = Tiles()

        self.players[1] = player_1
        self.players[2] = player_2
        self.players[1].board = self.board
        self.players[2].board = self.board
        self.players[1].tiles = self.tiles
        self.players[2].tiles = self.tiles
        self.players[1].deck = Deck()
        self.players[2].deck = Deck()
        self.players[1].score = Score()
        self.players[2].score = Score()

        self.representation = get_representation(self.params)
        self.move_value = get_value_type(self.params)
        self.turn_of = np.random.choice([1, 2])
        self.other = get_other_player(self.turn_of)
        self.players[self.turn_of].pick_up(self.tiles)
        self.players[self.other].pick_up(self.tiles)

    def switch_player(self):
        tmp = self.other
        self.other = self.turn_of
        self.turn_of = tmp

    def find_winner(self):
        p1_score = self.players[1].get_score()
        p2_score = self.players[2].get_score()
        return find_winner_fast(p1_score, p2_score)

    def next_(self, generate_representation=True):
        if not self.ingenious:
            if self.should_exchange[self.turn_of] and self.players[self.turn_of].can_exchange_tiles():
                self.players[self.turn_of].exchange_tiles(self.tiles)
                self.players[self.turn_of].exchange_tiles(self.tiles)
            else:
                self.players[self.turn_of].pick_up(self.tiles)

        if self.ingenious:
            self.ingenious = False
            was_ingenious = True
        else:
            self.switch_player()
            was_ingenious = False
            self.num_ingenious = 0

        if generate_representation: # turn of: 1 your turn, 0 not your turn
            before_move_representation_self = self.representation.generate(
                self.board,
                self.players[self.turn_of].deck,
                self.players[self.turn_of].score,
                self.players[get_other_player(self.turn_of)].score,
                was_ingenious,
                self.num_ingenious,
                self.players[self.turn_of].can_exchange_tiles(),
                False,
                np.array([1], dtype=np.uint8)
                )
            before_move_representation_other = self.representation.generate(
                self.board,
                self.players[get_other_player(self.turn_of)].deck,
                self.players[get_other_player(self.turn_of)].score,
                self.players[self.turn_of].score,
                was_ingenious,
                self.num_ingenious,
                self.players[get_other_player(self.turn_of)].can_exchange_tiles(),
                False,
                np.array([0], dtype=np.uint8)
                )
        else:
            before_move_representation_self = None
            before_move_representation_other = None

        move_output = self.players[self.turn_of].make_move(
            self.players,
            self.turn_of,
            self.representation.generate_batched,
            inference=False
        )

        is_ingenious, _, _, should_exchange, num_ingenious = move_output
        self.ingenious = is_ingenious
        self.num_ingenious = num_ingenious
        self.should_exchange[self.turn_of] = should_exchange

        if generate_representation: # turn of 0 yourself, 1 other player
            identifier_1 = 1 if is_ingenious else 0
            identifier_2 = 0 if is_ingenious else 1
            after_move_representation_self = self.representation.generate(
                self.board,
                self.players[self.turn_of].deck,
                self.players[self.turn_of].score,
                self.players[get_other_player(self.turn_of)].score,
                is_ingenious,
                num_ingenious,
                self.players[self.turn_of].can_exchange_tiles(),
                should_exchange,
                np.array([identifier_1], dtype=np.uint8)
                )
            after_move_representation_other = self.representation.generate(
                self.board,
                self.players[get_other_player(self.turn_of)].deck,
                self.players[get_other_player(self.turn_of)].score,
                self.players[self.turn_of].score,
                is_ingenious,
                num_ingenious,
                self.players[get_other_player(self.turn_of)].can_exchange_tiles(),
                False,
                np.array([identifier_2], dtype=np.uint8)
                )
        else:
            after_move_representation_self = None
            after_move_representation_other = None

        if self.board.game_is_finished():
            winner = self.find_winner()
        else:
            winner = None

        representations = (
            before_move_representation_self,
            before_move_representation_other,
            after_move_representation_self,
            after_move_representation_other,
            )

        return representations, winner

    def generate_episode(self, p1, p2):
        while True:
            self.initialise_game(p1, p2)
            winner = None
            representations = (
                self.representation.get_new_reprs_buffer(), # before_move_representation_self
                self.representation.get_new_reprs_buffer(), # before_move_representation_other
                self.representation.get_new_reprs_buffer(), # after_move_representation_self
                self.representation.get_new_reprs_buffer(), # after_move_representation_other
            )
            while winner is None:
                move_representations, winner = self.next_(generate_representation=True)
                for i, representation in enumerate(representations):
                    representation.combine_reprs(move_representations[i])
                del move_representations

            if winner != 0:
                updated_representations = self.move_value.add_values_for_episode(representations, winner, self.turn_of)

                before_states = updated_representations[0]
                before_states.combine_reprs(updated_representations[1])
                after_states = updated_representations[2]
                after_states.combine_reprs(updated_representations[3])

                return winner, (before_states, after_states)

    def play_test_game(self, p1, p2):
        while True:
            self.initialise_game(p1, p2)
            winner = None
            while winner is None:
                _, winner = self.next_(generate_representation=False)
            if winner != 0:
                return winner, None

class Request:
    def __init__(self):
        self.actions = []

    def add_display_message(self, player, message):
        action = {"player": player,
                  "type": "display_message",
                  "body": message}
        self.actions.append(action)

    def add_make_move(self, player, move):
        action = {"player": player,
                  "type": "make_move",
                  "body": game_to_display_move(move)}
        self.actions.append(action)

    def add_request_pickup_tiles(self, player, number_to_pickup):
        action = {"player": player,
                  "type": "request_pickup",
                  "body": number_to_pickup}
        self.actions.append(action)

    def add_request_exchange_tiles(self, player):
        action = {"player": player,
                  "type": "request_exchange",
                  "body": None}
        self.actions.append(action)

    def add_computer_exchange_tiles(self, player):
        action = {"player": player,
                  "type": "computer_exchange_tiles",
                  "body": None}
        self.actions.append(action)

    def possible_exchange(self, player):
        action = {"player": player,
                  "type": "possible_exchange",
                  "body": None}
        self.actions.append(action)

    def add_request_move(self, player):
        action = {"player": player,
                  "type": "request_move",
                  "body": None}
        self.actions.append(action)

    def add_update_score(self, player, score):
        action = {"player": player,
                  "type": "update_score",
                  "body": game_to_display_score(score)}
        self.actions.append(action)

    def add_update_deck(self, player):
        action = {"player": player,
                  "type": "update_deck",
                  "body": None}
        self.actions.append(action)

    def add_game_finished(self, player, winner):
        action = {"player": player,
                  "type": "game_finished",
                  "body": winner}
        self.actions.append(action)

    def action_iterator(self):
        for action in self.actions:
            yield action

class Move:
    def __init__(self, coord1, coord2, colour1, colour2):
        self.i1 = coord1[0]
        self.j1 = coord1[1]
        self.i2 = coord2[0]
        self.j2 = coord2[1]
        self.c1 = colour1
        self.c2 = colour2

    def iterator(self):
        hex1 = (self.i1, self.j1, self.c1)
        hex2 = (self.i2, self.j2, self.c2)
        for hex_ in [hex1, hex2]:
            yield hex_

    def to_game_coords(self):
        coords_1 = display_to_game_coords((self.i1, self.j1))
        coords_2 = display_to_game_coords((self.i2, self.j2))
        return np.array([
            coords_1[0],
            coords_1[1],
            coords_1[2],
            coords_2[0],
            coords_2[1],
            coords_2[2],
            self.c1 - 1,
            self.c2 - 1
        ], dtype=np.uint8)

def display_to_game_tiles(tiles):
    return np.array(tiles, dtype=np.uint8).reshape(-1, 2) - 1

def game_to_display_score(score):
    return score.flatten().tolist()

def game_to_display_move(move):
    coords_1 = game_to_display_coords(move[0:3])
    coords_2 = game_to_display_coords(move[3:6])
    colour_1, colour_2 = move[6:8] + 1
    return Move(coords_1, coords_2, colour_1, colour_2)

def game_to_display_coords(coords):
    conversion_dict = game_to_display_coords_dict()
    return conversion_dict[tuple(coords.tolist())]

def display_to_game_coords(coords):
    conversion_dict = display_to_game_coords_dict()
    return conversion_dict[coords]

def game_to_display_coords_dict():
    conversion_dict = display_to_game_coords_dict()
    return {v: k for k, v in conversion_dict.items()}

def display_to_game_coords_dict():
    return {
        (0, 0): (5, 10, 0),
        (0, 1): (4, 10, 1),
        (0, 2): (3, 10, 2),
        (0, 3): (2, 10, 3),
        (0, 4): (1, 10, 4),
        (0, 5): (0, 10, 5),
        (0, 6): (0, 9, 6),
        (0, 7): (0, 8, 7),
        (0, 8): (0, 7, 8),
        (0, 9): (0, 6, 9),
        (0, 10): (0, 5, 10),
        (1, 0): (6, 9, 0),
        (1, 1): (5, 9, 1),
        (1, 2): (4, 9, 2),
        (1, 3): (3, 9, 3),
        (1, 4): (2, 9, 4),
        (1, 5): (1, 9, 5),
        (1, 6): (1, 8, 6),
        (1, 7): (1, 7, 7),
        (1, 8): (1, 6, 8),
        (1, 9): (1, 5, 9),
        (1, 10): (1, 4, 10),
        (2, 0): (7, 8, 0),
        (2, 1): (6, 8, 1),
        (2, 2): (5, 8, 2),
        (2, 3): (4, 8, 3),
        (2, 4): (3, 8, 4),
        (2, 5): (2, 8, 5),
        (2, 6): (2, 7, 6),
        (2, 7): (2, 6, 7),
        (2, 8): (2, 5, 8),
        (2, 9): (2, 4, 9),
        (2, 10): (2, 3, 10),
        (3, 0): (8, 7, 0),
        (3, 1): (7, 7, 1),
        (3, 2): (6, 7, 2),
        (3, 3): (5, 7, 3),
        (3, 4): (4, 7, 4),
        (3, 5): (3, 7, 5),
        (3, 6): (3, 6, 6),
        (3, 7): (3, 5, 7),
        (3, 8): (3, 4, 8),
        (3, 9): (3, 3, 9),
        (3, 10): (3, 2, 10),
        (4, 0): (9, 6, 0),
        (4, 1): (8, 6, 1),
        (4, 2): (7, 6, 2),
        (4, 3): (6, 6, 3),
        (4, 4): (5, 6, 4),
        (4, 5): (4, 6, 5),
        (4, 6): (4, 5, 6),
        (4, 7): (4, 4, 7),
        (4, 8): (4, 3, 8),
        (4, 9): (4, 2, 9),
        (4, 10): (4, 1, 10),
        (5, 0): (10, 5, 0),
        (5, 1): (9, 5, 1),
        (5, 2): (8, 5, 2),
        (5, 3): (7, 5, 3),
        (5, 4): (6, 5, 4),
        (5, 5): (5, 5, 5),
        (5, 6): (5, 4, 6),
        (5, 7): (5, 3, 7),
        (5, 8): (5, 2, 8),
        (5, 9): (5, 1, 9),
        (5, 10): (5, 0, 10),
        (6, 1): (10, 4, 1),
        (6, 2): (9, 4, 2),
        (6, 3): (8, 4, 3),
        (6, 4): (7, 4, 4),
        (6, 5): (6, 4, 5),
        (6, 6): (6, 3, 6),
        (6, 7): (6, 2, 7),
        (6, 8): (6, 1, 8),
        (6, 9): (6, 0, 9),
        (7, 2): (10, 3, 2),
        (7, 3): (9, 3, 3),
        (7, 4): (8, 3, 4),
        (7, 5): (7, 3, 5),
        (7, 6): (7, 2, 6),
        (7, 7): (7, 1, 7),
        (7, 8): (7, 0, 8),
        (8, 3): (10, 2, 3),
        (8, 4): (9, 2, 4),
        (8, 5): (8, 2, 5),
        (8, 6): (8, 1, 6),
        (8, 7): (8, 0, 7),
        (9, 4): (10, 1, 4),
        (9, 5): (9, 1, 5),
        (9, 6): (9, 0, 6),
        (10, 5): (10, 0, 5)
    }
