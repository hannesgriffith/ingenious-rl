import numpy as np

from game.board import Board
from game.tiles import Tiles
from game.player import get_player, Deck, Score
from game.utils import get_other_player, find_winner_fast, add_values_for_episode
from learn.strategy import get_strategy_types
from learn.representation import RepresentationGenerator, RepresentationsBuffer
from ui.interface import Request, display_to_game_tiles

def get_gameplay(params):
    if params["game_type"] == "real":
        return RealGameplay(params)
    elif params["game_type"] == "computer":
        return ComputerGameplay(params)
    elif params["game_type"] == "training":
        return TrainingGameplay(params)
    else:
        raise ValueError("Invalid gameplay type chosen.")

class Gameplay:
    def __init__(self, params):
        self.params = params
        self.players = {1: None, 2: None}
        self.ingenious = False
        self.turn_of = None
        self.other = None

    def initialise_game(self):
        self.board = Board()
        self.tiles = Tiles()

    def switch_player(self):
        tmp = self.other
        self.other = self.turn_of
        self.turn_of = tmp

    def find_winner(self):
        p1_score = self.players[1].get_score()
        p2_score = self.players[2].get_score()
        return find_winner_fast(p1_score, p2_score)

    def next_(self):
        raise NotImplementedError("Required to implement next_()")

class RealGameplay(Gameplay):
    def __init__(self, params):
        super().__init__(params)
        self.num_to_pickup = 0
        self.number_moves = 0

    def initialise_game(self, player_to_start):
        super().initialise_game()
        strat_1, strat_2 = get_strategy_types(self.params)
        self.players[1] = get_player(self.params[1]["player_type"], self.board, strat_1, params=self.params[1])
        self.players[2] = get_player(self.params[2]["player_type"], self.board, strat_2, params=self.params[2])

        self.turn_of = get_other_player(player_to_start)
        self.other = get_other_player(self.turn_of)
        self.representation = RepresentationGenerator(self.params)

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

class ComputerGameplay(Gameplay):
    def __init__(self, params):
        super().__init__(params)
        self.number_moves = 0

    def initialise_game(self, _):
        super().initialise_game()
        strat_1, strat_2 = get_strategy_types(self.params)
        self.players[1] = get_player(self.params[1]["player_type"], self.board, strat_1, params=self.params[1])
        self.players[2] = get_player(self.params[2]["player_type"], self.board, strat_2, params=self.params[2])

        self.turn_of = np.random.choice([1, 2])
        self.other = get_other_player(self.turn_of)
        self.players[self.other].pick_up(self.tiles)
        self.players[self.turn_of].pick_up(self.tiles)
        self.representation = RepresentationGenerator()

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
                tile = move_made[4:6]
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

class TrainingGameplay(Gameplay):
    def __init__(self, params):
        super().__init__(params)
        self.should_exchange = {1: False, 2: False}

    def initialise_game(self, player_1, player_2):
        super().initialise_game()
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

        self.representation = RepresentationGenerator()
        self.turn_of = np.random.choice([1, 2])
        self.other = get_other_player(self.turn_of)
        self.players[self.turn_of].pick_up(self.tiles)
        self.players[self.other].pick_up(self.tiles)

    def next_(self, generate_representation=True):
        if not self.ingenious:
            if self.should_exchange[self.turn_of] and self.players[self.turn_of].can_exchange_tiles():
                self.players[self.turn_of].exchange_tiles(self.tiles)
                self.players[self.turn_of].exchange_tiles(self.tiles)
            else:
                self.players[self.turn_of].pick_up(self.tiles)

        if self.ingenious:
            self.ingenious = False
        else:
            self.switch_player()

        move_output = self.players[self.turn_of].make_move(
            self.players,
            self.turn_of,
            self.representation.generate_batched,
            inference=False
        )

        ingenious, _, _, should_exchange, num_ingenious = move_output
        self.ingenious = ingenious
        self.should_exchange[self.turn_of] = should_exchange

        if generate_representation:
            move_representations = self.representation.generate(
                self.board,
                self.players[self.turn_of].deck,
                self.players[self.turn_of].score,
                self.players[get_other_player(self.turn_of)].score,
                ingenious,
                num_ingenious,
                self.players[self.turn_of].can_exchange_tiles(),
                should_exchange,
                np.array([self.turn_of], dtype=np.uint8)
                )
        else:
            move_representations = None

        if self.board.game_is_finished():
            winner = self.find_winner()
        else:
            winner = None

        return move_representations, winner

    def generate_episode(self, p1, p2):
        while True:
            self.initialise_game(p1, p2)
            winner = None
            representations = RepresentationsBuffer()
            while winner is None:
                move_representations, winner = self.next_(generate_representation=True)
                representations.combine_reprs(move_representations)
                del move_representations

            if winner != 0:
                representations.values_repr = add_values_for_episode(
                    representations.values_repr,
                    representations.turn_of_repr,
                    winner
                    )
                return winner, representations

    def play_test_game(self, p1, p2):
        while True:
            self.initialise_game(p1, p2)
            winner = None
            while winner is None:
                _, winner = self.next_(generate_representation=False)
            if winner != 0:
                return winner, None