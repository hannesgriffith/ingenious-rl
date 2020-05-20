import random as rn

from game.board import Board
from game.tiles import Tiles
from game.player import get_player
from learn.representation import get_representation

# option to replace tiles in deck
# class to set up configuration, rather than get functions
# add personal tiles tracker for agent

def get_gameplay(params):
    if params["game_type"] == "real":
        return RealGameplay(params)
    elif params["game_type"] == "computer":
        return ComputerGameplay(params)

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
        self.players[1] = get_player(1, self.board, None, self.params)
        self.players[2] = get_player(2, self.board, None, self.params)
        self.turn_of = self.get_other_player(player_to_start)

    def get_other_player(self, player):
        other_player = [1, 2]
        other_player.remove(player)
        return other_player[0]

    def switch_player(self):
        self.other = self.turn_of
        self.turn_of = self.get_other_player(self.turn_of)

    def check_if_game_finished(self):
        possible_moves = self.board.get_all_possible_moves()
        num_possible_moves = len(possible_moves)
        return (num_possible_moves == 0)

    def find_winner(self):
        self.players[1].get_score().sort()
        self.players[2].get_score().sort()
        for i in range(6):
            if self.players[1].get_score()[i] < self.players[2].get_score()[i]:
                return 2
            elif self.players[2].get_score()[i] < self.players[1].get_score()[i]:
                return 1
        return 0

    def get_initial_request(self):
        initial_request = Request()
        initial_request.add_display_message(None,
            "Please pick inital pieces")
        initial_request.add_update_score(1, self.players[1].get_score())
        initial_request.add_update_score(2, self.players[2].get_score())
        for i in [1, 2]:
            if self.players[i].player_type == "computer":
                initial_request.add_request_pickup_tiles(i, 6)
        # initial_request.add_display_message(None,
        #     "Player {} starts".format(
        #         self.get_other_player(self.turn_of)))
        return initial_request

    def next(self, response):
        request = Request()

        for item in response.action_iterator():
            if item["type"] == "move_made":
                move_made = item["body"]
                self.ingenious, score = self.players[item["player"]].update_score(move_made)
                self.board.update_board(move_made)
                request.add_update_score(item["player"], score)
            elif item["type"] == "tiles_picked_up":
                tiles_picked_up = item["body"]
                self.players[item["player"]].update_deck(tiles_picked_up)
            elif item["type"] == "update_deck":
                new_deck = item["body"]
                self.players[item["player"]].deck.replace_deck(new_deck)
            else:
                raise ValueError('Unrecognised response item.')

        if self.number_moves > 20: # Games normally take 38/39 moves
            game_finished = self.check_if_game_finished()
            if game_finished:
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
            move_output = self.players[self.turn_of].make_move()
            self.ingenious, chosen_move, score, should_exchange = move_output
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
        self.players[1] = get_player(1, self.board, self.tiles, self.params)
        self.players[2] = get_player(2, self.board, self.tiles, self.params)
        self.turn_of = rn.choice([1, 2])
        self.other = self.get_other_player(self.turn_of)
        self.players[self.other].pick_up()
        self.players[self.turn_of].pick_up()

    def get_other_player(self, player):
        other_player = [1, 2]
        other_player.remove(player)
        return other_player[0]

    def switch_player(self):
        self.other = self.turn_of
        self.turn_of = self.get_other_player(self.turn_of)

    def check_if_game_finished(self):
        possible_moves = self.board.get_all_possible_moves()
        num_possible_moves = len(possible_moves)
        return (num_possible_moves == 0)

    def find_winner(self):
        player_1_scores = self.players[1].get_score()
        player_2_scores = self.players[2].get_score()
        player_1_scores.sort()
        player_2_scores.sort()
        for i in range(6):
            if player_1_scores[i] < player_2_scores[i]:
                return 2
            elif player_2_scores[i] < player_1_scores[i]:
                return 1
        return 0

    def get_initial_request(self):
        initial_request = Request()
        initial_request.add_display_message(None, "Picking up initial pieces")
        initial_request.add_update_score(1, self.players[1].get_score())
        initial_request.add_update_score(2, self.players[2].get_score())
        initial_request.add_update_deck(1)
        initial_request.add_update_deck(2)
        initial_request.add_display_message(None, "Player {} starts".format(self.other))
        return initial_request

    def next(self, response):
        request = Request()

        for item in response.action_iterator():
            if item["type"] == "move_made":
                move_made = item["body"]
                self.ingenious, score = self.players[item["player"]].update_score(move_made)
                self.board.update_board(move_made)

                request.add_update_score(item["player"], score)
                move_made = item["body"]
                tile = (move_made.colour1, move_made.colour2)
                self.players[item["player"]].deck.play_tile(tile)

                if not self.ingenious:
                    if self.players[item["player"]].can_exchange_tiles():
                        request.possible_exchange(item["player"])
                    else:
                        self.players[item["player"]].pick_up()
                        request.add_update_deck(item["player"])
            else:
                raise ValueError('Unrecognised response item.')

        if self.number_moves > 20: # Games normally take 38/39 moves
            game_finished = self.check_if_game_finished()
            if game_finished:
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
            move_output = self.players[self.turn_of].make_move()
            self.ingenious, chosen_move, score, should_exchange = move_output
            request.add_make_move(self.turn_of, chosen_move)
            request.add_update_score(self.turn_of, score)
            if not self.ingenious:
                if should_exchange and self.players[self.turn_of].can_exchange_tiles():
                    self.players[self.turn_of].exchange_tiles()
                    request.add_computer_exchange_tiles(self.turn_of)
                else:
                    self.players[self.turn_of].pick_up()
                    request.add_update_deck(self.turn_of)

        self.number_moves += 1
        return request

class TrainingGameplay:
    def __init__(self, params):
        self.params = params
        self.players = {1: None, 2: None}
        self.turn_of = None
        self.other = None
        self.ingenious = False
        self.should_exchange = {1: False, 2: False}
        self.num_moves = 0

    def initialise_game(self, player_1, player_2):
        self.board = Board()
        self.tiles = Tiles()
        self.representation = get_representation(self.params["representation"])
        # self.players[1] = get_player(1, self.board, self.tiles, self.params)
        # self.players[2] = get_player(2, self.board, self.tiles, self.params)
        self.players[1], self.players[2] = player_1, player_2
        self.turn_of = rn.choice([1, 2])
        self.other = self.get_other_player(self.turn_of)
        self.players[self.turn_of].pick_up()
        self.players[self.other].pick_up()

    def get_other_player(self, player):
        other_player = [1, 2]
        other_player.remove(player)
        return other_player[0]

    def switch_player(self):
        buffer = self.other
        self.other = self.turn_of
        self.turn_of = buffer

    def game_has_finished(self):
        possible_moves = self.board.get_all_possible_moves()
        num_possible_moves = len(possible_moves)
        return (num_possible_moves == 0)

    def find_winner(self):
        player_1_scores = self.players[1].get_score()
        player_2_scores = self.players[2].get_score()
        player_1_scores.sort()
        player_2_scores.sort()
        for i in range(6):
            if player_1_scores[i] < player_2_scores[i]:
                return 2
            elif player_2_scores[i] < player_1_scores[i]:
                return 1
        return 0

    def next(self, response):
        if not self.ingenious:
            if self.should_exchange[self.turn_of] and \
                    self.players[self.turn_of].can_exchange_tiles():
                self.players[self.turn_of].exchange_tiles()
            else:
                self.players[self.turn_of].pick_up()

        if self.game_has_finished():
            winner = self.find_winner()
        else:
            winner = None

        if self.ingenious:
            self.ingenious = False
        else:
            self.switch_player()

        move_output = self.players[self.turn_of].make_move()
        self.ingenious, chosen_move, score, should_exchange = move_output
        self.should_exchange[self.turn_of] = should_exchange

        move_representation = self.representation.generate(self.board,
                                                           self.players,
                                                           self.turn_of,
                                                           self.ingenious,
                                                           should_exchange,
                                                           self.move_number)

        return move_representation, winner

    def generate_episode(self, p1, p2):
        while True:
            self.initialise_game(p1, p2)
            moves = []
            winner = None
            while winner is None:
                move_repr, winner, move_num = self.next()
                moves.append(move_repr)

            if winner != 0:
                return calculate_values(moves, move_num, winner)

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
                  "body": move}
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
                  "body": score}
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
