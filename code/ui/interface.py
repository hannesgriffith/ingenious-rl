import numpy as np

class Response:
    def __init__(self):
        self.actions = []

    def add_move_made(self, player, move):
        action = {"player": player,
                  "type": "move_made",
                  "body": move}
        self.actions.append(action)

    def add_tiles_picked_up(self, player, tiles):
        action = {"player": player,
                  "type": "tiles_picked_up",
                  "body": tiles}
        self.actions.append(action)

    def action_iterator(self):
        for action in self.actions:
            yield action

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
            coords_2[0],
            coords_2[1],
            self.c1 - 1,
            self.c2 - 1
        ], dtype=np.uint8)

def display_to_game_tiles(tiles):
    return np.array(tiles, dtype=np.uint8).reshape(-1, 2) - 1

def game_to_display_score(score):
    return score.flatten().tolist()

def game_to_display_move(move):
    coords_1 = game_to_display_coords(move[0:2])
    coords_2 = game_to_display_coords(move[2:4])
    colour_1, colour_2 = move[4:6] + 1
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
        (0, 0): (1, 7),
        (0, 1): (2, 6),
        (0, 2): (3, 5),
        (0, 3): (4, 4),
        (0, 4): (5, 3),
        (0, 5): (6, 2),
        (0, 6): (7, 3),
        (0, 7): (8, 4),
        (0, 8): (9, 5),
        (0, 9): (10, 6),
        (0, 10): (11, 7),
        (1, 0): (1, 9),
        (1, 1): (2, 8),
        (1, 2): (3, 7),
        (1, 3): (4, 6),
        (1, 4): (5, 5),
        (1, 5): (6, 4),
        (1, 6): (7, 5),
        (1, 7): (8, 6),
        (1, 8): (9, 7),
        (1, 9): (10, 8),
        (1, 10): (11, 9),
        (2, 0): (1, 11),
        (2, 1): (2, 10),
        (2, 2): (3, 9),
        (2, 3): (4, 8),
        (2, 4): (5, 7),
        (2, 5): (6, 6),
        (2, 6): (7, 7),
        (2, 7): (8, 8),
        (2, 8): (9, 9),
        (2, 9): (10, 10),
        (2, 10): (11, 11),
        (3, 0): (1, 13),
        (3, 1): (2, 12),
        (3, 2): (3, 11),
        (3, 3): (4, 10),
        (3, 4): (5, 9),
        (3, 5): (6, 8),
        (3, 6): (7, 9),
        (3, 7): (8, 10),
        (3, 8): (9, 11),
        (3, 9): (10, 12),
        (3, 10): (11, 13),
        (4, 0): (1, 15),
        (4, 1): (2, 14),
        (4, 2): (3, 13),
        (4, 3): (4, 12),
        (4, 4): (5, 11),
        (4, 5): (6, 10),
        (4, 6): (7, 11),
        (4, 7): (8, 12),
        (4, 8): (9, 13),
        (4, 9): (10, 14),
        (4, 10): (11, 15),
        (5, 0): (1, 17),
        (5, 1): (2, 16),
        (5, 2): (3, 15),
        (5, 3): (4, 14),
        (5, 4): (5, 13),
        (5, 5): (6, 12),
        (5, 6): (7, 13),
        (5, 7): (8, 14),
        (5, 8): (9, 15),
        (5, 9): (10, 16),
        (5, 10): (11, 17),
        (6, 1): (2, 18),
        (6, 2): (3, 17),
        (6, 3): (4, 16),
        (6, 4): (5, 15),
        (6, 5): (6, 14),
        (6, 6): (7, 15),
        (6, 7): (8, 16),
        (6, 8): (9, 17),
        (6, 9): (10, 18),
        (7, 2): (3, 19),
        (7, 3): (4, 18),
        (7, 4): (5, 17),
        (7, 5): (6, 16),
        (7, 6): (7, 17),
        (7, 7): (8, 18),
        (7, 8): (9, 19),
        (8, 3): (4, 20),
        (8, 4): (5, 19),
        (8, 5): (6, 18),
        (8, 6): (7, 19),
        (8, 7): (8, 20),
        (9, 4): (5, 21),
        (9, 5): (6, 20),
        (9, 6): (7, 21),
        (10, 5): (6, 22)
    }