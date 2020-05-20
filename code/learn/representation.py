from copy import deepcopy

import numpy as np

def get_other_player(player):
    other_player = [1, 2]
    other_player.remove(player)
    return other_player[0]

def colour_to_vector(colour):
    vector = np.zeros((6,), dtype=np.int32)
    vector[colour - 1] = 1
    return vector.tolist()

def invert_binary(input):
    output = np.ones(input.shape, dtype=np.int32)
    output[input == 1] = 0
    return output

def score_to_vector(scores_list):
    output_vector = []
    for colour_score in scores_list:
        score_vector = np.zeros((18,), dtype=np.int32)
        score_vector[colour_score] = 1
        output_vector.append(score_vector)
    return np.concatenate(output_vector, axis=0)

def get_representation(type_):
    if type_ == "vanilla":
        return RepresentationVanilla()
    elif type_ == "vanilla_one_hot":
        return RepresentationVanillaOneHot()
    elif type_ == "features_1":
        return RepresentationFeatures1()
    elif type_ == "features_1_one_hot":
        return RepresentationFeatures1OneHot()
    else:
        assert False

class RepresentationVanilla:
    def __init__(self):
        pass

    def generate(board, players, turn_of, ingenious, should_exchange, move_num):
        board_state = np.expand_dims(np.copy(board.state), 2)
        board_occupied = invert_binary(np.copy(board.occupied))
        grid_input = self.generate_grid_input(board_state, board_occupied)
        vector_input = self.generate_vector_input(players, turn_of, ingenious,
                        should_exchange, move_num)
        return (grid_input, vector_input, turn_of, move_num)

    def generate_grid_input(self, board_state, board_occupied):
        h, w = board_state.shape
        colours = np.arange(7).reshape(1, 1, 7) * np.ones((h, w, 7),
                    dtype=np.int32)
        grid_input = np.zeros((h, w, 7), dtype=np.int32)
        grid_input[grid_input == board_state] = 1
        grid_input[:, :, 0] = board_occupied
        return grid_input

    def generate_vector_input(self, players, turn_of, ingenious,
            should_exchange, move_num):
        vector_input = []
        vector_input.extend(int(ingenious))
        vector_input.extend(int(should_exchange))
        vector_input.extend(int(move_num))
        vector_input.extend(deepcopy(players[turn_of].get_score()))
        vector_input.extend(
            deepcopy(players[get_other_player(turn_of)].get_score())
            )

        player_deck = deepcopy(players[turn_of].deck.deck)
        tile_vectors = []
        for tile in player_deck:
            tile_vectors.append(colour_to_vector(tile[0]))
            tile_vectors.append(colour_to_vector(tile[1]))
        while len(tile_vectors) < 12:
            tile_vectors.append(np.zeros((6,), dtype=np.int32).tolist())
        for tile_vector in tile_vectors:
            vector_input.extend(tile_vector)

        return np.array(vector_input)

class RepresentationVanillaOneHot:
    def __init__(self):
        pass

    def generate(board, players, turn_of, ingenious, should_exchange, move_num):
        board_state = np.expand_dims(np.copy(board.state), 2)
        board_occupied = invert_binary(np.copy(board.occupied))
        grid_input = self.generate_grid_input(board_state, board_occupied)
        vector_input = self.generate_vector_input(players, turn_of, ingenious,
                        should_exchange, move_num)
        return (grid_input, vector_input, turn_of, move_num)

    def generate_grid_input(self, board_state, board_occupied):
        h, w = board_state.shape
        colours = np.arange(7).reshape(1, 1, 7) * np.ones((h, w, 7),
                    dtype=np.int32)
        grid_input = np.zeros((h, w, 7), dtype=np.int32)
        grid_input[grid_input == board_state] = 1
        grid_input[:, :, 0] = board_occupied
        return grid_input

    def generate_vector_input(self, players, turn_of, ingenious,
            should_exchange, move_num):
        vector_input = []
        vector_input.extend(int(ingenious))
        vector_input.extend(int(should_exchange))
        vector_input.extend(int(move_num))

        score1 = deepcopy(players[turn_of].get_score())
        score2 = deepcopy(players[get_other_player(turn_of)].get_score())
        score1_one_hot = score_to_vector(score1)
        score2_one_hot = score_to_vector(score2)
        vector_input.extend(score1_one_hot.tolist())
        vector_input.extend(score2_one_hot.tolist())

        player_deck = deepcopy(players[turn_of].deck.deck)
        tile_vectors = []
        if len(player_deck) > 0:
            for tile in player_deck:
                tile_vectors.append(colour_to_vector(tile[0]))
                tile_vectors.append(colour_to_vector(tile[1]))
        while len(tile_vectors) < 12:
            tile_vectors.append(np.zeros((6,), dtype=np.int32).tolist())
        for tile_vector in tile_vectors:
            vector_input.extend(tile_vector)

        return np.array(vector_input)

class RepresentationFeatures1:
    def __init__(self, type_):
        pass

    def generate(board, players, turn_of, ingenious, should_exchange, move_num):
        board_state = np.expand_dims(np.copy(board.state), 2)
        board_occupied = invert_binary(np.copy(board.occupied))
        board_available = np.copy(board.available)
        grid_input = self.generate_grid_input(board_state, board_occupied,
                        board_available)
        vector_input = self.generate_vector_input(players, turn_of, ingenious,
                        should_exchange, move_num)
        return (grid_input, vector_input, turn_of, move_num)

    def generate_grid_input(self, board_state, board_occupied):
        h, w = board_state.shape
        colours = np.arange(7).reshape(1, 1, 7) * np.ones((h, w, 7),
                    dtype=np.int32)
        grid_input = np.zeros((h, w, 7), dtype=np.int32)
        grid_input[grid_input == board_state] = 1
        grid_input[:, :, 0] = board_occupied
        available = np.expand_dims(available, 2)
        grid_input = np.concatenate([grid_input, available], 2)
        return grid_input

    def generate_vector_input(self, players, turn_of, ingenious,
            should_exchange, move_num):
        vector_input = []
        vector_input.extend(int(ingenious))
        vector_input.extend(int(should_exchange))
        vector_input.extend(int(move_num))

        score1 = deepcopy(players[turn_of].get_score())
        score2 = deepcopy(players[get_other_player(turn_of)].get_score())
        score3 = (np.array(score2) - np.array(score1)).tolist()
        vector_input.extend(score1)
        vector_input.extend(score2)
        vector_input.extend(score3)

        player_deck = deepcopy(players[turn_of].deck.deck)
        tile_vectors = []
        for tile in player_deck:
            tile_vectors.append(colour_to_vector(tile[0]))
            tile_vectors.append(colour_to_vector(tile[1]))
        while len(tile_vectors) < 12:
            tile_vectors.append(np.zeros((6,), dtype=np.int32).tolist())
        for tile_vector in tile_vectors:
            vector_input.extend(tile_vector)

        return np.array(vector_input)

class RepresentationFeatures1OneHot:
    def __init__(self, type_):
        pass

    def generate(board, players, turn_of, ingenious, should_exchange, move_num):
        board_state = np.expand_dims(np.copy(board.state), 2)
        board_occupied = invert_binary(np.copy(board.occupied))
        board_available = np.copy(board.available)
        grid_input = self.generate_grid_input(board_state, board_occupied,
                        board_available)
        vector_input = self.generate_vector_input(players, turn_of, ingenious,
                        should_exchange, move_num)
        return (grid_input, vector_input, turn_of, move_num)

    def generate_grid_input(self, board_state, board_occupied):
        h, w = board_state.shape
        colours = np.arange(7).reshape(1, 1, 7) * np.ones((h, w, 7),
                    dtype=np.int32)
        grid_input = np.zeros((h, w, 7), dtype=np.int32)
        grid_input[grid_input == board_state] = 1
        grid_input[:, :, 0] = board_occupied
        available = np.expand_dims(available, 2)
        grid_input = np.concatenate([grid_input, available], 2)
        return grid_input

    def generate_vector_input(self, players, turn_of, ingenious,
                                should_exchange, move_num):
        vector_input = []
        vector_input.extend(int(ingenious))
        vector_input.extend(int(should_exchange))
        vector_input.extend(int(move_num))

        score1 = deepcopy(players[turn_of].get_score())
        score2 = deepcopy(players[get_other_player(turn_of)].get_score())
        score3 = (np.array(score2) - np.array(score1)).tolist()
        vector_input.extend(score_to_vector(score1).tolist())
        vector_input.extend(score_to_vector(score2).tolist())
        vector_input.extend(score_to_vector(score3).tolist())

        player_deck = deepcopy(players[turn_of].deck.deck)
        tile_vectors = []
        if len(player_deck) > 0:
            for tile in player_deck:
                tile_vectors.append(colour_to_vector(tile[0]))
                tile_vectors.append(colour_to_vector(tile[1]))
        while len(tile_vectors) < 12:
            tile_vectors.append(np.zeros((6,), dtype=np.int32).tolist())
        for tile_vector in tile_vectors:
            vector_input.extend(tile_vector)

        return np.array(vector_input)
