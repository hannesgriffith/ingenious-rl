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
        score_vector = np.zeros((19,), dtype=np.int32)
        score_vector[colour_score] = 1
        output_vector.append(score_vector)
    return np.concatenate(output_vector, axis=0)

def get_representation(params):
    type_ = params["representation"]
    if type_ == "features_2":
        return RepresentationFeatures2()
    else:
        assert False

class RepresentationFeatures2:
    def __init__(self):
        # in future could add more board shape related features
        pass

    def generate(self, next_board_state, scores, deck, turn_of, ingenious,
                    should_exchange, move_num):
        board_state, board_occupied, board_available = next_board_state
        board_occupied = board_occupied.astype(np.uint8)
        board_available = board_available.astype(np.uint8)

        general_features = generate_general_features(ingenious,
                            should_exchange, board_occupied, board_available)

        colour_specific_features = generate_colour_specific_features(i, scores,
                                                            board_state, deck)

        return (general_features, colour_specific_features, turn_of, move_num)

    def generate_general_features(self, ingenious, should_exchange,
                                  board_occupied, board_available):
        grid_sum = board_occupied.shape[0] * board_occupied.shape[1]
        num_left = np.sum(board_occupied == 0)
        num_playable = np.sum(board_available)
        features = np.array([int(ingenious), int(should_exchange),
                             num_left, num_playable], dtype=np.uint8)
        return features

    def generate_colour_specific_features(self, i, scores, board_state, deck):
        # scores of each player
        player1_score = np.array(scores[0]).reshape(6, 1)
        player2_score = np.array(scores[0]).reshape(6, 1)

        # difference of scores between players
        scores_diff = ((player_1_score + 18) - \
                       (player_2_score + 18)).reshape(6, 1)

        # counts of colours
        colour_counts = [np.sum(board_state == i) for i in range(1, 7)]
        colour_counts = np.array(colour_counts).reshape(6, 1)

        # details of pieces in your hand
        piece_details = np.zeros((6, 2), dtype=np.uint8)
        for tile in deck:
            if tile[0] == tile[1]:
                piece_details[tile[0], 1] += 1
            else:
                piece_details[tile[0], 0] += 1
                piece_details[tile[1], 0] += 1

        # number of playable spaces that score n of given colour (n=1, ..., 10+?)
        # total of any scoring playable space per colour
        

    def generate_grid_input(self, board_state, board_occupied, board_available):
        h, w = board_state.shape
        board_state = np.expand_dims(board_state, 2)
        colours = np.arange(7).reshape(1, 1, 7)
        colours = colours * np.ones((h, w, 7), dtype=np.int32)

        grid_input = np.zeros((h, w, 7), dtype=np.int32)
        grid_input[colours == board_state] = 1

        board_occupied = np.expand_dims(board_occupied, 2)
        board_available = np.expand_dims(board_available, 2)
        grid_input = np.concatenate([grid_input, board_occupied,
                                     board_available], 2)

        return grid_input

    def generate_vector_input(self, ingenious, should_exchange, scores, deck):
        vector_input = []
        vector_input.append(int(ingenious))
        vector_input.append(int(should_exchange))

        score1, score2 = scores
        score3 = (np.array(score2) - np.array(score1)).tolist()
        vector_input.extend(score_to_vector(score1).tolist())
        vector_input.extend(score_to_vector(score2).tolist())
        vector_input.extend(score_to_vector(score3).tolist())

        tile_vectors = []
        if len(deck) > 0:
            for tile in deck:
                tile_vectors.append(colour_to_vector(tile[0]))
                tile_vectors.append(colour_to_vector(tile[1]))
        while len(tile_vectors) < 12:
            tile_vectors.append(np.zeros((6,), dtype=np.int32).tolist())
        for tile_vector in tile_vectors:
            vector_input.extend(tile_vector)

        return np.array(vector_input)
