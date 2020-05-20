import numpy as np

from game.board import combine_moves_and_deck
# from learn.network_tf1_2 import prepare_example

def get_strategy(strategy_type):
    if strategy_type == "random":
        return RandomStrategy()
    elif strategy_type == "max":
        return MaxStrategy()
    elif strategy_type == "increase_min":
        return IncreaseMinStrategy()
    # elif strategy_type == "rl1":
    #     return RL1()
    else:
        raise Exception("Invalid strategy_type")

# def get_other_player(turn_of):
#     other_player = [1, 2]
#     other_player.remove(turn_of)
#     return other_player[0]

class RandomStrategy:
    def __init__(self):
        pass

    def choose_move(self, board, deck, score, players, player_num, repr_fn):
        move_combinations = board.get_all_possible_moves()
        possible_moves = combine_moves_and_deck(move_combinations, deck)
        random_idx = np.random.choice(possible_moves.shape[0])
        should_exchange = np.random.choice([True, False])
        return (possible_moves[random_idx], should_exchange), 0.5

class MaxStrategy:
    def __init__(self):
        pass

    def choose_move(self, board, deck, score, players, player_num, repr_fn):
        move_combinations = board.get_all_possible_moves()
        possible_moves = combine_moves_and_deck(move_combinations, deck)
        original_score = score.get_score().reshape(1, 6)
        move_scores = board.batch_calculate_move_scores(possible_moves)
        updated_scores, _, _ = score.batch_peek_next_scores(move_scores)
        scores_diff = updated_scores - original_score
        total_scores_diff = np.sum(scores_diff, axis=1)
        max_index = np.argmax(total_scores_diff, axis=0)
        return (possible_moves[max_index], True), 0.5

class IncreaseMinStrategy:
    def __init__(self):
        pass

    def choose_move(self, board, deck, score, players, player_num, repr_fn):
        move_combinations = board.get_all_possible_moves()
        possible_moves = combine_moves_and_deck(move_combinations, deck)

        original_score = score.get_score()
        min_score = np.min(original_score)
        min_idxs = np.where(original_score == min_score)[0]
        num_min_idxs = min_idxs.shape[0]

        moves_scores = board.batch_calculate_move_scores(possible_moves)
        updated_move_scores, _, _ = score.batch_peek_next_scores(moves_scores)
        min_scores = updated_move_scores[:, min_idxs].reshape(-1, num_min_idxs)
        total_min_scores = np.sum(min_scores, axis=1)
        max_total_min_score = np.max(total_min_scores)

        if max_total_min_score > 0:

            max_total_min_scores_idxs = np.where(total_min_scores == max_total_min_score)[0]

            if max_total_min_scores_idxs.shape[0] == 1:
                max_index = max_total_min_scores_idxs[0]
                return (possible_moves[max_index], True), 0.5

            best_moves_so_far = possible_moves[max_total_min_scores_idxs, :]
            best_moves_updated_scores = updated_move_scores[max_total_min_scores_idxs, :]
            scores_diff = best_moves_updated_scores - original_score.reshape(1, 6)
            total_scores_diff = np.sum(scores_diff, axis=1)
            max_index = np.argmax(total_scores_diff, axis=0)
            return (best_moves_so_far[max_index], True), 0.5

        else:
            # Deal with case where you can't increase your lowest score
            # In this case just go for max score
            move_scores = board.batch_calculate_move_scores(possible_moves)
            updated_scores, _, _ = score.batch_peek_next_scores(move_scores)
            scores_diff = updated_scores - original_score.reshape(1, 6)
            total_scores_diff = np.sum(scores_diff, axis=1)
            max_index = np.argmax(total_scores_diff, axis=0)
            return (possible_moves[max_index], True), 0.5

class IncreaseOtherMinStrategy:
    def __init__(self):
        pass

    def choose_move(self, board, deck, score, players, player_num, repr_fn):
        pass

class ReduceDifferenceStrategy:
    def __init__(self):
        pass

    def choose_move(self, board, deck, score, players, player_num, repr_fn):
        pass

class RLSupervisedTraining:
    def __init__(self):
        pass

    def choose_move(self, board, deck, score, players, player_num, repr_fn):
        pass

class RLSelfTraining:
    def __init__(self):
        pass

    def choose_move(self, board, deck, score, players, player_num, repr_fn):
        pass

# class RL1:
#     def __init__(self):
#         self.explore = False
#         self.eps = 0.0
#         self.temp = 0.0
#         self.model = None

#     def set_explore(self, explore):
#         self.explore = explore

#     def set_explore_params(self, eps, temp):
#         self.eps = eps
#         self.temp = temp

#     def set_model(self, model):
#         self.model = model

#     def choose_move(self, board, deck, score, players, player_num, repr_fn):
#         move_combinations = board.get_all_possible_moves()
#         possible_moves = combine_moves_and_deck(move_combinations, deck)
#         all_moves = []
#         test_inputs = []

#         for move in possible_moves:
#             next_board_state = board.peak_board_update(move)
#             nextState, nextOccupied, nextAvailable = next_board_state
#             next_move_score = board.peek_move_score(nextState, move)
#             next_score, ingenious = players[player_num].score.peek_next_score(next_move_score)
#             tile_to_play = (move.colour1, move.colour2)
#             deck_next = players[player_num].deck.peek_next_deck(tile_to_play)
#             other_player = get_other_player(player_num)
#             score_other = players[other_player].score.get_score_copy()
#             scores = (next_score, score_other)
#             can_exchange = players[player_num].peek_can_exchange_tiles(deck_next, next_score)

#             network_input = repr_fn(next_board_state,
#                                     scores,
#                                     deck_next,
#                                     player_num,
#                                     ingenious,
#                                     False,
#                                     board.move_num)[0]

#             test_input = (network_input[0], network_input[1])
#             test_inputs.append(test_input)
#             all_moves.append((move, False))

#             if can_exchange:
#                 network_input = repr_fn(next_board_state,
#                                         scores,
#                                         deck_next,
#                                         player_num,
#                                         ingenious,
#                                         True,
#                                         board.move_num)[0]

#                 test_input = (network_input[0], network_input[1])
#                 test_inputs.append(test_input)
#                 all_moves.append((move, True))

#         x_np, _ = prepare_example(test_inputs, training=False)
#         move_values = self.model(x_np)
#         move_values = np.squeeze(move_values)

#         if not self.explore:
#             best_value = np.max(move_values)
#             # print("Computer move value:", round(best_value, 2))
#             best_move_idx = np.argmax(move_values)
#             return all_moves[best_move_idx], best_value

#         num_moves = len(all_moves)
#         random_val = rn.uniform(0, 1)
#         if random_val <= self.eps:
#             random_idx = rn.randrange(num_moves)
#             return all_moves[random_idx], move_values[random_idx]

#         if self.temp is None:
#             best_value = np.max(move_values)
#             best_move_idx = np.argmax(move_values)
#             return all_moves[best_move_idx], best_value
#         else:
#             move_values = np.array(move_values).astype(np.float32)
#             scaled = move_values ** self.temp
#             probs = scaled / np.sum(scaled)
#             sampled_idx = np.random.choice(num_moves, p=probs)
#             return all_moves[sampled_idx], move_values[sampled_idx]
