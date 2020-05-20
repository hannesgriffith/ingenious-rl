import random as rn
from copy import deepcopy

import numpy as np

from game.misc import combine_moves_and_deck
from learn.network_tf1_2 import prepare_example

def get_strategy(strategy_type):
    if strategy_type == "random":
        return RandomStrategy()
    elif strategy_type == "max":
        return MaxStrategy()
    elif strategy_type == "increase_min":
        return IncreaseMinStrategy()
    elif strategy_type == "rl1":
        return RL1()
    else:
        print("Invalid strategy_type")
        raise

def get_other_player(turn_of):
    other_player = [1, 2]
    other_player.remove(turn_of)
    return other_player[0]

class RandomStrategy:
    def __init__(self):
        pass

    def choose_move(self, board, deck, score, players, player_num, repr_fn):
        move_combinations = board.get_all_possible_moves()
        possible_moves = combine_moves_and_deck(move_combinations, deck)
        random_idx = rn.randint(0, len(possible_moves) - 1)
        should_exchange = rn.choice([True, False])
        return (possible_moves[random_idx], should_exchange), 0

class MaxStrategy:
    def __init__(self):
        pass

    def choose_move(self, board, deck, score, players, player_num, repr_fn):
        move_combinations = board.get_all_possible_moves()
        possible_moves = combine_moves_and_deck(move_combinations, deck)
        move_scores = [0] * len(possible_moves)
        for idx, move in enumerate(possible_moves):
            move_score = np.array(board.calculate_move_score(move))
            original_score = np.array(score.get_score())
            score_increase = original_score + move_score
            score_increase[score_increase > 18] = 18
            score_increase -= original_score
            move_scores[idx] = np.sum(score_increase)
        max_score = max(move_scores)
        max_index = move_scores.index(max_score)
        return (possible_moves[max_index], True), 0

class IncreaseMinStrategy:
    def __init__(self):
        pass

    def choose_move(self, board, deck, score, players, player_num, repr_fn):
        should_exchange = True # always exchange when you don't have any of your lowest
        move_combinations = board.get_all_possible_moves()
        possible_moves = combine_moves_and_deck(move_combinations, deck)
        min_score = score.min_score()
        min_scores_idxs = [c for c, s in enumerate(score.get_score()) if s == min_score]

        best_moves = []
        best_score = -1
        for move in possible_moves:
            move_score = board.calculate_move_score(move)
            min_score = np.sum(np.array([move_score[i] for i in min_scores_idxs]))
            if min_score == best_score:
                best_moves.append((move, move_score))
            elif min_score > best_score:
                best_moves = [(move, move_score)]
                best_score = min_score

        if len(best_moves):
            if len(best_moves) == 1:
                return (best_moves[0][0], should_exchange), 0
            else:
                very_best_move_score = -1
                very_best_move = best_moves[0][0]
                for best_move, best_move_score in best_moves:
                    total_score = np.sum(np.array(best_move_score))
                    if total_score > very_best_move_score:
                        very_best_move = best_move
                        very_best_move_score = total_score
                return (very_best_move, should_exchange), 0

        else:
            # Deal with case where you can't increase your lowest score
            # In this case just go for max score
            move_scores = [0] * len(possible_moves)
            for idx, move in enumerate(possible_moves):
                move_score = np.array(board.calculate_move_score(move))
                original_score = np.array(score.get_score())
                score_increase = original_score + move_score
                score_increase[score_increase > 18] = 18
                score_increase -= original_score
                move_scores[idx] = np.sum(score_increase)
            max_score = max(move_scores)
            max_index = move_scores.index(max_score)
            return (possible_moves[max_index], True), 0

class RL1:
    def __init__(self):
        self.explore = False
        self.eps = 0.0
        self.temp = 0.0
        self.model = None

    def set_explore(self, explore):
        self.explore = explore

    def set_explore_params(self, eps, temp):
        self.eps = eps
        self.temp = temp

    def set_model(self, model):
        self.model = model

    def choose_move(self, board, deck, score, players, player_num, repr_fn):
        move_combinations = board.get_all_possible_moves()
        possible_moves = combine_moves_and_deck(move_combinations, deck)
        all_moves = []
        test_inputs = []

        for move in possible_moves:
            next_board_state = board.peak_board_update(move)
            nextState, _, _ = next_board_state
            next_move_score = board.peak_move_score(nextState, move)
            next_score, ingenious = players[player_num].score.peak_next_score(next_move_score)
            tile_to_play = (move.colour1, move.colour2)
            deck_next = players[player_num].deck.peak_next_deck(tile_to_play)
            other_player = get_other_player(player_num)
            score_other = players[other_player].score.get_score_copy()
            scores = (next_score, score_other)
            can_exchange = players[player_num].peak_can_exchange_tiles(deck_next, next_score)

            network_input = repr_fn(next_board_state,
                                    scores,
                                    deck_next,
                                    player_num,
                                    ingenious,
                                    False,
                                    board.move_num)[0]

            test_input = (network_input[0], network_input[1])
            test_inputs.append(test_input)
            all_moves.append((move, False))

            if can_exchange:
                network_input = repr_fn(next_board_state,
                                        scores,
                                        deck_next,
                                        player_num,
                                        ingenious,
                                        True,
                                        board.move_num)[0]

                test_input = (network_input[0], network_input[1])
                test_inputs.append(test_input)
                all_moves.append((move, True))

        x_np, _ = prepare_example(test_inputs, training=False)
        move_values = self.model(x_np)
        move_values = np.squeeze(move_values)

        if not self.explore:
            best_value = np.max(move_values)
            # print("Computer move value:", round(best_value, 2))
            best_move_idx = np.argmax(move_values)
            return all_moves[best_move_idx], best_value

        num_moves = len(all_moves)
        random_val = rn.uniform(0, 1)
        if random_val <= self.eps:
            random_idx = rn.randrange(num_moves)
            return all_moves[random_idx], move_values[random_idx]

        if self.temp is None:
            best_value = np.max(move_values)
            best_move_idx = np.argmax(move_values)
            return all_moves[best_move_idx], best_value
        else:
            move_values = np.array(move_values).astype(np.float32)
            scaled = move_values ** self.temp
            probs = scaled / np.sum(scaled)
            sampled_idx = np.random.choice(num_moves, p=probs)
            return all_moves[sampled_idx], move_values[sampled_idx]
