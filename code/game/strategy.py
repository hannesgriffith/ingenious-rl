import random as rn
from copy import deepcopy

import numpy as np

from game.misc import combine_moves_and_deck

def get_strategy(strategy_type, board, deck, score):
    if strategy_type == "random":
        return RandomStrategy(board, deck, score)
    elif strategy_type == "max":
        return MaxStrategy(board, deck, score)
    elif strategy_type == "increase_min":
        return IncreaseMinStrategy(board, deck, score)
    else:
        print("Invalid strategy_type")
        raise

class RandomStrategy:
    def __init__(self, board, deck, score):
        self.board = board
        self.deck = deck
        self.score = score

    def choose_move(self):
        move_combinations = self.board.get_all_possible_moves()
        possible_moves = combine_moves_and_deck(move_combinations, self.deck)
        random_idx = rn.randint(0, len(possible_moves) - 1)
        should_exchange = rn.choice([True, False])
        return possible_moves[random_idx], should_exchange

class MaxStrategy:
    def __init__(self, board, deck, score):
        self.board = board
        self.deck = deck
        self.score = score

    def choose_move(self):
        move_combinations = self.board.get_all_possible_moves()
        possible_moves = combine_moves_and_deck(move_combinations, self.deck)
        move_scores = [0] * len(possible_moves)
        for idx, move in enumerate(possible_moves):
            move_score = np.array(self.board.calculate_move_score(move))
            original_score = np.array(self.score.get_score())
            score_increase = original_score + move_score
            score_increase[score_increase > 18] = 18
            score_increase -= original_score
            move_scores[idx] = np.sum(score_increase)
        max_score = max(move_scores)
        max_index = move_scores.index(max_score)
        return possible_moves[max_index], True

class IncreaseMinStrategy:
    def __init__(self, board, deck, score):
        self.board = board
        self.deck = deck
        self.score = score

    def choose_move(self):
        should_exchange = True # always exchange when you don't have any of your lowest
        move_combinations = self.board.get_all_possible_moves()
        possible_moves = combine_moves_and_deck(move_combinations, self.deck)
        min_score = self.score.min_score()
        min_scores_idxs = [c for c, s in enumerate(self.score.get_score()) if s == min_score]

        best_moves = []
        best_score = 0
        for move in possible_moves:
            move_score = self.board.calculate_move_score(move)
            min_score = np.sum(np.array([move_score[i] for i in min_scores_idxs]))
            if min_score == best_score:
                best_moves.append((move, move_score))
            elif min_score > best_score:
                best_moves = [(move, move_score)]
                best_score = min_score

        if len(best_moves):
            if len(best_moves) == 1:
                return best_moves[0][0], should_exchange
            else:
                very_best_move_score = 0
                very_best_move = None
                for best_move, best_move_score in best_moves:
                    total_score = np.sum(np.array(best_move_score))
                    if total_score > very_best_move_score:
                        very_best_move = best_move
                        very_best_move_score = total_score
                return very_best_move, should_exchange

        else:
            # Deal with case where you can't increase your lowest score
            # In this case just go for max score
            move_scores = [0] * len(possible_moves)
            for idx, move in enumerate(possible_moves):
                move_score = np.array(self.board.calculate_move_score(move))
                original_score = np.array(self.score.get_score())
                score_increase = original_score + move_score
                score_increase[score_increase > 18] = 18
                score_increase -= original_score
                move_scores[idx] = np.sum(score_increase)
            max_score = max(move_scores)
            max_index = move_scores.index(max_score)
            return possible_moves[max_index], True
