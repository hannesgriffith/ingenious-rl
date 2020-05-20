def get_value_type(params):
    if params["value_type"] == "basic_1":
        return Basic1()
    # elif params["value_type"] == "lambda_return_1":
    #     return LambdaReturn1(params)
    else:
        assert False

class Basic1:
    def __init__(self):
        pass

    def add_values_for_episode(self, episode_representations, winner):
        updated = []
        for move_info in episode_representations:
            grid_input, vector_input, turn_of, move_num = move_info
            value = 1 if int(turn_of) == int(winner) else 0
            new_move_info = (grid_input, vector_input, value)
            updated.append(new_move_info)
        return updated

# class LambdaReturn1:
#     def __init__(self, params):
#         self.lambda = params["lambda"]
#
#     def calculate_value(turn_of, winner, move_num, total_moves):
#         if turn_of == winner:
#             final_reward = 1.
#         else:
#             final_reward = -1.
#
#         assert False # Revisit calculating this and setting lambda
#         moves_until_end = (total_moves - move_num) / 2.
#         discounted_reward = final_reward * (self.lambda ** moves_until_end)
#
#         return discounted_reward
#
#     def add_values_for_episode(self, episode_representation, winner):
#         updated = []
#         total_moves = episode_representation[-1][-1]
#
#         for move_info in episode_representation:
#             grid_input, vector_input, turn_of, move_num = move_info
#             value = calculate_value(turn_of, winner, move_num, total_moves)
#             new_move_info = (grid_input, vector_input, value)
#             updated.append(new_move_info)
#
#         return updated
