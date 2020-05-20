def get_value_type(params):
    if params["value_type"] == "basic_1":
        return Value()

    else:
        assert False

class Value:
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
