import numpy as np
from numba import jitclass, uint8, float32

def get_value_type(params):
    if params["value_type"] == "v1":
        return ValueV1()
    if params["value_type"] == "v2":
        return ValueV2(params["lambda"])
    else:
        raise ValueError("Incorrect value function name.")

value_spec_v1 = [
    ("version", uint8)
]

@jitclass(value_spec_v1)
class ValueV1:
    def __init__(self):
        self.version = 1

    def add_values_for_episode(self, representations, winner):
        representations.values_repr[representations.turn_of_repr == winner] = 1
        return representations

value_spec_v2 = [
    ("version", uint8),
    ("l", float32)
]

@jitclass(value_spec_v2)
class ValueV2:
    def __init__(self, l):
        self.version = 2
        self.l = l

    def add_values_for_episode(self, representations, winner):
        representations.values_repr[representations.turn_of_repr == winner, 0] = 1.

        move_num = representations.general_repr[:, 4].astype(np.float32)
        moves_from_end = np.max(move_num) - move_num - 1.
        moves_from_end[moves_from_end < 0.] = 0.
        credit = self.l ** moves_from_end
        representations.values_repr[:, 1] = credit

        return representations
