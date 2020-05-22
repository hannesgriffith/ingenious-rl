import numpy as np
from numba import jitclass, uint8, float32

def get_value_type(params):
    if params["value_type"] == "v1":
        return Value()
    else:
        raise ValueError("Incorrect value function name.")

value_spec = [
    ("version", uint8)
]

@jitclass(value_spec)
class Value:
    def __init__(self):
        self.version = 1

    def add_values_for_episode(self, representations, winner, last_turn_player):
        if last_turn_player == winner:
            self_value = 1
            other_value = 0
        else:
            self_value = 0
            other_value = 1

        representations[2].values_repr[-1 , 0] = self_value
        representations[3].values_repr[-1 , 0] = other_value
        return representations
