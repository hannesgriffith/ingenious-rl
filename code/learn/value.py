import numpy as np
from numba import jitclass, uint8, float32

def get_value_type(params):
    if params["value_type"] == "v1":
        return ValueV1()
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
