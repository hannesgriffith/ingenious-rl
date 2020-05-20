from numba import jitclass, float32, int32
import numpy as np

def get_exploration_policy(params):
    if params["exploration"] == "v1":
        return ExplorationV1()
    if params["exploration"] == "v2":
        return ExplorationV2(params["total_training_steps"])
    else:
        raise ValueError("Incorrect exploration policy name.")

exploration_spec_1 = [
    ('eps', float32),
    ('temp', float32),
]

@jitclass(exploration_spec_1)
class ExplorationV1:
    def __init__(self):
        self.eps = 0.0
        self.temp = 1.0

    def get_params(self, step):
        return self.eps, self.temp

exploration_spec_2 = [
    ('total_training_steps', float32),
    ('eps_start_val', float32),
    ('eps_end_val', float32),
    ('eps_decay_end_fraction', float32),
    ('eps_use_end_fraction', float32),
    ('temp_start_val', float32),
    ('temp_end_val', float32),
    ('temp_decay_end_fraction', float32),
    ('temp_use_end_fraction', float32),
    ('eps_decay_end_step', float32),
    ('eps_use_end_step', float32),
    ('temp_decay_end_step', float32),
    ('temp_use_end_step', float32),
]

@jitclass(exploration_spec_2)
class ExplorationV2:
    def __init__(self, total_training_steps):
        self.total_training_steps = total_training_steps

        # self.eps_start_val = 1. / 10.
        # self.eps_end_val = 0.0
        # self.eps_decay_end_fraction = 0.5
        # self.eps_use_end_fraction =  0.5

        # self.temp_start_val = 1.
        # self.temp_end_val = 3.
        # self.temp_decay_end_fraction = 0.5
        # self.temp_use_end_fraction = 0.5

        # self.eps_decay_end_step = self.total_training_steps * self.eps_decay_end_fraction
        # self.eps_use_end_step = self.total_training_steps * self.eps_use_end_fraction
        # self.temp_decay_end_step = self.total_training_steps * self.temp_decay_end_fraction
        # self.temp_use_end_step = self.total_training_steps * self.temp_use_end_fraction

        self.eps_start_val = 1. / 20.
        self.eps_end_val = 0.0
        self.eps_decay_end_fraction = 0.5
        self.eps_use_end_fraction =  0.5

        self.temp_start_val = 2.
        self.temp_end_val = 8.
        self.temp_decay_end_fraction = 0.5
        self.temp_use_end_fraction = 0.5

        self.eps_decay_end_step = self.total_training_steps * self.eps_decay_end_fraction
        self.eps_use_end_step = self.total_training_steps * self.eps_use_end_fraction
        self.temp_decay_end_step = self.total_training_steps * self.temp_decay_end_fraction
        self.temp_use_end_step = self.total_training_steps * self.temp_use_end_fraction

    def get_params(self, step):
        step = float(step)

        if step >= self.eps_use_end_step:
            eps = 0.0
        elif step >= self.eps_decay_end_step:
            eps = self.eps_end_val
        else:
            eps = self.eps_start_val - (1. - (self.eps_decay_end_step - step) / self.eps_decay_end_step) * (self.eps_start_val - self.eps_end_val)

        if step >= self.temp_use_end_step:
            temp = None
        elif step >= self.temp_decay_end_step:
            temp = self.temp_end_val
        else:
            temp = self.temp_start_val + (1. - (self.temp_decay_end_step - step) / self.temp_decay_end_step) * (self.temp_end_val - self.temp_start_val)

        return eps, temp
