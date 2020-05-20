def get_exploration_policy(params):
    if params["exploration"] == "exploration1":
        return Exploration1(params)
    else:
        assert False

class Exploration1:
    def __init__(self, params):
        self.total_num_steps = params["total_training_steps"]

        self.eps_start_val = 1. / 40.
        self.eps_end_val = 1. / 40.
        self.stop_decaying_eps_step = int(40000)
        self.stop_using_decay_after = int(40000)

        self.temp_start_val = 0.
        self.temp_end_val = 2.
        self.stop_decaying_temp_step = int(-1)
        self.stop_using_temp_after = int(-1)

    def get_params(self, step):
        if step > self.stop_using_decay_after:
            eps = 1e-4
        elif step > self.stop_decaying_eps_step:
            eps = self.eps_end_val
        else:
            eps = max(((float(self.stop_decaying_eps_step) - step) / \
                        float(self.stop_decaying_eps_step)) \
                        * self.eps_start_val, self.eps_end_val)

        if step > self.stop_using_temp_after:
            temp = None
        elif step > self.stop_decaying_temp_step:
            temp = self.temp_end_val
        else:
            temp = min((1. - ((float(self.stop_decaying_temp_step) - step) / \
                        float(self.stop_decaying_temp_step))) \
                        * self.temp_end_val, self.temp_end_val)

        return eps, temp
