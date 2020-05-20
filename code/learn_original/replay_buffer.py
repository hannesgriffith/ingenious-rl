import os

import tqdm

import random as rn
import numpy as np

class ReplayBuffer(object):
    """ Prioritized Experience Replay Buffer """
    def __init__(self, params, work_dir=None):
        self.buffer_size = int(params["replay_buffer_size"])
        self.buffer = []
        self.probs = np.ones((self.buffer_size,), dtype=np.float32) * 0.5

        if work_dir is not None:
            self.work_dir = os.path.join(work_dir, "buffer")
            if not os.path.exists(self.work_dir):
                os.mkdir(self.work_dir)

    def add(self, new_examples):
        self.buffer.extend(new_examples)
        self.probs = np.append(self.probs,
                        np.ones((len(new_examples),)) * np.mean(self.probs))
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]
            self.probs = self.probs[-self.buffer_size:]

    # def sample(self, num_to_sample):
    #     # Old implementation without prioritisation
    #     sample = rn.sample(self.buffer, num_to_sample)
    #     return sample

    def sample(self, num_to_sample):
        # NOTE: can only sample when replay buffer is full
        scaled_probs = self.probs / np.sum(self.probs)
        sampled_idxs = np.random.choice(self.buffer_size, size=num_to_sample,
                                        replace=False, p=scaled_probs).tolist()
        return [self.buffer[i] for i in sampled_idxs], sampled_idxs

    def is_not_full(self):
        return self.get_current_size() < self.buffer_size

    def get_current_size(self):
        return len(self.buffer)

    def update_probs(self, idxs, diffs):
        # print(diffs[:10])
        # print(self.probs[:10])
        for diff_idx, probs_idx in enumerate(idxs):
            self.probs[probs_idx] = diffs[diff_idx]

    def save_buffer_to_file(self):
        print("Saving buffer to {}".format(self.work_dir))
        for idx, item in enumerate(self.buffer):
            x0_name = os.path.join(self.work_dir, "x0_{}.npy".format(idx))
            x1_name = os.path.join(self.work_dir, "x1_{}.npy".format(idx))
            y_name = os.path.join(self.work_dir, "y_{}.npy".format(idx))
            
            np.save(x0_name, item[0])
            np.save(x1_name, item[1])
            np.save(y_name, item[2])

        probs_name = os.path.join(self.work_dir, "probs.npy")
        np.save(probs_name, self.probs)

    def load_buffer_from_file(self, loading_dir):
        print("Loading buffer from {}".format(loading_dir))
        filenames = os.listdir(loading_dir)
        loading_dict = {}
        self.buffer = []

        for filename in tqdm.tqdm(filenames):
            if filename != "probs.npy":
                type_, num  = filename.replace(".npy", "").split("_")
                num = int(num)

                if num not in loading_dict:
                    loading_dict[num] = {}
                loading_dict[num][type_] = np.load(os.path.join(loading_dir, filename))

        for num in tqdm.tqdm(range(len(loading_dict))):
            item = loading_dict[num]
            example = [item["x0"], item["x1"], item["y"]]
            self.buffer.append(example)

        self.probs = np.load(os.path.join(loading_dir, "probs.npy"))
