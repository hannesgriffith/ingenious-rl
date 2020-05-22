from numba import njit
import numpy as np

from learn.representation import get_representation

class ReplayBuffer(object):
    def __init__(self, params, work_dir=None):
        self.temp = float(params["replay_buffer_temp"])
        self.batch_size = int(params["batch_size"])
        self.buffer_size = int(params["replay_buffer_size"])

        self.buffer_before = get_representation(params).get_new_reprs_buffer()
        self.buffer_after = get_representation(params).get_new_reprs_buffer()
        self.probs = np.ones(self.buffer_size, dtype=np.float32)

    def __len__(self):
        assert self.buffer_before.size == self.buffer_after.size
        return self.buffer_before.size

    def is_not_full(self):
        return self.__len__() < self.buffer_size

    def add(self, new_reprs):
        self.buffer_before.combine_reprs(new_reprs[0])
        self.buffer_after.combine_reprs(new_reprs[1])
        self.probs = np.concatenate((
            np.ones(new_reprs[0].size, dtype=np.float32) * np.mean(self.probs),
            self.probs
        ))

        if self.buffer_before.size > self.buffer_size:
            self.buffer_before.clip_to_size(self.buffer_size)
            self.buffer_after.clip_to_size(self.buffer_size)
            self.probs = self.probs[:self.buffer_size]

    def update_probs(self, idxs, diffs):
        self.probs[idxs] = diffs

    def sample_examples(self, num_to_sample):
        scaled_probs = self.probs ** self.temp
        probs_norm = (scaled_probs / np.sum(scaled_probs))
        sampled_idxs = np.random.choice(self.buffer_size, size=num_to_sample, replace=False, p=probs_norm)
        before_states, _ = self.buffer_before.get_examples_by_idxs(sampled_idxs)
        after_states, labels = self.buffer_after.get_examples_by_idxs(sampled_idxs)
        return before_states, after_states, labels, sampled_idxs

    def preprocess(self, before_states, after_states, labels):
        states_augmented = self.buffer_before.augment(before_states, after_states)
        states_normalised = self.buffer_before.normalise2(*states_augmented)
        states_prepared = self.buffer_before.prepare2(*states_normalised)
        labels = labels.astype(np.int32)
        return states_prepared, states_augmented, labels

    def sample_training_minibatch(self):
        before_states, after_states, labels, sampled_idxs = self.sample_examples(self.batch_size)
        examples, vis_examples, labels = self.preprocess(before_states, after_states, labels)
        return examples, labels, vis_examples, sampled_idxs
