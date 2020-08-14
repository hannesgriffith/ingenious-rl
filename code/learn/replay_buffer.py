from numba import jitclass, njit, uint8, int32
import numpy as np

from learn.representation import get_representation

class ReplayBuffer:
    def __init__(self, params, work_dir=None):
        self.buffer_size = int(params["replay_buffer_size"])
        self.batch_size = min([
            int(params["effective_batch_size"]),
            int(params["max_train_batch_size"])
        ])

        self.buffer = get_representation(params).get_new_reprs_buffer()

    def __len__(self):
        return self.buffer.size

    def is_not_full(self):
        return self.__len__() < self.buffer_size

    def add(self, new_reprs):
        self.buffer.combine_reprs(new_reprs)
        if self.buffer.size > self.buffer_size:
            self.buffer.clip_to_size(self.buffer_size)

    def sample_examples(self, num_to_sample):
        sampled_idxs = np.random.choice(self.buffer_size, size=num_to_sample, replace=False)
        examples, labels = self.buffer.get_examples_by_idxs(sampled_idxs)
        return examples, labels, sampled_idxs

    def preprocess(self, inputs, labels):
        inputs_augmented = self.buffer.augment(*inputs)
        inputs_normalised = self.buffer.normalise(*inputs_augmented)
        inputs_prepared = self.buffer.prepare(*inputs_normalised)
        labels = labels.astype(np.int32)
        return inputs_prepared, inputs_prepared, labels

    def sample_training_minibatch(self):
        examples, labels, sampled_idxs = self.sample_examples(self.batch_size)
        examples, vis_examples, labels = self.preprocess(examples, labels)
        return examples, labels, vis_examples, sampled_idxs