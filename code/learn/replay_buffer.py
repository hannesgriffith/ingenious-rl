import numpy as np

from learn.representation import RepresentationsBuffer

class ReplayBuffer:
    def __init__(self, params):
        self.buffer_size = int(params["replay_buffer_size"])
        self.batch_size = min([
            int(params["effective_batch_size"]),
            int(params["max_train_batch_size"])
        ])

        self.buffer = RepresentationsBuffer()

    def __len__(self):
        return self.buffer.size

    def is_not_full(self):
        return self.__len__() < self.buffer_size

    def add(self, new_reprs):
        self.buffer.combine_reprs(new_reprs)
        if self.buffer.size > self.buffer_size:
            self.buffer.clip_to_size(self.buffer_size)

    def sample_training_minibatch(self):
        sampled_idxs = np.random.choice(self.buffer_size, size=self.batch_size, replace=False)
        examples, labels = self.buffer.get_examples_by_idxs(sampled_idxs)
        examples = self.buffer.preprocess(examples)
        labels = labels.astype(np.int32)
        return examples, labels