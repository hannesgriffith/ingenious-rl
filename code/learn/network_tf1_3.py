import numpy as np
import tensorflow as tf

from learn.train_settings import PARAMS

def get_network(params):
    if params["network_type"] == "network1":
        return Network1()
    else:
        assert False

def prepare_example(batch, training=False):
    x1 = np.stack([x[0] for x in batch], axis=0)
    x2 = np.vstack([x[1] for x in batch])
    x1 = x1.astype(np.float32)
    x2 = x2.astype(np.float32)
    x_np = (x1, x2)
    y_np = None
    if training:
        y = np.array([x[2] for x in batch])
        y_np = y.astype(np.float32)
    return x_np, y_np

class Network1():
    def __init__(self):
        self.number_conv_filters = 128  # 64
        self.number_resnet_blocks = 5   # 5
        self.num_hidden_units = 128     # 128
        self.dropout_rate = 0.1
        self.batch_size = PARAMS["batch_size"]
        self.lr = PARAMS["initial_learning_rate"]

        self.optimiser = tf.train.AdamOptimizer(learning_rate=lr)
        self.g = tf.Graph()
        self.build_graph()

    def build_graph(self):
        x1_in = tf.placeholder(tf.float32, shape=(None, 13, 13, 9))
        x2_in = tf.placeholder(tf.float32, shape=(None, 417))

        x1 = tf.layers.conv2d(x1_in, self.number_conv_filters, 3,
                padding='same', use_bias=False,
                kernel_initializer=tf.initializers.uniform_unit_scaling)

        x1 = self.resnet_stack(x1)
        x1 = tf.layers.flatten(x1)

        x3 = tf.concat([x1, x2_in], -1)
        x3 = tf.layers.dense(x3, self.num_hidden_units,
                kernel_initializer=tf.initializers.uniform_unit_scaling)
        # x3 = tf.layers.dropout(x3, rate=self.dropout_rate)
        x3 = tf.nn.relu(x3)

        # x3 = tf.layers.dense(x3, self.num_hidden_units,
        #         kernel_initializer=tf.initializers.uniform_unit_scaling)
        # x3 = tf.layers.dropout(x3, rate=self.dropout_rate)
        # x3 = tf.nn.relu(x3)

        logits = tf.layers.dense(x3, 1,
                kernel_initializer=tf.initializers.uniform_unit_scaling)

        labels = tf.placeholder(tf.float32, shape=(None,))
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                       logits=logits)
        probabilities = tf.math.sigmoid(logits)
        minimize_op = self.optimiser.minimize(loss)

    def resnet_stack(self, x):
        for _ in range(self.number_resnet_blocks):
            x = self.resnet_block(x)
        return x

    def resnet_block(self, x):
        x_in = x
        x = tf.layers.conv2d(x, self.number_conv_filters, 3,
                padding='same', use_bias=False,
                kernel_initializer=tf.initializers.uniform_unit_scaling)
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, self.number_conv_filters, 3,
                padding='same', use_bias=False,
                kernel_initializer=tf.initializers.uniform_unit_scaling)
        x = tf.layers.batch_normalization(x)
        x = x + x_in
        x = tf.nn.relu(x)
        return x

    def training_step(self, x_train, y_train):
        x1, x2 = x_train
        input_dict = {'grid_input': x1, 'vector_input': x2}
        train_loss = self.model.train_on_batch(input_dict, y_train)
        return train_loss[0]

    def __call__(self, x_np):
        return self.predict(x_np)

    def predict(self, x):
        x1, x2 = x
        input_dict = {'grid_input': x1, 'vector_input': x2}
        predictions = self.model.predict_on_batch(input_dict)
        return predictions

    def save_model(self, filepath):
        self.model.save(filepath)

    def save_weights(self, filepath):
        self.model.save_weights(filepath)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
