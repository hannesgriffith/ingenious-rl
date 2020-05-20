import numpy as np
import tensorflow as tf

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
    def __init__(self, params):
        self.num_conv_channels = 128
        self.num_fc_channels = 256
        self.num_resnet_blocks = 5
        self.num_input_channels = 9

        self.lr = params["initial_learning_rate"]
        self.optimizer = tf.train.AdamOptimizer(self.lr)

    def resnet_block(self, x):
        x_in = tf.identity(x)

        x = tf.compat.v1.layers.conv2d(
            x,
            self.num_conv_channels,
            [3, 3],
            activation_fn=None,
            weights_initializer=tf.initializers.he_normal(),
            biases_initializer=None
        )
        x = tf.compat.v2.keras.layers.BatchNormalization(axis=-1)
        x = tf.nn.relu(x)

        x = tf.compat.v1.layers.conv2d(
            x,
            self.num_conv_channels,
            [3, 3],
            activation_fn=None,
            weights_initializer=tf.initializers.he_normal(),
            biases_initializer=None
        )
        x = tf.compat.v2.keras.layers.BatchNormalization(axis=-1)
        x = tf.add(x, x_in)
        x = tf.nn.relu(x)
        return x

    def resnet_stack(self, x):
        for i in range(self.num_resnet_blocks):
            scope_name = "resnet_block_{}".format(i)
            with tf.compat.v1.variable_scope(scope_name):
                x = self.resnet_block(x)
        return x

    def net(self, x1, x2):
        x1 = self.resnet_stack(x1)

        x1 = tf.contrib.layers.flatten(x1)
        x2 = tf.squeeze(x2)
        x3 = tf.concat([x1, x2], 0)

        x = tf.layers.dense(
            x3,
            self.num_fc_channels,
            kernel_initializer=tf.initializers.he_normal(),
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=None,
            bias_regularizer=None
        )
        x = tf.layers.dropout(x, rate=0.1)
        x = tf.nn.relu(x)

        x = tf.layers.dense(
            x3,
            self.num_fc_channels,
            kernel_initializer=tf.initializers.he_normal(),
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=None,
            bias_regularizer=None
        )
        x = tf.layers.dropout(x, rate=0.1)

        return x

    def build(self):
        x1 = tf.placeholder(tf.float32, shape=(None, 13, 13, 9))
        x2 = tf.placeholder(tf.float32, shape=(None, 22))
        y = tf.placeholder(tf.float32, shape=(None,))

        logits = self.net(x1, x2)
        loss = tf.losses.sigmoid_cross_entropy(y, logits)
        train_op = self.optimizer.minimize(cross_entropy)

        init_op = tf.initialize_all_variables()
