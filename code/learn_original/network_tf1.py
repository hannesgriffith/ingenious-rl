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

class Network1(tf.keras.Model):
    def __init__(self):
        super(Network1, self).__init__()
        print("Initialising network")
        self.kernel_initializer = tf.initializers.VarianceScaling(scale=2.0)
        self.bias_initializer = tf.keras.initializers.Zeros()

        self.resnet5_hidden_size = 32
        self.fc_hidden_size = 64

        self.flatten = tf.keras.layers.Flatten()
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.concat = tf.keras.layers.Concatenate()

        self.conv2d_01 = tf.keras.layers.Conv2D(self.resnet5_hidden_size, 1, 1,
                                    padding='same',
                                    kernel_initializer=self.kernel_initializer,
                                    use_bias=True,
                                    bias_initializer=self.bias_initializer)
        self.conv2d_11 = tf.keras.layers.Conv2D(self.resnet5_hidden_size, 3, 1,
                                    padding='same',
                                    kernel_initializer=self.kernel_initializer,
                                    use_bias=True,
                                    bias_initializer=self.bias_initializer)
        self.conv2d_12 = tf.keras.layers.Conv2D(self.resnet5_hidden_size, 3, 1,
                                    padding='same',
                                    kernel_initializer=self.kernel_initializer,
                                    use_bias=True,
                                    bias_initializer=self.bias_initializer)
        self.conv2d_21 = tf.keras.layers.Conv2D(self.resnet5_hidden_size, 3, 1,
                                    padding='same',
                                    kernel_initializer=self.kernel_initializer,
                                    use_bias=True,
                                    bias_initializer=self.bias_initializer)
        self.conv2d_22 = tf.keras.layers.Conv2D(self.resnet5_hidden_size, 3, 1,
                                    padding='same',
                                    kernel_initializer=self.kernel_initializer,
                                    use_bias=True,
                                    bias_initializer=self.bias_initializer)
        self.conv2d_31 = tf.keras.layers.Conv2D(self.resnet5_hidden_size, 3, 1,
                                    padding='same',
                                    kernel_initializer=self.kernel_initializer,
                                    use_bias=True,
                                    bias_initializer=self.bias_initializer)
        self.conv2d_32 = tf.keras.layers.Conv2D(self.resnet5_hidden_size, 3, 1,
                                    padding='same',
                                    kernel_initializer=self.kernel_initializer,
                                    use_bias=True,
                                    bias_initializer=self.bias_initializer)
        self.conv2d_41 = tf.keras.layers.Conv2D(self.resnet5_hidden_size, 3, 1,
                                    padding='same',
                                    kernel_initializer=self.kernel_initializer,
                                    use_bias=True,
                                    bias_initializer=self.bias_initializer)
        self.conv2d_42 = tf.keras.layers.Conv2D(self.resnet5_hidden_size, 3, 1,
                                    padding='same',
                                    kernel_initializer=self.kernel_initializer,
                                    use_bias=True,
                                    bias_initializer=self.bias_initializer)
        self.conv2d_51 = tf.keras.layers.Conv2D(self.resnet5_hidden_size, 3, 1,
                                    padding='same',
                                    kernel_initializer=self.kernel_initializer,
                                    use_bias=True,
                                    bias_initializer=self.bias_initializer)
        self.conv2d_52 = tf.keras.layers.Conv2D(self.resnet5_hidden_size, 3, 1,
                                    padding='same',
                                    kernel_initializer=self.kernel_initializer,
                                    use_bias=True,
                                    bias_initializer=self.bias_initializer)
        self.fc_layer_1 = tf.keras.layers.Dense(self.fc_hidden_size,
                                     kernel_initializer=self.kernel_initializer,
                                     use_bias=True,
                                     bias_initializer=self.bias_initializer)
        # self.fc_layer_2 = tf.keras.layers.Dense(self.fc_hidden_size,
        #                              kernel_initializer=self.kernel_initializer,
        #                              use_bias=True,
        #                              bias_initializer=self.bias_initializer)
        self.out_layer = tf.keras.layers.Dense(1,
                                     kernel_initializer=self.kernel_initializer,
                                     use_bias=True,
                                     bias_initializer=self.bias_initializer)

    def call(self, x):
        x1, x2 = x

        x_res = x1
        # x = self.conv2d_01(x1)
        # x = self.batch_norm(x)
        # x_res = self.relu(x)

        x = self.conv2d_11(x_res)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv2d_12(x)
        x = self.batch_norm(x)
        # x += x_res
        x_res = self.relu(x)

        x = self.conv2d_21(x_res)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv2d_22(x)
        x = self.batch_norm(x)
        x += x_res
        x_res = self.relu(x)

        # x = self.conv2d_31(x_res)
        # x = self.batch_norm(x)
        # x = self.relu(x)
        # x = self.conv2d_32(x)
        # x = self.batch_norm(x)
        # x += x_res
        # x_res = self.relu(x)

        # x = self.conv2d_41(x_res)
        # x = self.batch_norm(x)
        # x = self.relu(x)
        # x = self.conv2d_42(x)
        # x = self.batch_norm(x)
        # x += x_res
        # x_res = self.relu(x)

        x = self.conv2d_51(x_res)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv2d_52(x)
        x = self.batch_norm(x)
        x += x_res
        x = self.relu(x)

        x = self.flatten(x)
        x3 = self.concat([x, x2])

        x = self.fc_layer_1(x3)
        # x = self.batch_norm(x)
        x = self.relu(x)

        # x = self.fc_layer_2(x)
        # x = self.batch_norm(x)
        # x = self.relu(x)

        x = self.out_layer(x)
        output = tf.keras.activations.sigmoid(x)

        return output
