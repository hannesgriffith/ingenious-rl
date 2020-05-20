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
        self.number_conv_filters = 64
        self.num_hidden_units = 128
        self.dropout_rate = 0.1

        self.batch_size = PARAMS["batch_size"]
        self.lr = PARAMS["initial_learning_rate"]

        # self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #         self.lr, decay_steps=25000, decay_rate=0.1, staircase=False)
        self.optimiser = tf.keras.optimizers.Adam(lr=self.lr)

        self.loss_function = tf.keras.losses.BinaryCrossentropy()
        self.metrics = [tf.keras.metrics.BinaryCrossentropy(
                                            name='train_binary_crossentropy')]

        self.build_mini_resnet_model()

    def compile_model(self):
        self.model.compile(optimizer=self.optimiser,
                           loss=self.loss_function,
                           metrics=self.metrics)

    def build_mini_resnet_model(self):
        x1_in = tf.keras.Input(shape=(13, 13, 9), name='grid_input')
        x2_in = tf.keras.Input(shape=(416,), name='vector_input')

        x1 = tf.keras.layers.Conv2D(self.number_conv_filters, 3, padding='same',
                use_bias=False, kernel_initializer='he_uniform')(x1_in)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        x1 = tf.keras.activations.relu(x1)

        x1 = self.resnet_stack(x1)
        x1 = tf.compat.v2.keras.layers.GlobalMaxPooling2D()(x1)

        x3 = tf.keras.layers.Concatenate(axis=-1)([x1, x2_in])
        x3 = tf.keras.layers.Dropout(self.dropout_rate)(x3)
        x3 = tf.keras.layers.Dense(self.num_hidden_units, use_bias=True,
                       kernel_initializer='he_uniform',
                       kernel_regularizer=tf.keras.regularizers.l2(l=1E-4))(x3)
        x3 = tf.keras.layers.Dropout(self.dropout_rate)(x3)
        x3 = tf.keras.activations.relu(x3)
        x3 = tf.keras.layers.Dense(1, kernel_initializer='he_uniform')(x3)
        x3_out = tf.keras.activations.sigmoid(x3)

        self.model = tf.keras.Model(inputs=[x1_in, x2_in], outputs=x3_out)

    def resnet_stack(self, x):
        x = self.residual_block(x, self.number_conv_filters)
        x = self.residual_block(x, self.number_conv_filters)
        x = self.residual_block(x, self.number_conv_filters * 2,
                                downsample=True)
        x = self.residual_block(x, self.number_conv_filters * 2)
        return x

    def residual_block(self, x, num_channels, downsample=False):
        if downsample:
            x_in = tf.keras.layers.AveragePooling2D(pool_size=(2, 2),
                                                    strides=(2, 2),
                                                    padding='valid')(x)
            x_in = tf.keras.layers.Conv2D(num_channels, 1, padding='same',
                    kernel_initializer='he_uniform')(x_in)
            x = tf.keras.layers.Conv2D(num_channels, 3, padding='valid',
                    strides=(2, 2), use_bias=False,
                    kernel_initializer='he_uniform')(x)
        else:
            x_in = x
            x = tf.keras.layers.Conv2D(num_channels, 3, padding='same',
                    strides=(1, 1), use_bias=False,
                    kernel_initializer='he_uniform')(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.Conv2D(num_channels, 3, padding='same',
                use_bias=False, kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([x, x_in])
        x = tf.keras.activations.relu(x)
        return x

    def training_step(self, x_train, y_train):
        x1, x2 = x_train
        input_dict = {'grid_input': x1, 'vector_input': x2}
        train_loss = self.model.train_on_batch(input_dict, y_train)
        preds = self.predict(x_train)
        diffs = np.abs(np.squeeze(preds) - np.squeeze(y_train))
        return train_loss, diffs

    def __call__(self, x_np):
        return self.predict(x_np)

    def predict(self, x):
        x1, x2 = x
        input_dict = {'grid_input': x1, 'vector_input': x2}
        predictions = self.model.predict_on_batch(input_dict)
        return predictions

    def save_weights(self, filepath, inlude_optimser=False):
        if inlude_optimser:
            self.model.save_weights(filepath)
        else:
            model_filepath = filepath.replace("ckpt", "h5")
            self.model.save(model_filepath, include_optimizer=False)
            dummy = get_network("computer", 1, None, None, None, "rl1")
            dummy.load_model(model_filepath)
            dummy.save_weights(filepath)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
