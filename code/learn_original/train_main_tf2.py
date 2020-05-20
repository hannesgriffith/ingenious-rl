import os
import time
import argparse

import numpy as np
import tensorflow as tf

from game.gameplay import get_gameplay
from game.player import get_player
from learn.network import get_network, prepare_example
from learn.exploration import get_exploration_policy
from learn.representation import get_representation
from learn.replay_buffer import ReplayBuffer
from utils.io import load_json
from learn import train_settings

class TrainingSession:
    def __init__(self, params, args):
        self.params = params
        self.game = get_gameplay(params)
        self.net = get_network(params)
        self.repr = get_representation(params)
        self.replay_buffer = ReplayBuffer(params)
        self.exploration = get_exploration_policy(params)
        self.logs_dir = args.logs_dir

        self.batch_size = int(params["batch_size"])
        self.test_every_n = int(params["test_every_n"])
        self.n_test_games = int(params["n_test_games"])
        self.n_other_games = int(params["n_other_games"])
        self.max_training_steps = int(params["total_training_steps"])

        if self.logs_dir is None:
            self.logs_dir = "logs/{}".format(time.strftime("%Y-%m-%d_%H:%M"))

    def apply_learning_update(self):
        print("Applying learning update")

        batch = self.replay_buffer.sample(self.batch_size)
        x_np, y_np = prepare_example(batch, training=True)

        with tf.GradientTape() as tape:
            scores = self.model(x_np)
            loss = self.loss_fn(y_np, scores)
            print("Loss: {}".format(loss))

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients,
                                           self.model.trainable_variables))

        return loss

    def train(self):
        print("Starting training...")

        file_writer = tf.summary.create_file_writer(self.logs_dir)
        file_writer.set_as_default()

        print("With device: {}".format(self.params["training_device"]))
        with tf.device(self.params["training_device"]): # '/device:GPU:0' or '/cpu:0'
            self.loss_fn = self.loss_function()
            self.optimizer = self.optimizer_init_fn()
            self.model = get_network(self.params)

            print("Set up training loss metric")
            train_loss = tf.keras.metrics.Mean(name='train_loss')

            print("Creating strategy players")
            p_random_1 = get_player("computer", 1, None, None, None, "random")
            p_random_2 = get_player("computer", 2, None, None, None, "random")
            p_max = get_player("computer", 2, None, None, None, "max")
            p_min = get_player("computer", 2, None, None, None, "increase_min")

            print("Generating random starting dataset")
            i = 0
            while self.replay_buffer.is_not_full():
                print("Initial buffer episode {}".format(i))
                _, examples = self.game.generate_episode(p_random_1, p_random_2)
                self.replay_buffer.add(examples)
                i += 1

            print("Creating main players")
            p_training = get_player("computer", 1, None, None, None, "rl1")
            p_adversary = get_player("computer", 2, None, None, None, "rl1")

            print("Initialise training model and save starting checkpoint")
            batch = self.replay_buffer.sample(self.batch_size)
            print("Preparing initialisation example")
            x_np, y_np = prepare_example(batch, training=True)
            _ = self.model(x_np) # First call to inialise weights
            first_ckpt_path = os.path.join(self.logs_dir, "first_ckpt")
            self.model.save_weights(first_ckpt_path, save_format='tf')
            best_model_path = first_ckpt_path

            print("Load initial checkpoints")
            p_training.strategy.set_model(get_network(self.params))
            p_training.strategy.model.load_weights(first_ckpt_path)
            _ = p_training.strategy.model(x_np)
            p_adversary.strategy.set_model(get_network(self.params))
            p_adversary.strategy.model.load_weights(first_ckpt_path)
            _ = p_adversary.strategy.model(x_np)

            print("Set initial exploration params")
            eps, temp = self.exploration.get_params(1)
            print("eps: {}, temp: {}".format(eps, temp))
            p_training.strategy.set_explore(True)
            p_training.strategy.set_explore_params(eps, temp)
            p_adversary.strategy.set_explore(True)
            p_adversary.strategy.set_explore_params(eps, temp)

            for i in range(self.max_training_steps):
                print("Step {} out of {}".format(i, self.max_training_steps))

                print("Generating episode")
                _, episode = self.game.generate_episode(p_training, p_adversary)

                print("Adding episode to replay buffer")
                self.replay_buffer.add(episode)

                loss = self.apply_learning_update()
                train_loss.update_state(loss)

                if i % 10 == 0:
                    tf.summary.scalar('training loss', data=train_loss.result(),
                                      step=i)
                    print("Training loss: {}".format(train_loss.result()))
                    train_loss.reset_states()

                if i % self.test_every_n == 0:
                    current_ckpt_path = os.path.join(self.logs_dir,
                                                     "ckpt-{}".format(i))
                    self.model.save_weights(current_ckpt_path,
                                            save_format='tf')

                    p_training.strategy.model.load_weights(current_ckpt_path)
                    p_training.strategy.set_explore(False)
                    p_adversary.strategy.set_explore(False)

                    num_wins = 0
                    print("Playing test games")
                    for n in range(self.n_test_games):
                        print("Test game {} of {}".format(n,
                                                        self.n_test_games))
                        winner, episode = self.game.generate_episode(p_training,
                                                          p_adversary)

                        self.replay_buffer.add(episode)
                        loss = self.apply_learning_update()
                        train_loss.update_state(loss)

                        if winner == 1:
                            num_wins += 1

                    win_rate = float(num_wins) / float(self.n_test_games)
                    tf.summary.scalar('model win %', data=win_rate, step=i)
                    print("Model win rate: {}".format(win_rate))

                    if win_rate > 0.55:
                        print("Model has improved!")
                        print("Best model is at step {}".format(i))
                        best_model_path = current_ckpt_path
                    else:
                        print("Model does not beat last ckpt")

                    p_training.strategy.model.load_weights(best_model_path)
                    p_adversary.strategy.model.load_weights(best_model_path)

                    num_max_wins = 0
                    print("Playing max strat test games")
                    for _ in range(self.n_other_games):
                        winner, episode = self.game.generate_episode(p_training,
                                                                     p_max)

                        self.replay_buffer.add(episode)
                        loss = self.apply_learning_update()
                        train_loss.update_state(loss)

                        if winner == 1:
                            num_max_wins += 1

                    win_rate = float(num_max_wins) / float(self.n_test_games)
                    tf.summary.scalar('max win %', data=win_rate, step=i)
                    print("Max win rate: {}".format(win_rate))

                    num_min_wins = 0
                    print("Playing min strat test games")
                    for _ in range(self.n_other_games):
                        winner, episode = self.game.generate_episode(p_training,
                                                                     p_min)

                        self.replay_buffer.add(episode)
                        loss = self.apply_learning_update()
                        train_loss.update_state(loss)

                        if winner == 1:
                            num_min_wins += 1

                    win_rate = float(num_min_wins) / float(self.n_test_games)
                    tf.summary.scalar('min win %', data=win_rate, step=i)
                    print("Min win rate: {}".format(win_rate))

                    num_rand_wins = 0
                    print("Playing rand strat test games")
                    for _ in range(self.n_other_games):
                        winner, episode = self.game.generate_episode(p_training,
                                                                     p_random_2)

                        self.replay_buffer.add(episode)
                        loss = self.apply_learning_update()
                        train_loss.update_state(loss)

                        if winner == 1:
                            num_rand_wins += 1

                    win_rate = float(num_rand_wins) / float(self.n_test_games)
                    tf.summary.scalar('random win %', data=win_rate, step=i)
                    print("Random win rate: {}".format(win_rate))

                    print("Setting new exploration params")
                    eps, temp = self.exploration.get_params(i)
                    print("eps: {}, temp: {}".format(eps, temp))
                    p_training.strategy.set_explore(True)
                    p_adversary.strategy.set_explore(True)
                    p_training.strategy.set_explore_params(eps, temp)
                    p_adversary.strategy.set_explore_params(eps, temp)

                    tf.summary.scalar('training loss', data=train_loss.result(),
                                      step=i)
                    print("Training loss: {}".format(train_loss.result()))
                    train_loss.reset_states()

        print("Final best model is {}".format(best_model))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs_dir', default=None)
    args = parser.parse_args()
    return args

def main():
    TrainingSession(train_settings.PARAMS, parse_args()).train()

if __name__ == "__main__":
    main()
