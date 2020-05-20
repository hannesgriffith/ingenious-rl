import os
import time
import argparse

import numpy as np
import tensorflow as tf

from game.gameplay import get_gameplay
from game.player import get_player
from learn.network_tf1_2 import get_network, prepare_example
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
            self.logs_dir = "logs/{}".format(time.strftime("%Y-%m-%d_%H-%M"))

    def apply_learning_update(self):
        print("Applying learning update")
        batch = self.replay_buffer.sample(self.batch_size)
        x_np, y_np = prepare_example(batch, training=True)
        loss = self.model.training_step(x_np, y_np)
        print("Loss: {}".format(loss))
        return loss

    def train(self):
        print("Starting training...")
        train_loss = []
        self.model = get_network(self.params)

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

        main_graph = tf.Graph()
        with main_graph.as_default():
            saver = tf.train.Saver()
            self.model.build_graph()
            accuracy_ph = tf.placeholder(tf.float32, shape=())
            tf.summary.scalar('accuracy', accuracy_ph)
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(os.path.join(save_dir, 'train'), sess.graph)
            test_writer = tf.summary.FileWriter(os.path.join(save_dir, 'test'), sess.graph)
            tf.global_variables_initializer().run()
            sess.run(tf.global_variables_initializer())

            print("Creating main players")
            p_training = get_player("computer", 1, None, None, None, "rl1")
            p_adversary = get_player("computer", 2, None, None, None, "rl1")

            print("Initialise training model and save starting checkpoint")
            first_ckpt_path = os.path.join(self.logs_dir, "first.ckpt")
            self.model.save_weights(first_ckpt_path)
            best_model_path = first_ckpt_path

            print("Load initial checkpoints")
            p_training.strategy.set_model(get_network(self.params))
            p_training.strategy.model.load_weights(first_ckpt_path)
            p_adversary.strategy.set_model(get_network(self.params))
            p_adversary.strategy.model.load_weights(first_ckpt_path)

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
                train_loss.append(loss)

                if i % 10 == 0:
                    avg_train_loss = np.mean(np.array(train_loss, dtype=np.float32))
                    tf.summary.scalar('training loss', avg_train_loss)
                    print("Training loss: {}".format(avg_train_loss))
                    train_loss = []

                if i % self.test_every_n == 0:
                    current_model_ckpt_path = os.path.join(self.logs_dir,
                                                "model-{}.ckpt".format(i))
                    self.model.save_model(current_model_ckpt_path)

                    current_weights_ckpt_path = os.path.join(self.logs_dir,
                                                "weights-{}.ckpt".format(i))
                    self.model.save_weights(current_weights_ckpt_path)

                    p_training.strategy.model.load_weights(current_weights_ckpt_path)
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
                        train_loss.append(loss)

                        if winner == 1:
                            num_wins += 1

                    win_rate = float(num_wins) / float(self.n_test_games)
                    tf.summary.scalar('model win %', win_rate)
                    print("Model win rate: {}".format(win_rate))

                    if win_rate > 0.6:
                        print("Model has improved!")
                        print("Best model is at step {}".format(i))
                        best_model_path = current_weights_ckpt_path
                        self.model.save_weights(os.path.join(self.logs_dir,
                                                         "best_weights.ckpt"))
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
                        train_loss.append(loss)

                        if winner == 1:
                            num_max_wins += 1

                    win_rate = float(num_max_wins) / float(self.n_other_games)
                    tf.summary.scalar('max win %', win_rate)
                    print("Max win rate: {}".format(win_rate))

                    num_min_wins = 0
                    print("Playing min strat test games")
                    for _ in range(self.n_other_games):
                        winner, episode = self.game.generate_episode(p_training,
                                                                     p_min)

                        self.replay_buffer.add(episode)
                        loss = self.apply_learning_update()
                        train_loss.append(loss)

                        if winner == 1:
                            num_min_wins += 1

                    win_rate = float(num_min_wins) / float(self.n_other_games)
                    tf.summary.scalar('min win %', win_rate)
                    print("Min win rate: {}".format(win_rate))

                    num_rand_wins = 0
                    print("Playing rand strat test games")
                    for _ in range(self.n_other_games):
                        winner, episode = self.game.generate_episode(p_training,
                                                                     p_random_2)

                        self.replay_buffer.add(episode)
                        loss = self.apply_learning_update()
                        train_loss.append(loss)

                        if winner == 1:
                            num_rand_wins += 1

                    win_rate = float(num_rand_wins) / float(self.n_other_games)
                    tf.summary.scalar('random win %', win_rate)
                    print("Random win rate: {}".format(win_rate))

                    print("Setting new exploration params")
                    eps, temp = self.exploration.get_params(i)
                    print("eps: {}, temp: {}".format(eps, temp))
                    p_training.strategy.set_explore(True)
                    p_adversary.strategy.set_explore(True)
                    p_training.strategy.set_explore_params(eps, temp)
                    p_adversary.strategy.set_explore_params(eps, temp)

                    avg_train_loss = np.mean(np.array(train_loss, dtype=np.float32))
                    tf.summary.scalar('training loss', avg_train_loss)
                    print("Training loss: {}".format(avg_train_loss))
                    train_loss = []

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
