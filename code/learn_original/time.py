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
from learn import train_settings

class TrainingSession:
    def __init__(self, params, args):

        self.params = params
        self.game = get_gameplay(params)
        self.net = get_network(params)
        self.repr = get_representation(params)
        self.exploration = get_exploration_policy(params)
        self.logs_dir = args.logs_dir

        self.batch_size = int(params["batch_size"])
        self.log_every_n = int(params["log_every_n"])
        self.test_every_n = int(params["test_every_n"])
        self.n_test_games = int(params["n_test_games"])
        self.n_other_games = int(params["n_other_games"])
        self.steps_per_episode = int(params["steps_per_episode"])
        self.max_training_steps = int(params["total_training_steps"])

        if self.logs_dir is None:
            self.logs_dir = "logs/{}".format(time.strftime("%Y-%m-%d_%H-%M"))
            if not os.path.exists(self.logs_dir):
                os.mkdir(self.logs_dir)

        self.replay_buffer = ReplayBuffer(params, work_dir=self.logs_dir)
        self.replay_buffer_init = ReplayBuffer({"replay_buffer_size": self.batch_size})

        if "restore_session" in self.params and self.params["restore_session"] is not None:
            self.restore_session_logs = self.params["restore_session"]
            self.start_ckpt = os.path.join(self.restore_session_logs,
                                           "recent_weights.ckpt")
            self.best_model = os.path.join(self.restore_session_logs,
                                           "best_weights.ckpt")
            self.start_replay_buffer = os.path.join(self.restore_session_logs,
                                           "buffer")
        else:
            if "start_ckpt" in self.params:
                self.start_ckpt = self.params["start_ckpt"]
            else:
                self.start_ckpt = None

            if "best_model" in self.params:
                self.best_model = self.params["best_model"]
            else:
                self.best_model = None

            if "start_replay_buffer" in self.params:
                self.start_replay_buffer = self.params["start_replay_buffer"]
            else:
                self.start_replay_buffer = None

    def apply_init_learning_update(self, model):
        print("Applying init learning update")
        batch, idxs = self.replay_buffer_init.sample(self.batch_size)
        x_np, y_np = prepare_example(batch, training=True)
        model.training_step(x_np, y_np)

    def apply_learning_update(self, model=None):
        print("Applying learning update")
        if model is None:
            model = self.model
        batch, idxs = self.replay_buffer.sample(self.batch_size)
        x_np, y_np = prepare_example(batch, training=True)
        loss, diffs = model.training_step(x_np, y_np)
        self.replay_buffer.update_probs(idxs, diffs)
        print("Loss: {}".format(loss[0]))
        return loss

    def train(self):
        print("Starting training...")
        writer = tf.compat.v1.summary.FileWriter(self.logs_dir,
                                        graph=tf.compat.v1.get_default_graph())
        print("Writing logs to: {}".format(self.logs_dir))

        # print("With device: {}".format(self.params["training_device"]))
        self.model = get_network(self.params)
        self.model.compile_model()

        print("Set up training loss metric")
        train_loss = []

        print("Creating strategy players")
        p_random = get_player("computer", 2, None, None, None, "random")
        p_max1 = get_player("computer", 1, None, None, None, "max")
        p_min1 = get_player("computer", 1, None, None, None, "increase_min")
        p_max2 = get_player("computer", 2, None, None, None, "max")
        p_min2 = get_player("computer", 2, None, None, None, "increase_min")

        print("Generating initial small random buffer")
        while self.replay_buffer_init.is_not_full():
            _, examples = self.game.generate_episode(p_max1, p_min2)
            self.replay_buffer_init.add(examples)

        print("Creating main players")
        p_training = get_player("computer", 1, None, None, None, "rl1")
        p_adversary = get_player("computer", 2, None, None, None, "rl1")

        print("Initialise training model and save starting checkpoint")
        self.apply_init_learning_update(self.model)

        if self.start_ckpt is not None:
            print("Loading model weights from:", self.start_ckpt)
            first_ckpt_path = self.start_ckpt
            self.model.load_weights(first_ckpt_path)
        else:
            first_ckpt_path = os.path.join(self.logs_dir, "first.ckpt")
            self.model.save_weights(first_ckpt_path)

        p_training.strategy.set_model(get_network(self.params))
        # self.apply_init_learning_update(p_training.strategy.model)
        p_adversary.strategy.set_model(get_network(self.params))
        # self.apply_init_learning_update(p_adversary.strategy.model)

        if self.best_model is not None:
            print("Loading best model weights from:", self.best_model)
            best_model_path = self.best_model
            p_training.strategy.model.load_weights(best_model_path)
            p_adversary.strategy.model.load_weights(best_model_path)
            print("Saving best weights to new logs directory")
            best_model_path = os.path.join(self.logs_dir,
                                             "best_weights.ckpt")
            p_training.strategy.model.save_weights(best_model_path)
        else:
            print("Load initial checkpoints")
            best_model_path = first_ckpt_path
            p_training.strategy.model.load_weights(first_ckpt_path)
            p_adversary.strategy.model.load_weights(first_ckpt_path)

        print("Set initial exploration params")
        eps, temp = self.exploration.get_params(1)
        print("eps: {}, temp: {}".format(eps, temp))
        p_training.strategy.set_explore(True)
        p_training.strategy.set_explore_params(eps, temp)
        p_adversary.strategy.set_explore(True)
        p_adversary.strategy.set_explore_params(eps, temp)

        if self.start_replay_buffer is not None:
            self.replay_buffer.load_buffer_from_file(self.start_replay_buffer)
        else:
            j = 0
            print("Generating random starting dataset")
            while self.replay_buffer.is_not_full():
                print("Initial buffer episode {}".format(j))
                if self.best_model is not None:
                    _, examples = self.game.generate_episode(p_training, p_adversary)
                    self.replay_buffer.add(examples)
                else:
                    _, examples = self.game.generate_episode(p_max1, p_max2)
                    self.replay_buffer.add(examples)
                    _, examples = self.game.generate_episode(p_min1, p_min2)
                    self.replay_buffer.add(examples)
                    _, examples = self.game.generate_episode(p_max1, p_min2)
                    self.replay_buffer.add(examples)
                print("Replay buffer size: {}".format(self.replay_buffer.get_current_size()))
                j += 1

        for i in range(1, self.max_training_steps + 1):
            start_time = time.time()
            print("Episode {} / {}, ({} training steps)".format(i,
                    self.max_training_steps, self.steps_per_episode * i))
            print("Generating episode")
            _, episode = self.game.generate_episode(p_training, p_adversary)

            print("Adding episode to replay buffer")
            self.replay_buffer.add(episode)

            end_time = time.time()
            elapsed = end_time - start_time
            print("Time to generate episode: {}s".format(round(elapsed, 2)))

            for _ in range(self.steps_per_episode):
                loss = self.apply_learning_update()
                train_loss.append(loss)

            if i % self.log_every_n == 0 or i == 1:
                avg_train_loss = np.mean(np.array(train_loss, dtype=np.float32))
                summary = tf.compat.v1.Summary(
                            value=[tf.compat.v1.Summary.Value(tag="train_loss",
                                   simple_value=avg_train_loss)])
                writer.add_summary(summary, i)
                print("Training loss: {}".format(avg_train_loss))
                train_loss = []

                latest_weights_ckpt_path = os.path.join(self.logs_dir,
                                            "latest_weights_autosave.ckpt".format(i))
                self.model.save_weights(latest_weights_ckpt_pat, inlude_optimser=True)

            if i % self.test_every_n == 0 or i == 1:
                current_weights_ckpt_path = os.path.join(self.logs_dir,
                                            "weights-{}.ckpt".format(i))
                recent_weights_ckpt_path = os.path.join(self.logs_dir,
                                            "recent_weights.ckpt".format(i))

                self.model.save_weights(current_weights_ckpt_path)
                self.model.save_weights(recent_weights_ckpt_path, inlude_optimser=True)

                p_training.strategy.model.load_weights(current_weights_ckpt_path)
                p_training.strategy.set_explore(False)
                p_adversary.strategy.set_explore(False)

                num_wins = 0
                print("Playing test games")
                for n in range(self.n_test_games):
                    print("Test game {} of {}".format(n + 1,
                                                    self.n_test_games))
                    winner, episode = self.game.generate_episode(p_training,
                                                      p_adversary)

                    self.replay_buffer.add(episode)
                    for _ in range(self.steps_per_episode):
                        loss = self.apply_learning_update()
                        train_loss.append(loss)

                    if winner == 1:
                        num_wins += 1

                win_rate = float(num_wins) / float(self.n_test_games)
                summary = tf.compat.v1.Summary(
                            value=[tf.compat.v1.Summary.Value(tag="test_wins_self",
                                   simple_value=win_rate)])
                writer.add_summary(summary, i)
                print("Model win rate: {}".format(win_rate))

                if win_rate >= 0.6:
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
                                                                 p_max2)

                    self.replay_buffer.add(episode)
                    for _ in range(self.steps_per_episode):
                        loss = self.apply_learning_update()
                        train_loss.append(loss)

                    if winner == 1:
                        num_max_wins += 1

                win_rate = float(num_max_wins) / float(self.n_other_games)
                summary = tf.compat.v1.Summary(
                            value=[tf.compat.v1.Summary.Value(tag="test_wins_max",
                                   simple_value=win_rate)])
                writer.add_summary(summary, i)
                print("Max win rate: {}".format(win_rate))

                num_min_wins = 0
                print("Playing min strat test games")
                for _ in range(self.n_other_games):
                    winner, episode = self.game.generate_episode(p_training,
                                                                 p_min2)

                    self.replay_buffer.add(episode)
                    for _ in range(self.steps_per_episode):
                        loss = self.apply_learning_update()
                        train_loss.append(loss)

                    if winner == 1:
                        num_min_wins += 1

                win_rate = float(num_min_wins) / float(self.n_other_games)
                summary = tf.compat.v1.Summary(
                            value=[tf.compat.v1.Summary.Value(tag="test_wins_min",
                                   simple_value=win_rate)])
                writer.add_summary(summary, i)
                print("Min win rate: {}".format(win_rate))

                num_rand_wins = 0
                play_n_rand_games = 10
                print("Playing rand strat test games")
                for _ in range(play_n_rand_games):
                    winner, episode = self.game.generate_episode(p_training,
                                                                 p_random)

                    self.replay_buffer.add(episode)
                    for _ in range(self.steps_per_episode):
                        loss = self.apply_learning_update()
                        train_loss.append(loss)

                    if winner == 1:
                        num_rand_wins += 1

                win_rate = float(num_rand_wins) / float(play_n_rand_games)
                summary = tf.compat.v1.Summary(
                            value=[tf.compat.v1.Summary.Value(tag="test_wins_rand",
                                   simple_value=win_rate)])
                writer.add_summary(summary, i)
                print("Random win rate: {}".format(win_rate))

                print("Setting new exploration params")
                eps, temp = self.exploration.get_params(i)
                print("eps: {}, temp: {}".format(eps, temp))
                p_training.strategy.set_explore(True)
                p_adversary.strategy.set_explore(True)
                p_training.strategy.set_explore_params(eps, temp)
                p_adversary.strategy.set_explore_params(eps, temp)

                avg_train_loss = np.mean(np.array(train_loss, dtype=np.float32))
                summary = tf.compat.v1.Summary(
                            value=[tf.compat.v1.Summary.Value(tag="train_loss",
                                   simple_value=avg_train_loss)])
                writer.add_summary(summary, i)
                print("Training loss: {}".format(avg_train_loss))
                train_loss = []

                # self.replay_buffer.save_buffer_to_file()

        print("Final best model is {}".format(best_model_path))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs_dir', default=None)
    args = parser.parse_args()
    return args

def main():
    TrainingSession(train_settings.PARAMS, parse_args()).train()

if __name__ == "__main__":
    main()
