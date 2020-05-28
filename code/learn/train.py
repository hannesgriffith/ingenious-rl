import os
import time
import json
import argparse
from itertools import product

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils.io import load_json, make_dir_if_not_exists
from game.player import get_player
from game.gameplay import get_gameplay

from learn.network import get_network
from learn.replay_buffer import ReplayBuffer
from learn.exploration import get_exploration_policy
from learn.representation import get_representation
from learn.visualisation import generate_debug_visualisation

def get_training_session(args, config):
    # if config["training_type"] == "supervised":
    #     return SupervisedTrainingSession(args, config)
    if config["training_type"] == "self_play":
        return SelfPlayTrainingSession(args, config)
    else:
        raise ValueError("Incorrect training type name.")

class Params(object): 
    def __init__(self, d):
        self.__dict__ = d

    def save(self, save_path):
        output_path = os.path.join(save_path, "config.json")
        with open(output_path, 'w') as f:
            json.dump(self.__dict__, f)

class SelfPlayTrainingSession:
    def __init__(self, args, config):
        self.config = config
        self.p = Params(self.config)
        self.game = get_gameplay(self.config)
        self.repr = get_representation(self.config)
        self.replay_buffer = ReplayBuffer(self.config)
        self.exploration = get_exploration_policy(self.config)

        self.logs_dir = "logs/self_play_{}_{}".format(self.p.network_type, time.strftime("%Y-%m-%d_%H-%M"))
        self.logs_base_str = os.path.join(self.logs_dir, "ckpt-{}.pth")

        make_dir_if_not_exists(self.logs_dir)
        make_dir_if_not_exists(os.path.join(self.logs_dir, "tensorboard"))
        self.p.save(self.logs_dir)

        self.best_model_step = 0
        self.best_self_ckpt_path = os.path.join(self.logs_dir, "best_self.pth")
        self.best_rule_ckpt_path = os.path.join(self.logs_dir, "best_rule.pth")
        self.latest_ckpt_path = os.path.join(self.logs_dir, "latest.pth")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = get_network(self.config).to(self.device)

        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=self.p.initial_learning_rate,
            weight_decay=self.p.weight_decay
            )
        # self.criterion = nn.BCELoss(reduction='mean')
        self.criterion = nn.MSELoss(reduction='mean')

        self.strategy_types = ["random", "max", "increase_min", "reduce_deficit", "mixed_4"]
        self.writer = SummaryWriter(os.path.join(self.logs_dir, "tensorboard"))

    def log_network_weights_hists(self, step):
        for name, params in self.net.named_parameters():
            self.writer.add_histogram(f"weights/{name}", params, global_step=step)

    def initialise_rule_based_players(self):
        self.players = {}
        for strat in self.strategy_types:
            self.players[strat] = get_player("computer", None, strat)

        self.players_rand = {1: [], 2: []}
        for p, strat in product([1, 2], self.strategy_types):
            self.players_rand[p].append(get_player("computer", None, strat))

    def fill_replay_buffer(self, p1, p2):
        print("Filling replay buffer")
        while self.replay_buffer.is_not_full():
            print(f"Filled {len(self.replay_buffer)} / {self.replay_buffer.buffer_size}")
            _, new_reprs = self.game.generate_episode(p1, p2)
            self.replay_buffer.add(new_reprs)

    def apply_learning_update(self):
        inputs, labels, vis_inputs, idxs = self.replay_buffer.sample_training_minibatch()

        grid_input_device = torch.tensor(inputs[0], dtype=torch.float32, device=self.device)
        vector_input_device = torch.tensor(inputs[1], dtype=torch.float32, device=self.device)

        labels[labels == 0] = -1
        labels_device = torch.tensor(labels, dtype=torch.float32, device=self.device)

        self.optimizer.zero_grad()
        predictions = self.net(grid_input_device, vector_input_device)
        loss = self.criterion(torch.squeeze(predictions), torch.squeeze(labels_device))
        loss.backward()
        self.optimizer.step()

        labels_np = np.squeeze(labels).astype(np.float32)
        loss_np = loss.detach().cpu().numpy().astype(np.float32)
        predictions_np = torch.squeeze(predictions).detach().cpu().numpy().astype(np.float32)

        diffs = np.abs(labels_np - predictions_np)
        self.replay_buffer.update_probs(idxs, diffs)
        mean_abs_error = np.mean(diffs) 
        return loss_np, mean_abs_error, (vis_inputs, labels_np, predictions_np)

    def apply_n_learning_updates(self, n):
        self.net = self.net.train()

        loss_sum = 0.
        abs_error_sum = 0.
        for i in range(int(n)):
            loss, abs_error, vis_inputs = self.apply_learning_update()
            loss_sum += loss
            abs_error_sum += abs_error

        avg_loss = loss_sum / float(n)
        mean_abs_error = abs_error_sum / float(n)
        return avg_loss, mean_abs_error, vis_inputs

    def add_n_games_to_replay_buffer(self, p1, p2, n):
        for i in range(int(n)):
            _, new_reprs = self.game.generate_episode(p1, p2)
            self.replay_buffer.add(new_reprs)

    def play_n_test_games(self, p1, p2, n, learn=True):
        num_wins = 0
        avg_loss_sum = 0
        abs_error_sum = 0.

        episode_fn = self.game.generate_episode if learn else self.game.play_test_game

        for i in range(int(n)):
            winner, new_reprs = episode_fn(p1, p2)
            if learn:
                self.replay_buffer.add(new_reprs)
                avg_loss, abs_error, _ = self.apply_n_learning_updates(self.p.updates_per_episode)
                avg_loss_sum += avg_loss
                abs_error_sum += abs_error

            if winner == 1:
                num_wins += 1

        p1_win_rate = num_wins / float(n)
        avg_avg_loss = avg_loss_sum / float(n)
        mean_abs_error = abs_error_sum / float(n)
        return p1_win_rate, avg_avg_loss, mean_abs_error

    def add_graph_to_logs(self):
        inputs, _, _, _ = self.replay_buffer.sample_training_minibatch()
        grid_input_device = torch.tensor(inputs[0], dtype=torch.float32, device=self.device)
        vector_input_device = torch.tensor(inputs[1], dtype=torch.float32, device=self.device)
        self.writer.add_graph(self.net, (grid_input_device, vector_input_device))

    def train(self):
        self.initialise_rule_based_players()

        self.training_p1 = get_player("computer", None, "rl")
        self.training_p2 = get_player("computer", None, "rl")
        self.training_p1.strategy.set_model(get_network(self.config).to(self.device))
        self.training_p2.strategy.set_model(get_network(self.config).to(self.device))

        self.test_player = get_player("computer", None, "rl")
        self.test_player.strategy.set_model(self.net)
        self.test_player.strategy.set_explore(False)

        if self.p.start_ckpt_path is not None:
            self.net.load_state_dict(torch.load(self.p.start_ckpt_path))
            self.latest_ckpt_path = self.p.start_ckpt_path
        else:
            self.latest_ckpt_path = self.logs_base_str.format("start")
            torch.save(self.net.state_dict(), self.latest_ckpt_path)

        self.training_p1.strategy.model.load_state_dict(torch.load(self.latest_ckpt_path))
        self.training_p2.strategy.model.load_state_dict(torch.load(self.latest_ckpt_path))

        if self.p.start_ckpt_path is not None:
            p1 = self.training_p1
            p2 = self.training_p2
        else:
            p1 = get_player("computer", None, "random")
            p2 = get_player("computer", None, "random")

        self.fill_replay_buffer(p1, p2)
        self.add_n_games_to_replay_buffer(self.training_p1, self.training_p2, 2)

        running_loss, running_error = 0.0, 0.0
        best_win_rate_rule = 0.0
        self.add_graph_to_logs()

        self.training_p1.strategy.set_explore(True)
        self.training_p2.strategy.set_explore(True)

        print("Start training")
        for i in range(int(self.p.total_training_steps + 1)):
            print("Step {} / {}".format(i, self.p.total_training_steps))

            self.add_n_games_to_replay_buffer(self.training_p1, self.training_p2, self.p.episodes_per_step)
            avg_loss, abs_error, vis_inputs = self.apply_n_learning_updates(self.p.episodes_per_step * self.p.updates_per_episode)
            running_loss += avg_loss
            running_error += abs_error

            if i > 0 and i % int(self.p.log_every_n_steps) == 0:
                avg_running_loss = running_loss / float(self.p.log_every_n_steps)
                mean_abs_error = running_error / float(self.p.log_every_n_steps)
                running_loss, running_error = 0.0, 0.0

                self.writer.add_scalar('metrics/train_loss', avg_running_loss, i)
                self.writer.add_scalar('metrics/train_error', mean_abs_error, i)
                self.writer.add_scalar('metrics/base_lr', self.lr_tracker, i)
                self.log_network_weights_hists(i)

            if i > 0 and i % int(self.p.vis_every_n_steps) == 0:
                vis_figs = generate_debug_visualisation(vis_inputs)
                self.writer.add_figure('examples', vis_figs, global_step=i)

            if i % int(self.p.test_every_n_steps) == 0:
                self.latest_ckpt_path = self.logs_base_str.format(i)
                torch.save(self.net.state_dict(), self.latest_ckpt_path)
                self.training_p1.strategy.set_explore(False)

                print(f"Playing {self.p.n_test_games} test games against self")
                self_win_rate, avg_loss, abs_error = self.play_n_test_games(self.test_player, self.training_p1, self.p.n_test_games, learn=False)
                self.writer.add_scalar('win_rates/rl', self_win_rate, i)
                print("Win rate: {:.2f}".format(self_win_rate))

                if self_win_rate >= 0.6:
                    print("Best self model improved!")
                    self.training_p1.strategy.model.load_state_dict(torch.load(self.latest_ckpt_path))
                    self.training_p2.strategy.model.load_state_dict(torch.load(self.latest_ckpt_path))
                    torch.save(self.training_p1.strategy.model.state_dict(), self.best_self_ckpt_path)
                    self.best_model_step = i

                self.training_p1.strategy.set_explore(True)

                win_rate_rule = 0.0
                for strat in self.strategy_types:
                    print(f"Playing {self.p.n_other_games} test games against {strat}")
                    win_rate, avg_loss, abs_error = self.play_n_test_games(self.test_player, self.players[strat], self.p.n_other_games, learn=False)
                    self.writer.add_scalar(f'win_rates/{strat}', win_rate, i)
                    print("Win rate: {:.2f}".format(win_rate))
                    win_rate_rule += win_rate

                if win_rate_rule > best_win_rate_rule:
                    best_win_rate_rule = win_rate_rule
                    print("Best rule model improved!")
                    torch.save(self.net.state_dict(), self.best_rule_ckpt_path)

        print(f"Final best model was at step {self.best_model_step}")
        self.writer.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_name', default=None)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config = load_json(args.config_name)
    get_training_session(args, config).train()

if __name__ == "__main__":
    main()