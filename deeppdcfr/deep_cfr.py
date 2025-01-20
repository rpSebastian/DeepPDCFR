import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import stats
from deeppdcfr.logger import Logger

from deeppdcfr.game import read_game_config
from deeppdcfr.utils import (
    SpielState,
    evalute_explotability,
    play_n_games_against_random,
    play_n_poker_games_against_random,
)


class DeepCFR:
    def __init__(
        self,
        game_name,
        advantage_buffer_size=100_0000,
        ave_policy_buffer_size=100_0000,
        learning_rate=1e-4,
        num_iterations=10,
        num_traversals=20,
        advantage_network_train_steps=1,
        ave_policy_network_train_steps=1,
        advantage_batch_size=-1,
        ave_policy_batch_size=-1,
        num_layers=2,
        num_hiddens=128,
        evaluation_frequency=10,
        reinitialize_advantage_networks=True,
        linear_weighted=True,
        use_regret_matching_argmax=True,
        play_against_random=False,
        num_random_games=20000,
        logger=None,
        device="cpu",
    ):
        self.game_name = game_name
        self.play_against_random = play_against_random
        self.game = self.load_game()
        self.num_players = self.game.num_players()
        self.infostate_size = self.game.information_state_tensor_size()
        self.action_size = self.game.num_distinct_actions()
        self.advantage_buffer_size = advantage_buffer_size
        self.policy_buffer_size = ave_policy_buffer_size
        self.num_iterations = num_iterations
        self.num_traversals = num_traversals
        self.advantage_network_train_steps = advantage_network_train_steps
        self.ave_policy_network_train_steps = ave_policy_network_train_steps
        self.ave_policy_batch_size = ave_policy_batch_size
        self.advantage_batch_size = advantage_batch_size
        self.reinitialize_advantage_networks = reinitialize_advantage_networks
        self.learning_rate = learning_rate
        self.logger = logger or Logger(writer_strings=[])
        self.device = device
        self.evaluation_frequency = evaluation_frequency
        self.linear_weighted = linear_weighted
        self.use_regret_matching_argmax = use_regret_matching_argmax
        self.num_random_games = num_random_games
        network_layers = [num_hiddens for _ in range(num_layers)]

        self.ave_policy_trainer = AvePolicyTrainer(
            self.infostate_size,
            self.action_size,
            network_layers,
            self.learning_rate,
            self.policy_buffer_size,
            self.ave_policy_batch_size,
            self.ave_policy_network_train_steps,
            self.logger,
            self.linear_weighted,
            self.device,
        )
        self.regret_trainers = [
            RegretTrainer(
                self.infostate_size,
                self.action_size,
                network_layers,
                self.learning_rate,
                self.advantage_buffer_size,
                self.advantage_batch_size,
                self.advantage_network_train_steps,
                self.logger,
                self.linear_weighted,
                self.use_regret_matching_argmax,
                self.device,
            )
            for _ in range(self.num_players)
        ]
        self.num_iteration = 0
        self.nodes_touched = 0


    def solve(self):
        root_state = self.game.new_initial_state()
        advantage_losses = {0: [], 1: []}
        for self.num_iteration in range(1, self.num_iterations + 1):
            for p in range(self.num_players):
                for _ in range(self.num_traversals):
                    self.dfs(root_state, p)
                if self.reinitialize_advantage_networks:
                    self.regret_trainers[p].reset()
                advantage_loss = self.regret_trainers[p].train_model()
                advantage_losses[p].append(advantage_loss)
                if advantage_loss is not None:
                    self.logger.record("advantage_loss_{}".format(p), advantage_loss)
            if (self.num_iteration % self.evaluation_frequency == 0
                or self.num_iteration < self.evaluation_frequency
            ):
                self.ave_policy_trainer.reset()
                ave_policy_loss = self.ave_policy_trainer.train_model()
                if self.play_against_random:
                    if self.poker_game:
                        reward = play_n_poker_games_against_random(
                            self.game,
                            self.ave_policy_trainer.action_probabilities,
                            self.num_random_games,
                        )
                    else:
                        reward = play_n_games_against_random(
                            self.game,
                            self.ave_policy_trainer.action_probabilities,
                            self.num_random_games,
                        )
                    self.logger.record("reward", reward)
                else:
                    exp = evalute_explotability(
                        self.game, self.ave_policy_trainer.action_probabilities
                    )
                    self.logger.record("exp", exp)
                self.logger.record("nodes_touched", self.nodes_touched)
                self.logger.record("iteration", self.num_iteration)
                if ave_policy_loss is not None:
                    self.logger.record("ave_policy_loss", ave_policy_loss)
                self.logger.dump(self.nodes_touched)
        return advantage_losses, ave_policy_loss


    def dfs(self, s, traverser):
        self.nodes_touched += 1
        player = s.current_player()
        if player == -4:
            return s.returns()[traverser]
        if player == -1:
            actions, probs = zip(*s.chance_outcomes())
            action = np.random.choice(actions, p=probs)
            return self.dfs(s.child(action), traverser)
        legal_actions = s.legal_actions()
        policy = self.regret_trainers[player].get_policy(s)
        if player == 1 - traverser:
            action = np.random.choice(range(self.action_size), p=policy)
            self.ave_policy_trainer.add_data(
                s.information_state_tensor(player), policy, self.num_iteration
            )
            return self.dfs(s.child(action), traverser)
        else:
            q_values = np.zeros_like(policy)
            for action in legal_actions:
                q_values[action] = self.dfs(s.child(action), traverser)
            value = np.dot(q_values, policy)
            cf_regrets = np.zeros_like(policy)
            cf_regrets[legal_actions] = (q_values - value)[legal_actions]
            self.regret_trainers[player].add_data(
                s.information_state_tensor(player), cf_regrets, self.num_iteration
            )
            return value

    def load_game(self):
        game_config = read_game_config(self.game_name)
        self.poker_game = game_config.poker
        if game_config.large_game:
            self.play_against_random = True
        game = game_config.load_game()
        return game


class Trainer:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        network_layers: list[int],
        learning_rate: float,
        buffer_size: int,
        batch_size: int,
        train_steps: int,
        logger: Logger,
        linear_weighted: bool,
        device: str = "cpu",
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.network_layers = network_layers
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.logger = logger
        self.linear_weighted = linear_weighted
        self.device = device

        self.model = MLP(self.input_size, self.network_layers, self.output_size).to(
            self.device
        )
        self.buffer = ReservoirBuffer(
            self.buffer_size, self.input_size, self.output_size, device=self.device
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.softmax_fn = nn.Softmax(dim=-1)

    def reset(self):
        self.model.reset_parameters()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def add_data(self, infostate, q_value, iteration):
        self.buffer.add(infostate, q_value, iteration)

    def forward(self, x):
        with torch.device(self.device):
            x = torch.as_tensor(x, dtype=torch.float32)
            with torch.no_grad():
                return self.model(x).cpu().numpy()

    def predict(self, x):
        return self.model(x)


class RegretTrainer(Trainer):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        network_layers: list[int],
        learning_rate: float,
        buffer_size: int,
        batch_size: int,
        train_steps: int,
        logger: Logger,
        linear_weighted: bool,
        use_regret_matching_argmax: bool,
        device: str = "cpu",
    ):
        super().__init__(
            input_size,
            output_size,
            network_layers,
            learning_rate,
            buffer_size,
            batch_size,
            train_steps,
            logger,
            linear_weighted,
            device,
        )
        self.use_regret_matching_argmax = use_regret_matching_argmax

    def train_model(self):
        if self.batch_size > 0 and len(self.buffer) < self.batch_size:
            return
        for _ in range(self.train_steps):
            samples = self.buffer.sample(self.batch_size)
            infostates, advantages, iterations = samples
            outputs = self.predict(infostates)
            if self.linear_weighted:
                iterations = torch.sqrt(iterations)
                loss = self.loss_fn(outputs * iterations, advantages * iterations)
            else:
                loss = self.loss_fn(outputs, advantages)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.item()

    def get_policy(self, s: SpielState) -> np.ndarray:
        regrets = self.get_regrets(s)
        legal_actions = s.legal_actions()
        return self.regret_matching(regrets, legal_actions)

    def get_regrets(self, s: SpielState) -> np.ndarray:
        regrets = self.forward(s.information_state_tensor())
        return regrets

    def regret_matching(self, regrets, legal_actions):
        legal_regrets = np.zeros_like(regrets)
        legal_regrets[legal_actions] = regrets[legal_actions]
        pos_sum = np.sum(np.maximum(legal_regrets, 0))
        if pos_sum > 0:
            return np.maximum(legal_regrets, 0) / pos_sum
        else:
            policy = np.zeros_like(regrets)
            if self.use_regret_matching_argmax:
                max_action_id = legal_actions[np.argmax(regrets[legal_actions])]
                policy[max_action_id] = 1
            else:
                policy[legal_actions] = 1 / len(legal_actions)
            return policy


class AvePolicyTrainer(Trainer):
    def train_model(self):
        if self.batch_size > 0 and len(self.buffer) < self.batch_size:
            return
        for _ in range(self.train_steps):
            samples = self.buffer.sample(self.batch_size)
            infostates, policy, iterations = samples
            logits = self.predict(infostates)
            outputs = self.softmax_fn(logits)
            if self.linear_weighted:
                iterations = torch.sqrt(iterations)
                loss = self.loss_fn(outputs * iterations, policy * iterations)
            else:
                loss = self.loss_fn(outputs, policy)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.item()

    def action_probabilities(self, s):
        player = s.current_player()
        infostate = torch.tensor(
            s.information_state_tensor(player), dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            logits = self.model(infostate)
            probs = self.softmax_fn(logits).cpu().numpy()
        policy = np.zeros(s.num_distinct_actions())
        legal_actions = s.legal_actions()
        policy[legal_actions] = probs[legal_actions]
        if sum(probs) != 0:
            policy /= sum(policy)
        else:
            policy[legal_actions] = 1 / len(legal_actions)
        prob_dict = {action: policy[action] for action in s.legal_actions()}
        return prob_dict


class ReservoirBuffer:
    def __init__(self, buffer_size, infostate_size, action_size, device="cpu"):
        self.buffer_size = buffer_size
        self.infostate_size = infostate_size
        self.action_size = action_size
        self.device = device
        self.reset()

    def reset(self):
        self.infostate_buf = np.zeros(
            [self.buffer_size, self.infostate_size], dtype=float
        )
        self.q_value_buf = np.zeros([self.buffer_size, self.action_size], dtype=float)
        self.iteration_buf = np.zeros([self.buffer_size, 1], dtype=float)
        self.cur_id = 0

    def add(self, infostate, q_value, iteration):
        if self.cur_id < self.buffer_size:
            self.add_data(self.cur_id, infostate, q_value, iteration)
        else:
            idx = np.random.randint(low=0, high=self.cur_id + 1)
            if idx < self.buffer_size:
                self.add_data(idx, infostate, q_value, iteration)
        self.cur_id += 1

    def add_data(self, idx, infostate, q_value, iteration):
        self.infostate_buf[idx] = infostate
        self.q_value_buf[idx] = q_value
        self.iteration_buf[idx] = iteration

    def sample(self, num_samples=-1):
        self.data_length = min(self.cur_id, self.buffer_size)
        if num_samples == -1:
            idxs = list(range(self.data_length))
        else:
            idxs = random.sample(range(self.data_length), num_samples)
        data = (
            self.infostate_buf[idxs],
            self.q_value_buf[idxs],
            self.iteration_buf[idxs],
        )
        data_tensor = tuple(map(self.numpy_to_tensor, data))
        return data_tensor

    def numpy_to_tensor(self, data):
        return torch.as_tensor(data, dtype=torch.float32, device=self.device)

    def __len__(self):
        return self.cur_id


class SonnetLinear(nn.Module):
    def __init__(self, in_features, out_features, activation=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stddev = 1 / math.sqrt(self.in_features)
        self.weight = nn.Parameter(
            torch.Tensor(
                stats.truncnorm.rvs(
                    -2,
                    2,
                    loc=0,
                    scale=stddev,
                    size=[self.out_features, self.in_features],
                ),
            )
        )
        self.bias = nn.Parameter(torch.zeros([self.out_features]))

    def forward(self, input):
        y = F.linear(input, self.weight, self.bias)
        if self.activation:
            y = F.relu(y)
        return y


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        input_sizes = [input_size, *hidden_size]
        output_sizes = [*hidden_size, output_size]
        activations = [True] * (len(hidden_size)) + [False]
        self.layers = nn.Sequential(
            *[
                SonnetLinear(in_size, out_size, activation)
                for in_size, out_size, activation in zip(
                    input_sizes, output_sizes, activations
                )
            ]
        )

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x):
        return self.layers(x)
