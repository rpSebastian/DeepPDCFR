import math
import random
from typing import Optional

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import stats
from xhlib.logger import Logger
from xhlib.utils import init_object, set_seed

from xdcfr.game import read_game_config
from xdcfr.utils import (
    SpielState,
    evalute_explotability,
    play_n_games_against_random,
    play_n_poker_games_against_random,
)


class OSDeepCumuCFR:
    def __init__(
        self,
        game_name,
        num_episodes=400,
        advantage_buffer_size=100_0000,
        ave_policy_buffer_size=100_0000,
        learning_rate=1e-4,
        num_traversals=20,
        advantage_network_train_steps=1,
        ave_policy_network_train_steps=1,
        advantage_batch_size=-1,
        ave_policy_batch_size=-1,
        num_layers=2,
        num_hiddens=128,
        evaluation_frequency=10,
        reinitialize_advantage_networks=True,
        use_regret_matching_argmax=True,
        epsilon=0.6,
        logger=None,
        gamma=0,
        play_against_random=False,
        num_random_games=20000,
        device="cpu",
        seed=0,
    ):
        self.game_name = game_name
        self.play_against_random = play_against_random
        self.logger = logger or Logger(writer_strings=[])
        self.game = self.load_game()
        self.num_players = self.game.num_players()
        self.infostate_size = self.game.information_state_tensor_size()
        self.action_size = self.game.num_distinct_actions()
        self.advantage_buffer_size = advantage_buffer_size
        self.ave_policy_buffer_size = ave_policy_buffer_size
        self.num_episodes = num_episodes
        self.num_traversals = num_traversals
        self.num_iterations = self.num_episodes // (self.num_traversals * 2)
        self.advantage_network_train_steps = advantage_network_train_steps
        self.ave_policy_network_train_steps = ave_policy_network_train_steps
        self.ave_policy_batch_size = ave_policy_batch_size
        self.advantage_batch_size = advantage_batch_size
        self.reinitialize_advantage_networks = reinitialize_advantage_networks
        self.learning_rate = learning_rate
        self.device = device
        self.gamma = gamma
        self.evaluation_frequency = evaluation_frequency
        self.use_regret_matching_argmax = use_regret_matching_argmax
        self.max_utility = max(self.game.max_utility(), abs(self.game.min_utility()))
        self.network_layers = [num_hiddens for _ in range(num_layers)]
        self.epsilon = epsilon
        self.num_random_games = num_random_games

        self.root_node = self.game.new_initial_state()
        self.num_iteration = 0
        self.nodes_touched = 0
        self.episode = 0
        set_seed(seed)
        self.init_ave_policy_trainer()
        self.init_regret_trainers()

    def init_ave_policy_trainer(self):
        self.ave_policy_trainer = AvePolicyTrainer(
            self.infostate_size,
            self.action_size,
            self.network_layers,
            self.learning_rate,
            self.ave_policy_buffer_size,
            self.ave_policy_batch_size,
            self.ave_policy_network_train_steps,
            self.logger,
            self.device,
            self.gamma,
        )

    def init_regret_trainers(self):
        self.regret_trainers = [
            RegretTrainer(
                self.infostate_size,
                self.action_size,
                self.network_layers,
                self.learning_rate,
                self.advantage_buffer_size,
                self.advantage_batch_size,
                self.advantage_network_train_steps,
                self.logger,
                self.use_regret_matching_argmax,
                self.device,
            )
            for _ in range(self.num_players)
        ]

    def solve(
        self, is_search_hyper: bool = False, trial: Optional[optuna.Trial] = None
    ):
        self.is_search_hyper = is_search_hyper
        self.trial = trial

        self.evaluate()
        for _ in range(self.num_iterations):
            self.iteration()

        if self.is_search_hyper:
            self.train_average_policy()
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
                return -reward
            else:
                exp = evalute_explotability(
                    self.game, self.ave_policy_trainer.action_probabilities
                )
                return exp

    def iteration(self):
        self.num_iteration += 1
        for player in range(self.num_players):
            self.collect_training_data(player)
            self.train_regret(player)
        if (
            self.num_iteration % self.evaluation_frequency == 0
            or self.num_iteration < self.evaluation_frequency
            and self.num_iteration % (self.evaluation_frequency // 3) == 0
        ):
            self.train_average_policy()
            self.evaluate()

    def collect_training_data(self, player):
        self.regret_trainers[player].reset_buffer()
        for _ in range(self.num_traversals):
            self.episode += 1
            self.dfs(self.root_node, player)
        # self.logger.record("regret buffer", self.regret_trainers[player].buffer)
        # self.logger.dump(step=0)

    def train_regret(self, player):
        if self.reinitialize_advantage_networks:
            self.regret_trainers[player].reset()
        regret_loss = self.regret_trainers[player].train_model(self.num_iteration)
        self.logger.record("regret_loss_{}".format(player), regret_loss)

    def train_average_policy(self):
        self.ave_policy_trainer.reset()
        ave_policy_loss = self.ave_policy_trainer.train_model(self.num_iteration)
        self.logger.info("average policy loss: {}".format(ave_policy_loss))

    def evaluate(self):
        self.logger.record("nodes_touched", self.nodes_touched)
        self.logger.record("iteration", self.num_iteration)
        self.logger.record("episode", self.episode)
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
            self.logger.dump(step=self.episode)
            if self.is_search_hyper and self.trial is not None:
                self.trial.report(-reward, self.episode)
                # if self.trial.should_prune():
                #     raise optuna.TrialPruned()
        else:
            exp = evalute_explotability(
                self.game, self.ave_policy_trainer.action_probabilities
            )
            self.logger.record("exp", exp)
            self.logger.dump(step=self.episode)
            if self.is_search_hyper and self.trial is not None:
                self.trial.report(exp, self.episode)
                # if self.trial.should_prune():
                #     raise optuna.TrialPruned()

    def dfs(self, s, traverser, my_reach=1.0, opp_reach=1.0, sample_reach=1.0):
        self.nodes_touched += 1
        player = s.current_player()
        if player == -4:
            return s.returns()[traverser] / self.max_utility
        if player == -1:
            actions, probs = zip(*s.chance_outcomes())
            aid = np.random.choice(range(len(actions)), p=probs)
            action, prob = actions[aid], probs[aid]
            return self.dfs(
                s.child(action),
                traverser,
                my_reach,
                opp_reach * prob,
                sample_reach * prob,
            )
        legal_actions = s.legal_actions()
        policy = self.regret_trainers[player].get_policy(s, self.num_iteration)
        num_actions = s.num_distinct_actions()
        uniform_policy = np.array(s.legal_actions_mask()) / len(s.legal_actions())
        sample_policy = (
            uniform_policy * self.epsilon + policy * (1 - self.epsilon)
            if player == traverser
            else policy
        )
        sample_policy /= sample_policy.sum()
        action = np.random.choice(range(num_actions), p=sample_policy)
        sample_prob, prob = (
            sample_policy[action].item(),
            policy[action].item(),
        )

        if player == 1 - traverser:
            self.ave_policy_trainer.add_data(
                s.information_state_tensor(player),
                policy,
                s.legal_actions_mask(),
                self.num_iteration,
            )
            return self.dfs(
                s.child(action),
                traverser,
                my_reach,
                opp_reach * prob,
                sample_reach * sample_prob,
            )
        else:
            q_values = np.zeros_like(policy)
            q_values[action] = (
                self.dfs(
                    s.child(action),
                    traverser,
                    my_reach * prob,
                    opp_reach,
                    sample_reach * sample_prob,
                )
                / sample_prob
            )
            value = np.dot(q_values, policy)
            cf_regrets = np.zeros_like(policy)
            cf_regrets[legal_actions] = -value * opp_reach / sample_reach
            cf_regrets[action] += q_values[action] * opp_reach / sample_reach
            self.regret_trainers[player].add_data(
                s.information_state_tensor(player),
                cf_regrets,
                s.legal_actions_mask(),
                self.num_iteration,
            )
            return value

    def load_game(self):
        game_config = read_game_config(self.game_name)
        self.poker_game = game_config.poker
        if game_config.large_game:
            if not self.play_against_random:
                self.logger.warn("The game is too large, play against random instead.")
            self.play_against_random = True

        game = game_config.load_game()
        return game

    @classmethod
    def search_hyper(cls, game_name="KuhnPoker", device="cpu"):
        cls.game_name = game_name
        cls.device = device
        study_name = "{}_{}".format(game_name, cls.__name__)
        storage_name = "mysql+pymysql://xuhang:111111@10.10.100.51/sweep"
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            direction="minimize",
        )
        study.optimize(cls.objective, n_trials=100)

    @classmethod
    def objective(cls, trial):
        hyper_params = cls.get_hyper_params(trial)
        obj = init_object(cls, hyper_params, device=cls.device)
        exp = obj.solve(is_search_hyper=True, trial=trial)
        return exp

    @classmethod
    def get_hyper_params(cls, trial: optuna.Trial):
        raise NotImplemented


class VaniOSDeepCumuCFR(OSDeepCumuCFR):
    def dfs(self, s, traverser, my_reach=1.0, opp_reach=1.0, sample_reach=1.0):
        self.nodes_touched += 1
        player = s.current_player()
        if player == -4:
            return s.returns()[traverser] / self.max_utility
        if player == -1:
            actions, probs = zip(*s.chance_outcomes())
            aid = np.random.choice(range(len(actions)), p=probs)
            action, prob = actions[aid], probs[aid]
            return self.dfs(
                s.child(action),
                traverser,
                my_reach,
                opp_reach * prob,
                sample_reach * prob,
            )
        legal_actions = s.legal_actions()
        policy = self.regret_trainers[player].get_policy(s, self.num_iteration)
        num_actions = s.num_distinct_actions()
        uniform_policy = np.array(s.legal_actions_mask()) / len(s.legal_actions())
        sample_policy = (
            uniform_policy * self.epsilon + policy * (1 - self.epsilon)
            if player == traverser
            else policy
        )
        sample_policy /= sample_policy.sum()
        action = np.random.choice(range(num_actions), p=sample_policy)
        sample_prob, prob = (
            sample_policy[action].item(),
            policy[action].item(),
        )

        if player == 1 - traverser:
            return self.dfs(
                s.child(action),
                traverser,
                my_reach,
                opp_reach * prob,
                sample_reach * sample_prob,
            )
        else:
            q_values = np.zeros_like(policy)
            q_values[action] = (
                self.dfs(
                    s.child(action),
                    traverser,
                    my_reach * prob,
                    opp_reach,
                    sample_reach * sample_prob,
                )
                / sample_prob
            )
            value = np.dot(q_values, policy)
            cf_regrets = np.zeros_like(policy)
            cf_regrets[legal_actions] = -value * opp_reach / sample_reach
            cf_regrets[action] += q_values[action] * opp_reach / sample_reach
            self.regret_trainers[player].add_data(
                s.information_state_tensor(player),
                cf_regrets,
                s.legal_actions_mask(),
                self.num_iteration,
            )
            self.ave_policy_trainer.add_data(
                s.information_state_tensor(player),
                policy * my_reach / sample_reach,
                s.legal_actions_mask(),
                self.num_iteration,
            )
            return value


class OSDeepCumuAdv(OSDeepCumuCFR):
    def dfs(
        self,
        s,
        traverser,
        my_reach=1.0,
        opp_reach=1.0,
        opp_sample_reach=1.0,
    ):
        self.nodes_touched += 1
        player = s.current_player()
        if player == -4:
            return s.returns()[traverser] / self.max_utility
        if player == -1:
            actions, probs = zip(*s.chance_outcomes())
            aid = np.random.choice(range(len(actions)), p=probs)
            action, prob = actions[aid], probs[aid]
            return self.dfs(
                s.child(action),
                traverser,
                my_reach,
                opp_reach * prob,
                opp_sample_reach * prob,
            )
        legal_actions = s.legal_actions()
        policy = self.regret_trainers[player].get_policy(s, self.num_iteration)
        num_actions = s.num_distinct_actions()
        uniform_policy = np.array(s.legal_actions_mask()) / len(s.legal_actions())
        sample_policy = (
            uniform_policy * self.epsilon + policy * (1 - self.epsilon)
            if player == traverser
            else policy
        )
        sample_policy /= sample_policy.sum()
        action = np.random.choice(range(num_actions), p=sample_policy)
        sample_prob, prob = (
            sample_policy[action].item(),
            policy[action].item(),
        )

        if player == 1 - traverser:
            self.ave_policy_trainer.add_data(
                s.information_state_tensor(player),
                policy,
                s.legal_actions_mask(),
                self.num_iteration,
            )
            return self.dfs(
                s.child(action),
                traverser,
                my_reach,
                opp_reach * prob,
                opp_sample_reach * prob,
            )
        else:
            q_values = np.zeros_like(policy)
            q_values[action] = (
                self.dfs(
                    s.child(action),
                    traverser,
                    my_reach * prob,
                    opp_reach,
                    opp_sample_reach,
                )
                / sample_prob
            )
            value = np.dot(q_values, policy)
            cf_regrets = np.zeros_like(policy)
            cf_regrets[legal_actions] = -value * opp_reach / opp_sample_reach
            cf_regrets[action] += q_values[action] * opp_reach / opp_sample_reach
            self.regret_trainers[player].add_data(
                s.information_state_tensor(player),
                cf_regrets,
                s.legal_actions_mask(),
                self.num_iteration,
            )
            return value

    @classmethod
    def get_hyper_params(cls, trial: optuna.Trial):
        game_name = cls.game_name
        device = cls.device
        num_episodes = 1000000
        buffer_size = 1000000
        regret_learning_rate = trial.suggest_categorical(
            "regret_learning_rate", [1e-2, 1e-3, 3e-4]
        )
        policy_learning_rate = trial.suggest_categorical(
            "policy_learning_rate", [1e-2, 1e-3, 3e-4]
        )
        num_traversals = trial.suggest_categorical(
            "num_traversals", [500, 1000, 1500, 2000, 5000]
        )
        advantage_network_train_steps = trial.suggest_categorical(
            "advantage_network_train_steps", [500, 1000]
        )
        ave_policy_network_train_steps = trial.suggest_categorical(
            "ave_policy_network_train_steps", [500, 1000, 2000]
        )
        ave_policy_batch_size = 2 ** trial.suggest_int(
            "log2_ave_policy_batch_size", low=10, high=12
        )
        advantage_batch_size = -1
        reinitialize_advantage_networks = False
        num_hiddens = trial.suggest_categorical("num_hiddens", [64, 128, 256])
        num_layers = 3
        evaluation_frequency = 20
        use_regret_matching_argmax = True
        return locals()


class VaniOSDeepCumuAdv(OSDeepCumuCFR):
    def dfs(
        self,
        s,
        traverser,
        my_reach=1.0,
        opp_reach=1.0,
        sample_reach=1.0,
    ):
        self.nodes_touched += 1
        player = s.current_player()
        if player == -4:
            return s.returns()[traverser] / self.max_utility
        if player == -1:
            actions, probs = zip(*s.chance_outcomes())
            aid = np.random.choice(range(len(actions)), p=probs)
            action, prob = actions[aid], probs[aid]
            return self.dfs(
                s.child(action),
                traverser,
                my_reach,
                opp_reach * prob,
                sample_reach * prob,
            )
        legal_actions = s.legal_actions()
        policy = self.regret_trainers[player].get_policy(s, self.num_iteration)
        num_actions = s.num_distinct_actions()
        uniform_policy = np.array(s.legal_actions_mask()) / len(s.legal_actions())
        sample_policy = (
            uniform_policy * self.epsilon + policy * (1 - self.epsilon)
            if player == traverser
            else policy
        )
        sample_policy /= sample_policy.sum()
        action = np.random.choice(range(num_actions), p=sample_policy)
        sample_prob, prob = (
            sample_policy[action].item(),
            policy[action].item(),
        )

        if player == 1 - traverser:
            return self.dfs(
                s.child(action),
                traverser,
                my_reach,
                opp_reach * prob,
                sample_reach * sample_prob,
            )
        else:
            q_values = np.zeros_like(policy)
            q_values[action] = (
                self.dfs(
                    s.child(action),
                    traverser,
                    my_reach * prob,
                    opp_reach,
                    sample_reach * sample_prob,
                )
                / sample_prob
            )
            value = np.dot(q_values, policy)
            cf_regrets = np.zeros_like(policy)
            cf_regrets[legal_actions] = -value
            cf_regrets[action] += q_values[action]
            self.regret_trainers[player].add_data(
                s.information_state_tensor(player),
                cf_regrets,
                s.legal_actions_mask(),
                self.num_iteration,
            )
            self.ave_policy_trainer.add_data(
                s.information_state_tensor(player),
                policy * my_reach / sample_reach,
                s.legal_actions_mask(),
                self.num_iteration,
            )
            return value


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

    def reset_buffer(self):
        self.buffer.reset()

    def add_data(self, infostate, q_value, q_value_mask, iteration):
        self.buffer.add(infostate, q_value, q_value_mask, iteration)


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
            device,
        )
        self.use_regret_matching_argmax = use_regret_matching_argmax
        self.target_model = MLP(
            self.input_size, self.network_layers, self.output_size
        ).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

    def forward(self, model, x, mask):
        with torch.device(self.device):
            x = torch.as_tensor(x, dtype=torch.float32)
            mask = torch.as_tensor(mask, dtype=torch.float32)
            with torch.no_grad():
                legal_regrets = self.predict(model, x, mask)
                return legal_regrets.cpu().numpy()

    def predict(self, model, x, mask):
        legal_regrets = model(x) * mask
        return legal_regrets

    def train_model(self, T):
        for train_step in range(self.train_steps):
            samples = self.buffer.sample(self.batch_size)
            loss = self.compute_loss(samples, T)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if train_step % 100 == 0:
                self.logger.info(
                    "[{}/{}] regret loss: {}".format(
                        train_step, self.train_steps, loss.item()
                    )
                )
        self.target_model.load_state_dict(self.model.state_dict())
        return loss.item()

    def compute_loss(self, samples, T):
        infostates, cf_regrets, legal_actions_mask, iterations = samples
        with torch.no_grad():
            target_outputs = self.predict(
                self.target_model, infostates, legal_actions_mask
            )
        regrets = target_outputs + cf_regrets
        outputs = self.predict(self.model, infostates, legal_actions_mask)
        loss = self.loss_fn(outputs, regrets)
        return loss

    def get_policy(self, s: SpielState, T: int) -> np.ndarray:
        regrets = self.get_regrets(s)
        legal_actions = s.legal_actions()
        return self.regret_matching(regrets, legal_actions)

    def get_regrets(self, s: SpielState) -> np.ndarray:
        regrets = self.forward(
            self.model, s.information_state_tensor(), s.legal_actions_mask()
        )
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
        device: str = "cpu",
        gamma: int = 0,
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
            device=device,
        )
        self.gamma = gamma

    def forward(self, x, mask):
        with torch.device(self.device):
            x = torch.as_tensor(x, dtype=torch.float32)
            mask = torch.as_tensor(mask, dtype=torch.float32)
            with torch.no_grad():
                policy = self.predict(x, mask)
                return policy.cpu().numpy()

    def predict(self, x, mask):
        logits = self.model(x)
        legal_logits = torch.where(mask == 1, logits, -10e20)
        policy = self.softmax_fn(legal_logits)
        return policy

    def train_model(self, T):
        for train_step in range(self.train_steps):
            samples = self.buffer.sample(self.batch_size)
            loss = self.compute_loss(samples, T)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if train_step % 100 == 0:
                self.logger.info(
                    "[{}/{}] policy loss: {}".format(
                        train_step, self.train_steps, loss.item()
                    )
                )
        return loss.item()

    def compute_loss(self, samples, T):
        infostates, policy, legal_actions_mask, iterations = samples
        iterations = torch.pow(iterations / T * 2, self.gamma / 2)
        outputs = self.predict(infostates, legal_actions_mask)
        loss = self.loss_fn(outputs * iterations, policy * iterations)
        return loss

    def action_probabilities(self, s, probs_as_dict=True):
        policy = self.forward(s.information_state_tensor(), s.legal_actions_mask())
        if probs_as_dict:
            probs_dict = {action: policy[action] for action in s.legal_actions()}
            return probs_dict
        else:
            return policy


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
        self.q_value_mask_buf = np.zeros(
            [self.buffer_size, self.action_size], dtype=float
        )
        self.iteration_buf = np.zeros([self.buffer_size, 1], dtype=float)
        self.cur_id = 0

    def add(self, infostate, q_value, q_value_mask, iteration):
        if self.cur_id < self.buffer_size:
            self.add_data(self.cur_id, infostate, q_value, q_value_mask, iteration)
        else:
            idx = np.random.randint(low=0, high=self.cur_id + 1)
            if idx < self.buffer_size:
                self.add_data(idx, infostate, q_value, q_value_mask, iteration)
        self.cur_id += 1

    def add_data(self, idx, infostate, q_value, q_value_mask, iteration):
        self.infostate_buf[idx] = infostate
        self.q_value_buf[idx] = q_value
        self.q_value_mask_buf[idx] = q_value_mask
        self.iteration_buf[idx] = iteration

    def sample(self, num_samples=-1):
        self.data_length = min(self.cur_id, self.buffer_size)
        if num_samples > self.data_length:
            num_samples = -1
        if num_samples == -1:
            idxs = list(range(self.data_length))
        else:
            idxs = random.sample(range(self.data_length), num_samples)
        data = (
            self.infostate_buf[idxs],
            self.q_value_buf[idxs],
            self.q_value_mask_buf[idxs],
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


class ZeroInitLinear(nn.Module):
    def __init__(self, in_features, out_features, activation=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.weight = nn.Parameter(torch.zeros([self.out_features, self.in_features]))
        self.bias = nn.Parameter(torch.zeros([self.out_features]))

    def forward(self, input):
        y = F.linear(input, self.weight, self.bias)
        if self.activation:
            y = F.relu(y)
        return y


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        input_sizes = [input_size, *hidden_size[:-1]]
        output_sizes = [*hidden_size]
        activations = [True] * (len(hidden_size))
        layer_list = [
            SonnetLinear(in_size, out_size, activation)
            for in_size, out_size, activation in zip(
                input_sizes, output_sizes, activations
            )
        ]
        layer_list.append(ZeroInitLinear(hidden_size[-1], output_size, False))
        self.layers = nn.Sequential(*layer_list)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x):
        return self.layers(x)
