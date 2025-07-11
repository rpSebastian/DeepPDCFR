import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from scipy import stats
import math

from deeppdcfr.logger import Logger
from deeppdcfr.os_deep_cfr import OSDeepCFR
from deeppdcfr.deep_cfr import Trainer, RegretTrainer
from deeppdcfr.utils import (
    evalute_explotability,
    play_n_games_against_random,
    play_n_poker_games_against_random,
)
from deeppdcfr.utils import SpielState
import torch.optim as optim

torch.set_printoptions(precision=4, sci_mode=False)


class DREAM(OSDeepCFR):
    def __init__(
        self,
        game_name,
        advantage_buffer_size=100_0000,
        ave_policy_buffer_size=100_0000,
        baseline_buffer_size=100_0000,
        learning_rate=1e-4,
        num_episodes=400,
        num_traversals=20,
        advantage_network_train_steps=1,
        ave_policy_network_train_steps=1,
        baseline_network_train_steps=1,
        advantage_batch_size=-1,
        ave_policy_batch_size=-1,
        baseline_batch_size=-1,
        num_layers=2,
        num_hiddens=128,
        epsilon=0.6,
        evaluation_frequency=10,
        reinitialize_advantage_networks=True,
        linear_weighted=True,
        use_regret_matching_argmax=True,
        play_against_random=False,
        num_random_games=20000,
        fit_advantage=False,
        logger=None,
        device="cpu",
    ):
        super().__init__(
            game_name,
            advantage_buffer_size,
            ave_policy_buffer_size,
            learning_rate,
            num_episodes,
            num_traversals,
            advantage_network_train_steps,
            ave_policy_network_train_steps,
            advantage_batch_size,
            ave_policy_batch_size,
            num_layers,
            num_hiddens,
            epsilon,
            evaluation_frequency,
            reinitialize_advantage_networks,
            linear_weighted,
            use_regret_matching_argmax,
            play_against_random,
            num_random_games,
            logger,
            device,
        )
        self.baseline_buffer_size = baseline_buffer_size
        self.baseline_network_train_steps = baseline_network_train_steps
        self.baseline_batch_size = baseline_batch_size
        self.fit_advantage = fit_advantage
        root_state = self.game.new_initial_state()
        history_tensor = np.append(
            root_state.information_state_tensor(0),
            root_state.information_state_tensor(1),
        )
        network_layers = [num_hiddens for _ in range(num_layers)]
        self.history_size = len(history_tensor)
        self.q_value_trainer = QValueTrainer(
            self.history_size,
            self.infostate_size,
            self.action_size,
            network_layers,
            self.learning_rate,
            self.baseline_buffer_size,
            self.baseline_batch_size,
            self.baseline_network_train_steps,
            self.logger,
            False,
            self.regret_trainers,
            self.device,
        )
        # self.q_value_checker = QValueChecker(self.game, self)
        # self.q_value_checker.check()

    def solve(self):
        self.episode = 0
        advantage_losses = {0: [], 1: []}
        self.evaluate()
        for self.num_iteration in range(1, self.num_iterations + 1):
            for p in range(self.num_players):
                for _ in range(self.num_traversals):
                    self.episode += 1
                    root_state = self.skip_chance_state(self.game.new_initial_state())
                    self.dfs(root_state, p)
                baseline_loss = self.q_value_trainer.train_model()
                # self.q_value_checker.check()
                if self.reinitialize_advantage_networks:
                    self.regret_trainers[p].reset()
                advantage_loss = self.regret_trainers[p].train_model()
                advantage_losses[p].append(advantage_loss)
                if advantage_loss is not None:
                    self.logger.record("advantage_loss_{}".format(p), advantage_loss)
                if baseline_loss is not None:
                    self.logger.record("baseline_loss_{}".format(p), baseline_loss)
            if (
                self.num_iteration % self.evaluation_frequency == 0
                or self.num_iteration < self.evaluation_frequency
                and self.num_iteration % (self.evaluation_frequency // 3) == 0
                or self.num_iteration == self.num_iterations
            ):
                self.ave_policy_trainer.reset()
                ave_policy_loss = self.ave_policy_trainer.train_model()
                if ave_policy_loss is not None:
                    self.logger.record("ave_policy_loss", ave_policy_loss)
                self.evaluate()
        return advantage_losses, ave_policy_loss

    def evaluate(self):
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
        self.logger.record("episode", self.episode)
        self.logger.dump(step=self.episode)

    def dfs(self, s, traverser, my_reach=1.0, opp_reach=1.0, sample_reach=1.0):
        self.nodes_touched += 1
        player = s.current_player()
        if player == -4:
            return s.returns()[traverser] / self.max_utility
        legal_actions = s.legal_actions()
        policy = self.regret_trainers[player].get_policy(s)
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
        ns = self.skip_chance_state(s.child(action))
        # print()
        # print()
        # print()
        # print("start", s.history_str(), "apply", action, "to", ns.history_str(), policy, sample_policy)
        if player == 1 - traverser:
            self.ave_policy_trainer.add_data(
                s.information_state_tensor(player), policy, self.num_iteration
            )
            q_values = self.q_value_trainer.get_baseline(s, traverser)
            # exact_q_values = np.array(self.q_value_checker.exact_q_values[s.history_str()]) * (1 if traverser == 0 else -1)
            # q_values = exact_q_values
            # print(s, q_values, exact_q_values)
            action_value = self.dfs(
                ns,
                traverser,
                my_reach,
                opp_reach * prob,
                sample_reach * sample_prob,
            )
            q_values[action] += (action_value - q_values[action]) / sample_prob
            value = np.dot(q_values, policy)
            # print("p1", exact_q_values, action_value, q_values, value)
        else:
            q_values = self.q_value_trainer.get_baseline(s, traverser)
            # exact_q_values = np.array(self.q_value_checker.exact_q_values[s.history_str()]) * (1 if traverser == 0 else -1)
            # q_values = exact_q_values
            # print(s, q_values, exact_q_values)
            action_value = self.dfs(
                ns,
                traverser,
                my_reach * prob,
                opp_reach,
                sample_reach * sample_prob,
            )
            q_values[action] += (action_value - q_values[action]) / sample_prob
            value = np.dot(q_values, policy)

            cf_regrets = np.zeros_like(policy)
            im_weight = 1 if self.fit_advantage else opp_reach / sample_reach
            cf_regrets[legal_actions] = -value * im_weight
            cf_regrets[legal_actions] += q_values[legal_actions] * im_weight
            self.regret_trainers[player].add_data(
                s.information_state_tensor(player), cf_regrets, self.num_iteration
            )
        # print(s, "   ", action, "   ", ns, next_policy, int(ns.is_terminal()), ns.returns()[traverser])
        # print(s, "           ", action, "              ", ns, "       ", ns.is_terminal())
        self.q_value_trainer.add_data(
            np.append(s.information_state_tensor(0), s.information_state_tensor(1)),
            action,
            np.append(ns.information_state_tensor(0), ns.information_state_tensor(1)),
            ns.information_state_tensor() if not ns.is_terminal() else None,
            ns.legal_actions_mask() if not ns.is_terminal() else None,
            ns.current_player(),
            int(ns.is_terminal()),
            ns.returns()[0] / self.max_utility, 
        )

        return value


class QValueTrainer(Trainer):
    def __init__(
        self,
        history_size: int,
        state_size: int,
        action_size: int,
        network_layers: list[int],
        learning_rate: float,
        buffer_size: int,
        batch_size: int,
        train_steps: int,
        logger: Logger,
        linear_weighted: bool,
        regret_trainers: list[RegretTrainer],
        device: str = "cpu",
    ):
        super().__init__(
            history_size,
            action_size,
            network_layers,
            learning_rate,
            buffer_size,
            batch_size,
            train_steps,
            logger,
            linear_weighted,
            device,
        )
        self.state_size = state_size
        self.history_size = history_size
        self.action_size = action_size
        self.buffer = CircularBuffer(
            self.buffer_size, history_size, state_size, action_size, device=self.device
        )
        self.model = MLP(history_size, self.network_layers, action_size).to(self.device)
        self.target_model = MLP(
            self.history_size, self.network_layers, self.action_size
        ).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.regret_trainers = regret_trainers

    def get_baseline(self, s: SpielState, player: int) -> np.ndarray:
        history_tensor = np.append(
            s.information_state_tensor(0),
            s.information_state_tensor(1),
        )
        coef = 1 if player == 0 else -1
        baseline = self.forward(history_tensor) * coef
        baseline = baseline * np.array(s.legal_actions_mask(), dtype=float)
        # baseline =  np.zeros(s.num_distinct_actions(), dtype=float)
        return baseline

    def add_data(
        self,
        history,
        action,
        next_history,
        next_state,
        next_legal_actions_mask,
        next_player,
        done,
        reward,
    ):
        if done:
            next_state = np.zeros([self.state_size], dtype=float)
            next_legal_actions_mask = np.zeros([self.action_size], dtype=int)
            next_player = 0
        self.buffer.add(
            history,
            action,
            next_history,
            next_state,
            next_legal_actions_mask,
            next_player,
            done,
            reward,
        )

    def train_model(self):
        self.model = MLP(self.history_size, self.network_layers, self.action_size).to(
            self.device
        )
        self.target_model = MLP(
            self.history_size, self.network_layers, self.action_size
        ).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if self.batch_size > 0 and len(self.buffer) < self.batch_size:
            return
        best_loss = float("inf")
        self.best_model = MLP(
            self.history_size, self.network_layers, self.action_size
        ).to(self.device)
        for train_step in range(self.train_steps + 1):
            samples = self.buffer.sample(self.batch_size)
            (
                histories,
                next_histories,
                next_states,
                rewards,
                next_legal_actions_mask,
                next_players,
                actions,
                dones,
            ) = samples

            q_values = self.model(histories)  # [B, A]
            # actions [B]
            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]

            with torch.no_grad():
                next_q_values = self.target_model(next_histories)  # [B, A]

                # calculate next strategies
                players_next_stragies = []
                for player in [0, 1]:
                    p0_regrets = self.regret_trainers[player].predict(
                        next_states
                    )  # [B, A]
                    p0_legal_regrets = p0_regrets * next_legal_actions_mask  # [B, A]
                    p0_regrets_pos = torch.clamp(p0_regrets, min=0)  # [B, A]
                    p0_regrets_pos_sum = torch.sum(
                        p0_regrets_pos, dim=1, keepdim=True
                    )  # [B, 1]
                    p0_rm_next_strategies = (
                        p0_regrets_pos / p0_regrets_pos_sum
                    )  # [B, A]
                    _, p0_max_legal_action_id = torch.max(
                        torch.where(
                            next_legal_actions_mask == 1,
                            p0_legal_regrets,
                            torch.tensor(float("-inf"), device=self.device),
                        ),
                        dim=1,
                    )  # [B]
                    p0_max_next_strategies = F.one_hot(
                        p0_max_legal_action_id, self.action_size
                    )  # [B, A]
                    p0_next_strategies = torch.where(
                        p0_regrets_pos_sum == 0,
                        p0_max_next_strategies,
                        p0_rm_next_strategies,
                    )  # [B, A]
                    players_next_stragies.append(p0_next_strategies)

                next_strategies = torch.where(
                    next_players.unsqueeze(1) == 0,
                    players_next_stragies[0],
                    players_next_stragies[1],
                )

            target = rewards + (1 - dones) * torch.sum(
                next_q_values * next_strategies, dim=1, keepdim=False
            )

            loss = self.loss_fn(q_value, target)

            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            if train_step % 50 == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            if loss.item() < best_loss:
                best_loss = loss.item()
                self.best_model.load_state_dict(self.model.state_dict())

            if train_step % 100 == 0:
                print(
                    "train_step[{}/{}]: loss {}, best_loss {}".format(
                        train_step, self.train_steps, loss.item(), best_loss
                    )
                )

        self.model.load_state_dict(self.best_model.state_dict())
        return loss.item()


class CircularBuffer:
    def __init__(
        self, buffer_size, history_size, state_size, action_size, device="cpu"
    ):
        self.buffer_size = buffer_size
        self.history_size = history_size
        self.action_size = action_size
        self.state_size = state_size
        self.device = device
        self.reset()

    def reset(self):
        self.history_buf = np.ones([self.buffer_size, self.history_size], dtype=float)
        self.action_buf = np.ones([self.buffer_size], dtype=int)
        self.next_history_buf = np.ones(
            [self.buffer_size, self.history_size], dtype=float
        )
        self.next_state_buf = np.ones([self.buffer_size, self.state_size], dtype=float)
        self.next_legal_actions_mask_buf = np.ones(
            [self.buffer_size, self.action_size], dtype=int
        )
        self.next_player_buf = np.ones([self.buffer_size], dtype=int)
        self.done_buf = np.ones([self.buffer_size], dtype=int)
        self.reward_buf = np.ones([self.buffer_size], dtype=float)
        self.cur_id = 0

    def add(
        self,
        history,
        action,
        next_history,
        next_state,
        next_legal_actions_mask,
        next_player,
        done,
        reward,
    ):
        self.add_data(
            self.cur_id,
            history,
            action,
            next_history,
            next_state,
            next_legal_actions_mask,
            next_player,
            done,
            reward,
        )
        self.cur_id = (self.cur_id + 1) % self.buffer_size

    def add_data(
        self,
        idx,
        history,
        action,
        next_history,
        next_state,
        next_legal_actions_mask,
        next_player,
        done,
        reward,
    ):
        self.history_buf[idx] = history
        self.action_buf[idx] = action
        self.next_history_buf[idx] = next_history
        self.next_state_buf[idx] = next_state
        self.next_legal_actions_mask_buf[idx] = next_legal_actions_mask
        self.next_player_buf[idx] = next_player
        self.done_buf[idx] = done
        self.reward_buf[idx] = reward

    def sample(self, num_samples=-1):
        data_length = len(self)
        if num_samples == -1:
            idxs = list(range(data_length))
        else:
            idxs = random.sample(range(data_length), num_samples)
        float_data = (
            self.history_buf[idxs],
            self.next_history_buf[idxs],
            self.next_state_buf[idxs],
            self.reward_buf[idxs],
        )
        int_data = (
            self.next_legal_actions_mask_buf[idxs],
            self.next_player_buf[idxs],
            self.action_buf[idxs],
            self.done_buf[idxs],
        )
        data_tensor = (
            *map(self.numpy_to_float_tensor, float_data),
            *map(self.numpy_to_int_tensor, int_data),
        )
        return data_tensor

    def numpy_to_float_tensor(self, data):
        return torch.as_tensor(data, dtype=torch.float32, device=self.device)

    def numpy_to_int_tensor(self, data):
        return torch.as_tensor(data, dtype=torch.int64, device=self.device)

    def __len__(self):
        return min(self.cur_id, self.buffer_size)


class QValueChecker:
    def __init__(self, game, solver):
        self.game = game
        self.solver = solver

    def check(self):
        self.compute_exact_value()
        errors = []
        for state_str, exact_q_value in self.exact_q_values.items():
            state = self.states[state_str]
            q_value = self.solver.q_value_trainer.get_baseline(state, 0)
            error = np.mean(np.square(q_value - exact_q_value))
            errors.append(error)
            # print(state, error, q_value, exact_q_value)
        print("mean error", np.mean(errors))

    def compute_exact_value(self):
        self.exact_q_values = {}
        self.states = {}
        self.dfs(self.game.new_initial_state(), 0)

    def dfs(self, state, player):
        if state.is_terminal():
            value = state.returns()[player] / self.solver.max_utility
        elif state.is_chance_node():
            value = 0
            for action, prob in state.chance_outcomes():
                new_state = state.child(action)
                child_value = self.dfs(new_state, player)
                value += prob * child_value
        else:
            policy = self.solver.regret_trainers[state.current_player()].get_policy(
                state
            )
            # print(state, policy)
            value = 0
            q_values = [0 for _ in range(state.num_distinct_actions())]
            for action in state.legal_actions():
                new_state = state.child(action)
                q_values[action] = self.dfs(new_state, player)
                value += policy[action] * q_values[action]
            self.states[state.history_str()] = state
            self.exact_q_values[state.history_str()] = q_values
        return value


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
