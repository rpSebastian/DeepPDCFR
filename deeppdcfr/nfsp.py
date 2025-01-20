# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
# Copyright 2021 Artificial Intelligence Center, Czech Techical University
# Copied and adapted from OpenSpiel (https://github.com/deepmind/open_spiel)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NFSP agents trained on Leduc Poker."""

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from open_spiel.python import policy, rl_environment
from open_spiel.python.algorithms import exploitability, nfsp

from xhlib.logger import Logger
from xhlib.utils import set_seed

from xdcfr.game import read_game_config
from xdcfr.utils import play_n_games_against_random, play_n_poker_games_against_random


class NFSP:
    def __init__(
        self,
        game_name="KuhnPoker",
        num_train_episodes=int(10e7),
        eval_every=10000,
        num_hidden=128,
        num_layers=1,
        replay_buffer_capacity=int(2e5),
        reservoir_buffer_capacity=int(2e6),
        min_buffer_size_to_learn=1000,
        anticipatory_param=0.1,
        batch_size=128,
        learn_every=64,
        rl_learning_rate=0.01,
        sl_learning_rate=0.01,
        optimizer_str="sgd",
        loss_str="mse",
        update_target_network_every=19200,
        discount_factor=1.0,
        epsilon_decay_duration=int(20e6),
        epsilon_start=0.06,
        epsilon_end=0.001,
        play_against_random=False,
        num_random_games=20000,
        logger=None,
        device="cpu",
        seed=0,
    ):
        self.game_name = game_name
        self.num_train_episodes = num_train_episodes
        self.eval_every = eval_every
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.replay_buffer_capacity = replay_buffer_capacity
        self.reservoir_buffer_capacity = reservoir_buffer_capacity
        self.min_buffer_size_to_learn = min_buffer_size_to_learn
        self.anticipatory_param = anticipatory_param
        self.batch_size = batch_size
        self.learn_every = learn_every
        self.rl_learning_rate = rl_learning_rate
        self.sl_learning_rate = sl_learning_rate
        self.optimizer_str = optimizer_str
        self.loss_str = loss_str
        self.update_target_network_every = update_target_network_every
        self.discount_factor = discount_factor
        self.epsilon_decay_duration = epsilon_decay_duration
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.num_players = 2
        self.play_against_random = play_against_random
        self.num_random_games = num_random_games
        self.logger = logger or Logger(["stdout"])
        self.seed = seed
        self.nodes_touched = 0
        self.device = device

        if self.device != "cpu" and "cuda:" not in self.device:
            raise ValueError(
                "Invalid device: {}, which should be cpu or cuda:x".format(self.device)
            )
        if self.device == "cpu":
            self.gpu_config = tf.ConfigProto(
                device_count={"GPU": 0},
            )
        else:
            self.gpu_config = tf.ConfigProto(
                device_count={"GPU": 1},
                gpu_options=tf.GPUOptions(
                    allow_growth=True, visible_device_list=self.device.split(":")[-1]
                ),
            )

        set_seed(seed)

    def solve(self):
        game_config = read_game_config(self.game_name)
        self.poker_game = game_config.poker
        if game_config.large_game:
            self.play_against_random = True
        game = game_config.load_game()
        max_utility = max(game.max_utility(), abs(game.min_utility()))
        env_configs = {"players": self.num_players}
        env = rl_environment.Environment(game, **env_configs)
        env.seed(self.seed)
        info_state_size = env.observation_spec()["info_state"][0]
        num_actions = env.action_spec()["num_actions"]

        hidden_layers_sizes = [self.num_hidden for _ in range(self.num_layers)]
        kwargs = {
            "replay_buffer_capacity": self.replay_buffer_capacity,
            "reservoir_buffer_capacity": self.reservoir_buffer_capacity,
            "min_buffer_size_to_learn": self.min_buffer_size_to_learn,
            "anticipatory_param": self.anticipatory_param,
            "batch_size": self.batch_size,
            "learn_every": self.learn_every,
            "rl_learning_rate": self.rl_learning_rate,
            "sl_learning_rate": self.sl_learning_rate,
            "optimizer_str": self.optimizer_str,
            "loss_str": self.loss_str,
            "update_target_network_every": self.update_target_network_every,
            "discount_factor": self.discount_factor,
            "epsilon_decay_duration": self.epsilon_decay_duration,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
        }

        tf.reset_default_graph()
        with tf.Session(config=self.gpu_config) as sess:
            tf.random.set_random_seed(self.seed)
            # pylint: disable=g-complex-comprehension
            agents = [
                nfsp.NFSP(
                    sess,
                    idx,
                    info_state_size,
                    num_actions,
                    hidden_layers_sizes,
                    **kwargs
                )
                for idx in range(self.num_players)
            ]
            joint_avg_policy = NFSPPolicies(env, agents, nfsp.MODE.average_policy)

            sess.run(tf.global_variables_initializer())

            self.evaluate(env, agents, joint_avg_policy, 0)
            for ep in range(1, self.num_train_episodes + 1):
                if (
                    ep % self.eval_every == 0
                    or ep < self.eval_every
                    and ep % (self.eval_every // 3) == 0
                ):
                    self.evaluate(env, agents, joint_avg_policy, ep)

                time_step = env.reset()
                while not time_step.last():
                    player_id = time_step.observations["current_player"]
                    agent_output = agents[player_id].step(time_step)
                    action_list = [agent_output.action]
                    time_step = env.step(action_list)
                    time_step = rl_environment.TimeStep(
                        observations=time_step.observations,
                        rewards=[
                            time_step.rewards[p] / max_utility
                            for p in range(self.num_players)
                        ],
                        discounts=time_step.discounts,
                        step_type=time_step.step_type,
                    )

                # Episode is over, step all agents with final info state.
                for agent in agents:
                    agent.step(time_step)

    def evaluate(self, env, agents, joint_avg_policy, ep):
        losses = [agent.loss for agent in agents]
        self.logger.info("Losses: {}".format(losses))

        if self.play_against_random:
            if self.poker_game:
                reward = play_n_poker_games_against_random(
                    env.game,
                    joint_avg_policy,
                    self.num_random_games,
                )
            else:
                reward = play_n_games_against_random(
                    env.game,
                    joint_avg_policy,
                    self.num_random_games,
                )
            self.logger.record("reward", reward)
        else:
            expl = exploitability.exploitability(env.game, joint_avg_policy)
            self.logger.record("exp", expl)

        nodes_touched = agents[0]._step_counter + agents[1]._step_counter
        self.logger.record("episode", ep)
        self.logger.record("nodes_touched", nodes_touched)
        self.logger.dump(step=ep)


class NFSPPolicies(policy.Policy):
    """Joint policy to be evaluated."""

    def __init__(self, env, nfsp_policies, mode):
        game = env.game
        self.num_players = 2
        player_ids = list(range(self.num_players))
        super(NFSPPolicies, self).__init__(game, player_ids)
        self._policies = nfsp_policies
        self._mode = mode
        self._obs = {
            "info_state": [None] * self.num_players,
            "legal_actions": [None] * self.num_players,
        }

    def action_probabilities(self, state, player_id=None):
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)

        self._obs["current_player"] = cur_player
        self._obs["info_state"][cur_player] = state.information_state_tensor(cur_player)
        self._obs["legal_actions"][cur_player] = legal_actions

        info_state = rl_environment.TimeStep(
            observations=self._obs, rewards=None, discounts=None, step_type=None
        )

        with self._policies[cur_player].temp_mode_as(self._mode):
            p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
        prob_dict = {action: p[action] for action in legal_actions}
        return prob_dict
