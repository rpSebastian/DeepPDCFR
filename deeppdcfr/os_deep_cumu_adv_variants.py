import math

import numpy as np
import torch
import torch.optim as optim
from deeppdcfr.logger import Logger

from deeppdcfr.os_deep_cumu_adv import MLP, DeepCumuAdv, RegretTrainer
from deeppdcfr.utils import SpielState



class DeepDCFRPlus(DeepCumuAdv):
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
        gamma=4,
        play_against_random=False,
        num_random_games=20000,
        alpha=1.5,
        device="cpu",
        seed=0,
    ):
        self.alpha = alpha
        super().__init__(
            game_name,
            num_episodes,
            advantage_buffer_size,
            ave_policy_buffer_size,
            learning_rate,
            num_traversals,
            advantage_network_train_steps,
            ave_policy_network_train_steps,
            advantage_batch_size,
            ave_policy_batch_size,
            num_layers,
            num_hiddens,
            evaluation_frequency,
            reinitialize_advantage_networks,
            use_regret_matching_argmax,
            epsilon,
            logger,
            gamma,
            play_against_random,
            num_random_games,
            device,
            seed,
        )

    def init_regret_trainers(self):
        self.regret_trainers = [
            DCFRPlusRegretTrainer(
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
                self.alpha,
            )
            for _ in range(self.num_players)
        ]



class DeepPDCFRPlus(DeepCumuAdv):
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
        reinitialize_imm_regret_networks=True,
        use_regret_matching_argmax=True,
        epsilon=0.6,
        logger=None,
        alpha=2.3,
        gamma=5,
        play_against_random=False,
        num_random_games=20000,
        device="cpu",
        seed=0,
    ):
        self.alpha = alpha
        self.reinitialize_imm_regret_networks = reinitialize_imm_regret_networks
        super().__init__(
            game_name,
            num_episodes,
            advantage_buffer_size,
            ave_policy_buffer_size,
            learning_rate,
            num_traversals,
            advantage_network_train_steps,
            ave_policy_network_train_steps,
            advantage_batch_size,
            ave_policy_batch_size,
            num_layers,
            num_hiddens,
            evaluation_frequency,
            reinitialize_advantage_networks,
            use_regret_matching_argmax,
            epsilon,
            logger,
            gamma,
            play_against_random,
            num_random_games,
            device,
            seed,
        )

    def init_regret_trainers(self):
        self.regret_trainers = [
            PDCFRPlusRegretTrainer(
                self.infostate_size,
                self.action_size,
                self.network_layers,
                self.learning_rate,
                self.advantage_buffer_size,
                self.advantage_batch_size,
                self.advantage_network_train_steps,
                self.logger,
                self.use_regret_matching_argmax,
                self.reinitialize_imm_regret_networks,
                self.device,
                self.alpha,
            )
            for _ in range(self.num_players)
        ]

    
class DCFRPlusRegretTrainer(RegretTrainer):
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
        device: str,
        alpha: float,
    ):
        self.alpha = alpha
        super().__init__(
            input_size,
            output_size,
            network_layers,
            learning_rate,
            buffer_size,
            batch_size,
            train_steps,
            logger,
            use_regret_matching_argmax,
            device,
        )

    def compute_loss(self, samples, T):
        infostates, cf_regrets, legal_actions_mask, iterations = samples
        with torch.no_grad():
            target_outputs = self.predict(
                self.target_model, infostates, legal_actions_mask
            )
        zero = torch.zeros_like(target_outputs)
        target_outputs = torch.maximum(target_outputs, zero)
        regrets = (
            target_outputs
            * math.pow(T - 1, self.alpha)
            / (math.pow(T - 1, self.alpha) + 1.5)
            + cf_regrets
        )
        outputs = self.predict(self.model, infostates, legal_actions_mask)
        loss = self.loss_fn(outputs, regrets)
        return loss


class PDCFRPlusRegretTrainer(RegretTrainer):
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
        reinitialize_imm_regret_networks: bool,
        use_regret_matching_argmax: bool,
        device: str = "cpu",
        alpha: float = 2.3,
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
            use_regret_matching_argmax,
            device,
        )
        self.alpha = alpha
        self.reinitialize_imm_regret_networks = reinitialize_imm_regret_networks
        self.imm_model = MLP(self.input_size, self.network_layers, self.output_size).to(
            self.device
        )
        self.imm_optimizer = torch.optim.Adam(
            self.imm_model.parameters(), lr=self.learning_rate
        )

    def reset(self):
        super().reset()
        if self.reinitialize_imm_regret_networks:
            self.imm_model.reset_parameters()
            self.imm_model.to(self.device)
            self.imm_optimizer = optim.Adam(
                self.model.parameters(), lr=self.learning_rate
            )

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

            loss = self.compute_imm_loss(samples, T)
            self.imm_optimizer.zero_grad()
            loss.backward()
            self.imm_optimizer.step()
            if train_step % 100 == 0:
                self.logger.info(
                    "[{}/{}] imm regret loss: {}".format(
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
        zero = torch.zeros_like(target_outputs)
        target_outputs = torch.maximum(target_outputs, zero)
        regrets = (
            target_outputs
            * math.pow(T - 1, self.alpha)
            / (math.pow(T - 1, self.alpha) + 1)
            + cf_regrets
        )
        outputs = self.predict(self.model, infostates, legal_actions_mask)
        loss = self.loss_fn(outputs, regrets)
        return loss

    def compute_imm_loss(self, samples, T):
        infostates, cf_regrets, legal_actions_mask, iterations = samples
        outputs = self.predict(self.imm_model, infostates, legal_actions_mask)
        loss = self.loss_fn(outputs, cf_regrets)
        return loss

    def get_policy(self, s: SpielState, T: int) -> np.ndarray:
        regrets = self.get_regrets(s)
        imm_regrets = self.get_imm_regrets(s)
        legal_actions = s.legal_actions()
        return self.predictive_regret_matching(regrets, imm_regrets, legal_actions, T)

    def get_imm_regrets(self, s: SpielState) -> np.ndarray:
        imm_regrets = self.forward(
            self.imm_model, s.information_state_tensor(), s.legal_actions_mask()
        )
        return imm_regrets

    def predictive_regret_matching(self, regrets, imm_regrets, legal_actions, T):
        predictive_regrets = np.maximum(
            np.maximum(regrets, 0)
            * np.power(T - 1, self.alpha)
            / (np.power(T - 1, self.alpha) + 1)
            + imm_regrets,
            0,
        )
        return self.regret_matching(predictive_regrets, legal_actions)

