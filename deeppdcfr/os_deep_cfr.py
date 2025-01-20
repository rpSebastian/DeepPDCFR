import numpy as np

from xdcfr.deep_cfr import DeepCFR
from xdcfr.utils import (
    evalute_explotability,
    play_n_games_against_random,
    play_n_poker_games_against_random,
)


class OSDeepCFR(DeepCFR):
    def __init__(
        self,
        game_name,
        advantage_buffer_size=100_0000,
        ave_policy_buffer_size=100_0000,
        learning_rate=1e-4,
        num_episodes=400,
        num_traversals=20,
        advantage_network_train_steps=1,
        ave_policy_network_train_steps=1,
        advantage_batch_size=-1,
        ave_policy_batch_size=-1,
        num_layers=2,
        num_hiddens=128,
        epsilon=0.6,
        evaluation_frequency=10,
        reinitialize_advantage_networks=True,
        linear_weighted=True,
        use_regret_matching_argmax=True,
        play_against_random=False,
        num_random_games=20000,
        logger=None,
        device="cpu",
    ):
        num_iterations = num_episodes // (num_traversals * 2)
        super().__init__(
            game_name,
            advantage_buffer_size,
            ave_policy_buffer_size,
            learning_rate,
            num_iterations,
            num_traversals,
            advantage_network_train_steps,
            ave_policy_network_train_steps,
            advantage_batch_size,
            ave_policy_batch_size,
            num_layers,
            num_hiddens,
            evaluation_frequency,
            reinitialize_advantage_networks,
            linear_weighted,
            use_regret_matching_argmax,
            play_against_random,
            num_random_games,
            logger,
            device,
        )
        self.epsilon = epsilon
        self.max_utility = max(self.game.max_utility(), abs(self.game.min_utility()))

    def solve(self):
        self.episode = 0
        root_state = self.game.new_initial_state()
        advantage_losses = {0: [], 1: []}
        self.evaluate()
        for self.num_iteration in range(1, self.num_iterations + 1):
            for p in range(self.num_players):
                for _ in range(self.num_traversals):
                    self.episode += 1
                    self.dfs(root_state, p)
                if self.reinitialize_advantage_networks:
                    self.regret_trainers[p].reset()
                advantage_loss = self.regret_trainers[p].train_model()
                advantage_losses[p].append(advantage_loss)
                if advantage_loss is not None:
                    self.logger.record("advantage_loss_{}".format(p), advantage_loss)

            if (
                self.num_iteration % self.evaluation_frequency == 0
                or self.num_iteration < self.evaluation_frequency
                and self.num_iteration % (self.evaluation_frequency // 3) == 0
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

        if player == 1 - traverser:
            self.ave_policy_trainer.add_data(
                s.information_state_tensor(player), policy, self.num_iteration
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
                s.information_state_tensor(player), cf_regrets, self.num_iteration
            )
            return value


class OSDeepCFR2(OSDeepCFR):
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

        if player == 1 - traverser:
            self.ave_policy_trainer.add_data(
                s.information_state_tensor(player), policy, self.num_iteration
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
            cf_regrets[legal_actions] = -value
            cf_regrets[action] += q_values[action]
            self.regret_trainers[player].add_data(
                s.information_state_tensor(player), cf_regrets, self.num_iteration
            )
            return value


class VaniOSDeepCFR(OSDeepCFR):
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
                s.information_state_tensor(player), cf_regrets, self.num_iteration
            )
            self.ave_policy_trainer.add_data(
                s.information_state_tensor(player),
                policy * my_reach / sample_reach,
                self.num_iteration,
            )
            return value
