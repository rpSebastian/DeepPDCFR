import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from open_spiel.python import policy, rl_environment
from open_spiel.python.algorithms import policy_gradient
from deeppdcfr.logger import Logger

from deeppdcfr.game import read_game_config
from deeppdcfr.utils import (
    evalute_explotability,
    play_n_games_against_random,
    play_n_poker_games_against_random,
    set_seed
)


class PolicyGradient:
    def __init__(
        self,
        num_episodes=int(1e6),
        game_name="KuhnPoker",
        loss_str="",
        num_hidden=64,
        num_layers=1,
        batch_size=16,
        entropy_cost=0.001,
        critic_learning_rate=0.01,
        pi_learning_rate=0.01,
        num_critic_before_pi=4,
        logfreq=100,
        logger=None,
        play_against_random=False,
        num_random_games=20000,
        device="cpu",
        seed=0,
    ):
        self.game_name = game_name
        self.num_episodes = num_episodes
        self.loss_str = loss_str
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.entropy_cost = entropy_cost
        self.critic_learning_rate = critic_learning_rate
        self.pi_learning_rate = pi_learning_rate
        self.num_critic_before_pi = num_critic_before_pi
        self.logfreq = logfreq
        self.logger = logger or Logger(["stdout"])
        self.seed = seed
        self.play_against_random = play_against_random
        self.num_random_games = num_random_games
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
        self.logger.info(
            f"\nnum_episodes: {num_episodes}"
            f"\nnum_hidden: {num_hidden}"
            f"\nnum_layers: {num_layers}"
            f"\nbatch_size: {batch_size}"
            f"\nentropy_cost: {entropy_cost}"
            f"\ncritic_learning_rate: {critic_learning_rate}"
            f"\npi_learning_rate: {pi_learning_rate}"
            f"\nnum_critic_before_pi: {num_critic_before_pi}"
            f"\nlogfreq: {logfreq}",
            f"\ndevice: {device}",
        )

    def solve(self):
        game_config = read_game_config(self.game_name)
        self.poker_game = game_config.poker
        if game_config.large_game:
            self.play_against_random = True
        game = game_config.load_game()
        max_utility = max(game.max_utility(), abs(game.min_utility()))
        num_players = 2
        env_configs = {"players": num_players}
        env = rl_environment.Environment(game, **env_configs)
        env.seed(self.seed)

        info_state_size = env.observation_spec()["info_state"][0]
        num_actions = env.action_spec()["num_actions"]
        tf.reset_default_graph()
        with tf.Session(config=self.gpu_config) as sess:
            tf.random.set_random_seed(self.seed)
            # pylint: disable=g-complex-comprehension
            hidden_layers_sizes = tuple(
                [int(self.num_hidden) for _ in range(int(self.num_layers))]
            )
            agents = [
                policy_gradient.PolicyGradient(
                    sess,
                    idx,
                    info_state_size,
                    num_actions,
                    loss_str=self.loss_str,
                    hidden_layers_sizes=hidden_layers_sizes,
                    batch_size=self.batch_size,
                    entropy_cost=self.entropy_cost,
                    critic_learning_rate=self.critic_learning_rate,
                    pi_learning_rate=self.pi_learning_rate,
                    num_critic_before_pi=self.num_critic_before_pi,
                )
                for idx in range(num_players)
            ]
            expl_policies_avg = PolicyGradientPolicies(env, agents)

            sess.run(tf.global_variables_initializer())
            self.evaluate(env, agents, expl_policies_avg, 0)
            for ep in range(1, self.num_episodes + 1):
                if (
                    ep % self.logfreq == 0
                    or ep < self.logfreq
                    and ep % (self.logfreq // 3) == 0
                ):
                    self.evaluate(env, agents, expl_policies_avg, ep)

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
                            for p in range(num_players)
                        ],
                        discounts=time_step.discounts,
                        step_type=time_step.step_type,
                    )

                # Episode is over, step all agents with final info state.
                for agent in agents:
                    agent.step(time_step)

    def evaluate(self, env, agents, expl_policies_avg, ep):
        if self.play_against_random:
            if self.poker_game:
                reward = play_n_poker_games_against_random(
                    env.game,
                    expl_policies_avg,
                    self.num_random_games,
                )
            else:
                reward = play_n_games_against_random(
                    env.game,
                    expl_policies_avg,
                    self.num_random_games,
                )
            self.logger.record("reward", reward)
        else:
            expl = evalute_explotability(env.game, expl_policies_avg)
            self.logger.record("exp", expl)

        nodes_touched = agents[0]._step_counter + agents[1]._step_counter
        self.logger.record("episode", ep)
        self.logger.record("nodes_touched", nodes_touched)
        self.logger.dump(step=ep)


class PolicyGradientPolicies(policy.Policy):
    """Joint policy to be evaluated."""

    def __init__(self, env, nfsp_policies):
        game = env.game
        player_ids = [0, 1]
        super(PolicyGradientPolicies, self).__init__(game, player_ids)
        self._policies = nfsp_policies
        self._obs = {"info_state": [None, None], "legal_actions": [None, None]}

    def action_probabilities(self, state, player_id=None):
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)

        self._obs["current_player"] = cur_player
        self._obs["info_state"][cur_player] = state.information_state_tensor(cur_player)
        self._obs["legal_actions"][cur_player] = legal_actions

        info_state = rl_environment.TimeStep(
            observations=self._obs, rewards=None, discounts=None, step_type=None
        )

        p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
        prob_dict = {action: p[action] for action in legal_actions}
        return prob_dict


class A2C(PolicyGradient):
    def __init__(
        self,
        num_episodes=int(1e6),
        game_name="KuhnPoker",
        num_hidden=64,
        num_layers=1,
        batch_size=16,
        entropy_cost=0.001,
        critic_learning_rate=0.01,
        pi_learning_rate=0.01,
        num_critic_before_pi=4,
        logfreq=100,
        logger=None,
        play_against_random=False,
        num_random_games=20000,
        device="cpu",
        seed=0,
    ):
        super().__init__(
            num_episodes=num_episodes,
            game_name=game_name,
            loss_str="a2c",
            num_hidden=num_hidden,
            num_layers=num_layers,
            batch_size=batch_size,
            entropy_cost=entropy_cost,
            critic_learning_rate=critic_learning_rate,
            pi_learning_rate=pi_learning_rate,
            num_critic_before_pi=num_critic_before_pi,
            logfreq=logfreq,
            logger=logger,
            play_against_random=play_against_random,
            num_random_games=num_random_games,
            device=device,
            seed=seed,
        )


class RPG(PolicyGradient):
    def __init__(
        self,
        num_episodes=int(1e6),
        game_name="KuhnPoker",
        num_hidden=64,
        num_layers=1,
        batch_size=16,
        entropy_cost=0.001,
        critic_learning_rate=0.01,
        pi_learning_rate=0.01,
        num_critic_before_pi=4,
        logfreq=100,
        logger=None,
        play_against_random=False,
        num_random_games=20000,
        device="cpu",
        seed=0,
    ):
        super().__init__(
            num_episodes=num_episodes,
            game_name=game_name,
            loss_str="rpg",
            num_hidden=num_hidden,
            num_layers=num_layers,
            batch_size=batch_size,
            entropy_cost=entropy_cost,
            critic_learning_rate=critic_learning_rate,
            pi_learning_rate=pi_learning_rate,
            num_critic_before_pi=num_critic_before_pi,
            logfreq=logfreq,
            logger=logger,
            play_against_random=play_against_random,
            num_random_games=num_random_games,
            device=device,
            seed=seed,
        )


class QPG(PolicyGradient):
    def __init__(
        self,
        num_episodes=int(1e6),
        game_name="KuhnPoker",
        num_hidden=64,
        num_layers=1,
        batch_size=16,
        entropy_cost=0.001,
        critic_learning_rate=0.01,
        pi_learning_rate=0.01,
        num_critic_before_pi=4,
        logfreq=100,
        logger=None,
        play_against_random=False,
        num_random_games=20000,
        device="cpu",
        seed=0,
    ):
        super().__init__(
            num_episodes=num_episodes,
            game_name=game_name,
            loss_str="qpg",
            num_hidden=num_hidden,
            num_layers=num_layers,
            batch_size=batch_size,
            entropy_cost=entropy_cost,
            critic_learning_rate=critic_learning_rate,
            pi_learning_rate=pi_learning_rate,
            num_critic_before_pi=num_critic_before_pi,
            logfreq=logfreq,
            logger=logger,
            play_against_random=play_against_random,
            num_random_games=num_random_games,
            device=device,
            seed=seed,
        )





