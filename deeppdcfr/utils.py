import datetime
import importlib
import inspect
import pickle
import random
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Type
from difflib import SequenceMatcher
import numpy as np
import torch

from typing import Any, Callable, Union

import numpy as np
import pyspiel

SpielState = Union[pyspiel.State, Any]
SpielGame = Union[pyspiel.Game, Any]

from open_spiel.python.algorithms import exploitability
from open_spiel.python.policy import tabular_policy_from_callable


def rescale_func(callable_func):
    def rescaled_callable_func(state: SpielState) -> dict:
        probs = callable_func(state)
        probs_sum = sum(probs.values())
        return {a: p / probs_sum for a, p in probs.items()}

    return rescaled_callable_func


def evalute_explotability(game: SpielGame, callable_func: Callable) -> float:
    policy = tabular_policy_from_callable(game, rescale_func(callable_func))
    exp = exploitability.exploitability(game, policy)
    return exp


def play_n_games_against_random(
    game: SpielGame, callable_func: Callable, num_random_games: int
) -> float:
    total_reward = 0
    for i in range(num_random_games // 2):
        reward = play_game_against_random(game, rescale_func(callable_func))
        total_reward += reward
    return total_reward / num_random_games


def play_game_against_random(game: SpielGame, callable_func: Callable) -> float:
    # play one game per player
    reward = 0
    for player in [0, 1]:
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes, probs = zip(*state.chance_outcomes())
                aidx = np.random.choice(range(len(outcomes)), p=probs)
                action = outcomes[aidx]
            else:
                cur_player = state.current_player()
                if cur_player == player:
                    probs = callable_func(state)
                    action = np.random.choice(
                        list(probs.keys()), p=list(probs.values())
                    )
                elif cur_player == 1 - player:
                    action = random.choice(state.legal_actions())
                else:
                    print("Got player ", str(cur_player))
                    break
            state.apply_action(action)
        reward += state.returns()[player]
    return reward


def play_n_poker_games_against_random(
    game: SpielGame, callable_func: Callable, num_random_games: int
) -> float:
    opponents = [
        AlwaysCallOpponent(),
        AlwaysBetOpponent(),
        RandomOpponent(),
        RandomCallBetOpponent(),
        MaybeBetOpponent(),
    ]
    total_reward = 0
    for i in range(num_random_games // 2 // len(opponents)):
        reward = play_poker_game_against_random(
            game, rescale_func(callable_func), opponents
        )
        total_reward += reward
    return total_reward / num_random_games


def play_poker_game_against_random(
    game: SpielGame, callable_func: Callable, opponents
) -> float:
    reward = 0
    for opponent in opponents:
        for player in [0, 1]:
            state = game.new_initial_state()
            while not state.is_terminal():
                if state.is_chance_node():
                    outcomes, probs = zip(*state.chance_outcomes())
                    aidx = np.random.choice(range(len(outcomes)), p=probs)
                    action = outcomes[aidx]
                else:
                    cur_player = state.current_player()
                    if cur_player == player:
                        probs = callable_func(state)
                        action = np.random.choice(
                            list(probs.keys()), p=list(probs.values())
                        )
                    elif cur_player == 1 - player:
                        probs = opponent(state)
                        action = np.random.choice(
                            list(probs.keys()), p=list(probs.values())
                        )
                    else:
                        print("Got player ", str(cur_player))
                        break
                state.apply_action(action)
            reward += state.returns()[player]
    return reward


class AlwaysFoldOpponent:
    def __call__(self, state: SpielState):
        policy = {}
        legal_actions = state.legal_actions()
        for action in state.legal_actions():
            policy[action] = 0
        if 0 in legal_actions:
            policy[0] = 1
        else:
            policy[1] = 1

        assert sum(list(policy.values())) == 1
        return policy


class AlwaysCallOpponent:
    def __call__(self, state: SpielState):
        policy = {}
        legal_actions = state.legal_actions()
        for action in state.legal_actions():
            policy[action] = 0
        policy[1] = 1

        assert sum(list(policy.values())) == 1
        return policy


class AlwaysBetOpponent:
    def __call__(self, state: SpielState):
        policy = {}
        legal_actions = state.legal_actions()
        for action in state.legal_actions():
            policy[action] = 0
        if 2 in legal_actions:
            policy[2] = 1
        else:
            policy[1] = 1

        assert sum(list(policy.values())) == 1
        return policy


class RandomOpponent:
    def __call__(self, state: SpielState):
        policy = {}
        legal_actions = state.legal_actions()
        for action in state.legal_actions():
            policy[action] = 1 / len(legal_actions)

        assert sum(list(policy.values())) == 1
        return policy


class RandomCallBetOpponent:
    def __call__(self, state: SpielState):
        policy = {}
        legal_actions = state.legal_actions()
        for action in state.legal_actions():
            policy[action] = 0
        if 2 in legal_actions:
            policy[1] = 0.5
            policy[2] = 0.5
        else:
            policy[1] = 1

        assert sum(list(policy.values())) == 1
        return policy


class MaybeBetOpponent:
    def __call__(self, state: SpielState):
        policy = {}
        legal_actions = state.legal_actions()
        for action in state.legal_actions():
            policy[action] = 0
        if 2 in legal_actions:
            policy[1] = 0.7
            policy[2] = 0.3
        else:
            policy[1] = 1

        assert sum(list(policy.values())) == 1
        return policy


def load_module(name):
    if ":" in name:
        mod_name, attr_name = name.split(":")
    else:
        li = name.split(".")
        mod_name, attr_name = ".".join(li[:-1]), li[-1]
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_pickle(data, file):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, "wb") as f:
        pickle.dump(data, f)


def load_pickle(file):
    file = Path(file)
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data


def string_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def init_object(init_class: Type[object], possible_args: Dict[str, Any], **kwargs):
    possible_args_copy = deepcopy(possible_args)
    for k, v in kwargs.items():
        possible_args_copy[k] = v
    args = inspect.getfullargspec(init_class.__init__).args
    for possible_key in possible_args_copy:
        if possible_key in args:
            continue
        for key in args:
            if 0.8 <= string_similarity(possible_key, key) < 1:
                print(possible_key, key, string_similarity(possible_key, key))
                text = "Possible spelling error between passed {} and true {}".format(
                    possible_key, key
                )
                raise ValueError(text)
    params = {k: v for k, v in possible_args_copy.items() if k in args}
    new_object = init_class(**params)
    return new_object


def run_method(
    method: Callable,
    possible_args: Dict[str, Any],
    **kwargs,
):
    possible_args_copy = deepcopy(possible_args)
    for k, v in kwargs.items():
        possible_args_copy[k] = v
    args = inspect.getfullargspec(method).args
    for possible_key in possible_args_copy:
        if possible_key in args:
            continue
        for key in args:
            if 0.8 <= string_similarity(possible_key, key) < 1:
                print(possible_key, key, string_similarity(possible_key, key))
                text = "Possible spelling error between passed {} and true {}".format(
                    possible_key, key
                )
                raise ValueError(text)
    params = {k: v for k, v in possible_args_copy.items() if k in args}
    result = method(**params)
    return result

def get_host_ip():
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip

def get_server_id():
    ip = get_host_ip()
    server_id = int(ip.split(".")[-1])
    return server_id
