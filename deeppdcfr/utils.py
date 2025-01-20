import random
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


def test_play_against_random():
    from xdcfr.game import read_game_config

    game_config = read_game_config("UniversalFHP2_5")
    game = game_config.load_game()

    def policy(state: SpielState):
        return {a: 1 / len(state.legal_actions()) for a in state.legal_actions()}

    reward = play_n_poker_games_against_random(game, policy, 10000)
    print(reward)


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


def init_cuda():
    a = np.array([100, 500])
    a_cuda = torch.from_numpy(a).cuda()


class Timer:
    def __init__(self):
        self.func_single_time = defaultdict(float)
        self.func_total_time = defaultdict(float)

    def timer(self, func):
        def wrap_func(*args, **kwargs):
            t1 = time.time()
            self.active = True
            result = func(*args, **kwargs)
            self.activate = False
            t2 = time.time()
            if not self.activate:
                self.func_single_time[func.__name__] = t2 - t1
                self.func_total_time[func.__name__] += t2 - t1
            return result

        return wrap_func

    def reset(self):
        self.func_single_time = defaultdict(float)
        self.func_total_time = defaultdict(float)

    def print_total(self):
        for key, value in self.func_total_time.items():
            print(f"{key}: {value:.4f}", end=" ")
        print()


timer = Timer()


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


g_log_time = defaultdict(list)


def log_time(log_text=None):
    def decorator(func):
        def wrapper(*args, **kws):
            start = datetime.datetime.now()
            result = func(*args, **kws)
            end = datetime.datetime.now()
            time = (end - start).total_seconds()
            log_key = log_text or func.__name__
            g_log_time[log_key].append(time)
            return result

        return wrapper

    return decorator


def log_time_func(text, end=False):
    now = datetime.datetime.now()
    if (
        text in g_log_time
        and len(g_log_time[text]) > 0
        and isinstance(g_log_time[text][-1], datetime.datetime)
    ):
        start = g_log_time[text][-1]
        t = (now - start).total_seconds()
        g_log_time[text][-1] = t
    if not end:
        g_log_time[text].append(datetime.datetime.now())


def print_time(print_directly=True):
    info = []
    for item in g_log_time.items():
        if len(item) <= 1 or len(item[1]) == 0 or len(item[0]) == 0:
            continue
        mean = np.mean(item[1])
        max = np.max(item[1])
        info.append(
            "{} | mean:{:.3f}ms, max:{:.3f}ms, times:{}".format(
                item[0], mean * 1000, max * 1000, len(item[1])
            )
        )
        g_log_time[item[0]] = []
    info = "\n".join(info)
    if print_directly:
        print(info)
    else:
        return info


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

if __name__ == "__main__":
    test_play_against_random()

