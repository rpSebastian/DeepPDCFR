from pathlib import Path

import pyspiel
import tabulate

from deeppdcfr.utils import SpielGame, SpielState
import multiprocessing


class GameConfig:
    def __init__(
        self,
        game_name: str,
        transform: bool = False,
        large_game: bool = False,
        poker: bool = False,
    ):
        self.game_name = game_name
        self.params = {}
        self.transform = transform
        self.large_game = large_game
        self.poker = poker
        self.name = self.__class__.__name__

    def load_game(self) -> SpielGame:
        params = {}
        for p, v in self.params.items():
            if p == "filename":
                v = str(Path(__file__).absolute().parents[2] / v)
            params[p] = v
        game = pyspiel.load_game(self.game_name, params)
        if self.transform:
            game = pyspiel.convert_to_turn_based(game)
        return game

    def get_draw_file(self) -> Path:
        file = (
            Path(__file__).absolute().parents[1]
            / "images"
            / "VIS_{}.pdf".format(self.name)
        )
        file.parent.mkdir(parents=True, exist_ok=True)
        return file

    def __repr__(self):
        return self.name

    def get_size(self) -> list[int]:
        self.num_nodes = 0
        self.infostate_set = set()
        self.depth = 0
        self.terminal_nodes = 0
        game = self.load_game()
        self.calc_nodes(game.new_initial_state(), 1)
        return [
            self.num_nodes,
            len(self.infostate_set),
            self.depth,
            self.terminal_nodes,
        ]

    def calc_nodes(self, h: SpielState, depth: int):
        self.num_nodes += 1
        if self.num_nodes % 1000000 == 0:
            print(
                self.num_nodes, len(self.infostate_set), self.depth, self.terminal_nodes
            )
        if h.is_player_node():
            s_str = h.information_state_string()
            self.infostate_set.add(s_str)
        if h.is_terminal():
            self.terminal_nodes += 1
            self.depth = max(self.depth, depth)
            return
        for a in h.legal_actions():
            self.calc_nodes(h.child(a), depth + 1)

    def get_holdem_size(self) -> list[int]:
        game = self.load_game()
        h = game.new_initial_state()
        num_nodes, infostate_set, max_depth, terminal_nodes = 1, set(), 0, 0
        pool = multiprocessing.Pool(150, maxtasksperchild=1)
        tasks = []
        for c1 in (h.child(a) for a in h.legal_actions()):
            num_nodes += 1
            for c2 in (c1.child(a) for a in c1.legal_actions()):
                num_nodes += 1
                for c3 in (c2.child(a) for a in c2.legal_actions()):
                    num_nodes += 1
                    for c4 in (c3.child(a) for a in c3.legal_actions()):
                        tasks.append(pool.apply_async(self.calc_holdem_nodes, (c4,)))
        cnt = 0
        for task in tasks:
            c_num_nodes, c_infostae_set, c_max_depth, c_terminal_nodes = task.get()
            cnt += 1
            num_nodes += c_num_nodes
            infostate_set.update(c_infostae_set)
            max_depth = max(max_depth, c_max_depth)
            terminal_nodes += c_terminal_nodes
            print(
                cnt,
                len(tasks),
                [num_nodes, len(infostate_set), max_depth + 4, terminal_nodes],
            )
            del c_infostae_set
        return [num_nodes, len(infostate_set), max_depth + 4, terminal_nodes]

    def calc_holdem_nodes(self, h):
        num_nodes = 0
        infostate_set = set()
        max_depth = 0
        terminal_nodes = 0

        def calc_nodes(h, depth):
            nonlocal num_nodes, infostate_set, max_depth, terminal_nodes
            num_nodes += 1
            if h.is_player_node():
                s_str = h.information_state_string()
                infostate_set.add(s_str)
            if h.is_terminal():
                terminal_nodes += 1
                max_depth = max(max_depth, depth)
                return
            for a in h.legal_actions():
                calc_nodes(h.child(a), depth + 1)

        calc_nodes(h, 1)

        return [num_nodes, infostate_set, max_depth, terminal_nodes]

    def visulize(self):
        from open_spiel.python.visualizations import treeviz

        game = self.load_game()
        gametree = treeviz.GameTree(
            game,
            node_decorator=_zero_sum_node_decorator,
            group_infosets=True,
            group_terminal=False,
            group_pubsets=False,
            target_pubset="*",
        )
        gametree.draw(str(self.get_draw_file()), prog="dot")


def _zero_sum_node_decorator(state):
    """Custom node decorator that only shows the return of the first player."""
    from open_spiel.python.visualizations import treeviz

    attrs = treeviz.default_node_decorator(state)  # get default attributes
    if state.is_terminal():
        attrs["label"] = str(int(state.returns()[0]))
    return attrs


class KuhnPoker(GameConfig):
    def __init__(self):
        super().__init__(
            game_name="kuhn_poker",
            poker=True,
        )


class BattleShip(GameConfig):
    def __init__(
        self,
        board_width=2,
        board_height=2,
        num_shots=2,
    ):
        super().__init__(
            game_name="battleship",
        )
        self.params = {
            "board_width": board_width,
            "board_height": board_height,
            "ship_sizes": "[2]",
            "ship_values": "[2]",
            "num_shots": num_shots,
            "allow_repeated_shots": False,
        }
        self.name = "Battleship_{}{}_{}".format(board_width, board_height, num_shots)


class LeducPoker(GameConfig):
    def __init__(self):
        super().__init__(
            game_name="leduc_poker",
            poker=True,
        )


class GoofSpiel(GameConfig):
    def __init__(
        self,
        num_cards=3,
        imp_info=False,
        points_order="descending",
        returns_type="win_loss",
    ):
        super().__init__(
            game_name="goofspiel",
            transform=True,
        )
        self.params = {
            "num_cards": num_cards,
            "imp_info": imp_info,
            "points_order": points_order,
            "returns_type": returns_type,
        }
        self.name = "GoofSpiel"
        if imp_info:
            self.name += "Imp"
        self.name += str(num_cards)


class LiarsDice(GameConfig):
    def __init__(
        self,
        dice_sides=3,
        num_dice=1,
    ):
        super().__init__(
            game_name="liars_dice",
        )
        self.params = {"numdice": num_dice, "dice_sides": dice_sides}
        self.name = "LiarsDice{}".format(dice_sides)


class UniversalPoker(GameConfig):
    def __init__(
        self,
        num_players=2,
        betting="limit",
        blind="1 1",
        raise_size="1",
        first_player="1",
        max_raises="1",
        num_rounds=1,
        num_suits=1,
        num_ranks=4,
        num_hole_cards=1,
        num_board_cards="0",
        stack=None,
        betting_abstraction=None,
        pot_size=None,
        board_cards=None,
        hand_reaches=None,
        large_game=False,
    ):
        super().__init__(
            game_name="universal_poker",
            large_game=large_game,
            poker=True,
        )
        self.params = {
            "betting": betting,
            "bettingAbstraction": betting_abstraction,
            "blind": blind,
            "boardCards": board_cards,
            "firstPlayer": first_player,
            "handReaches": hand_reaches,
            "maxRaises": max_raises,
            "numBoardCards": num_board_cards,
            "numHoleCards": num_hole_cards,
            "numPlayers": num_players,
            "numRanks": num_ranks,
            "numRounds": num_rounds,
            "numSuits": num_suits,
            "potSize": pot_size,
            "raiseSize": raise_size,
            "stack": stack,
        }
        for key, value in list(self.params.items()):
            if value is None:
                del self.params[key]


class FHP(UniversalPoker):
    def __init__(self):
        super().__init__(
            num_players=2,
            betting="limit",
            blind="50 100",
            raise_size="100 100",
            first_player="1 2",
            max_raises="3 3",
            num_rounds=2,
            num_suits=4,
            num_ranks=13,
            num_hole_cards=2,
            num_board_cards="0 3",
            large_game=True,
        )
        self.name = "FHP"


class HULH(UniversalPoker):
    def __init__(self):
        super().__init__(
            num_players=2,
            betting="limit",
            blind="50 100",
            raise_size="100 100 200 200",
            first_player="1 2 2 2",
            max_raises="3 3 4 4",
            num_rounds=4,
            num_suits=4,
            num_ranks=13,
            num_hole_cards=2,
            num_board_cards="0 3 1 1",
            large_game=True,
        )


def get_game_configs() -> list[GameConfig]:
    game_configs = [
        KuhnPoker(),
        LeducPoker(),
        *[GoofSpiel(num_cards=cards) for cards in [5, 6]],
        *[GoofSpiel(num_cards=cards, imp_info=True) for cards in [5, 6]],
        *[LiarsDice(dice_sides=sides, num_dice=1) for sides in [5, 6]],
        BattleShip(board_width=3, board_height=2, num_shots=3),
        BattleShip(board_width=2, board_height=2, num_shots=3),
        FHP(),        
        HULH(),
    ]
    return game_configs


def read_game_config(game_name) -> GameConfig:
    game_configs = get_game_configs()
    game_dict = {game_config.name: game_config for game_config in game_configs}
    game_config = game_dict[game_name]
    return game_config


def print_game_info(game_config, size=False, visulize=False):
    game = game_config.load_game()
    try:
        observation_tensor_size = game.observation_tensor_size()
    except Exception:
        observation_tensor_size = None

    headers = ["Game Config"]
    row = [game_config]

    if size:
        if "FHP" in game_config.name or "HULH" in game_config.name:
            row += game_config.get_holdem_size()
        else:
            row += game_config.get_size()
        headers += ["Num Nodes", "Num InfoState Size", "Depth", "Num Terminals"]

    if visulize:
        game_config.visulize()

    return row, headers

def print_game():
    game_names = [
        "KuhnPoker",
        "LeducPoker",
        "LiarsDice5",
        "LiarsDice6",
        "GoofSpiel5",
        "GoofSpiel6",
        "GoofSpielImp5",
        "GoofSpielImp6",
        "Battleship_22_3",
        "Battleship_32_3",
        # "FHP",
        # "HULH",
    ]
    table = []
    for game_name in game_names:
        print(game_name)
        game_config = read_game_config(game_name)
        row, headers = print_game_info(game_config, size=True, visulize=False)
        table.append(row)
    print(tabulate.tabulate(table, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    print_game()
    