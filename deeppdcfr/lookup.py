from pathlib import Path
import numpy as np
from deeppdcfr.card_tools import card_tools


class Lookup:
    def __init__(self):
        file = Path(__file__).parents[1].absolute() / "matrix" / "random_board_wp.npy"
        self.random_board_wp = np.load(str(file))

        file = Path(__file__).parents[1].absolute() / "matrix" / "flop_matrix.npy"
        self.flop_matrix = np.load(str(file))

    def check(self, a, b):
        if b == None or a == None:
            return False
        for x in a:
            if x in b:
                return True
        return False

    def calc(self, hand1, board, opponent_range=None):
        if opponent_range is None:
            opponent_range = self.random_init_range([*hand1, *board])
        if self.check(hand1, board):
            raise Exception("same card!")
        if len(board) == 0:
            return self.calc2(hand1, board, opponent_range)
        elif len(board) == 3:
            return self.calc3(hand1, board, opponent_range)
        else:
            return self.calc1(hand1, board, opponent_range)

    def find(self, hand1, hand2, board=None):
        if board == []:
            board = None
        if (
            self.check(hand1, hand2)
            or self.check(hand1, board)
            or self.check(hand2, board)
        ):
            raise Exception("same card!")
        wp = self.random_board_wp[
            card_tools.hand_to_id(hand1), card_tools.hand_to_id(hand2)
        ]
        return wp

    def calc3(self, hand1, board, opponent_range):
        hand_ids = [card_tools.card_to_id(card) for card in hand1]
        board_ids = [card_tools.card_to_id(card) for card in board]
        hand_ids.sort()
        board_ids.sort()
        return self.flop_matrix[
            board_ids[0], board_ids[1], board_ids[2], hand_ids[0], hand_ids[1]
        ]

    def calc2(self, hand1, board, opponent_range):
        wp = 0
        for hid in range(1326):
            if opponent_range[hid] == 0:
                continue
            opponent_card = card_tools.id_to_hand(hid)
            wp += self.find(hand1, opponent_card, board) * opponent_range[hid]
        return wp

    def calc1(self, hand1, board, opponent_range):
        leave_cards = []
        for i in range(52):
            if (
                not card_tools.id_to_card(i) in hand1
                and not card_tools.id_to_card(i) in board
            ):
                leave_cards.append(i)
        self.outcome = 0
        self.enumerate_leave_public_card(-1, leave_cards, hand1, board, opponent_range)
        total = 1
        for i in range(5 - len(board)):
            total *= len(leave_cards) - i
        for i in range(5 - len(board)):
            total /= i + 1
        self.outcome /= total
        return self.outcome

    def enumerate_leave_public_card(
        self, start_index, leave_cards, hand, board, opponent_range
    ):
        if len(board) == 5:
            self.calc_win_probablity(hand, board, opponent_range)
            return
        for i in range(start_index + 1, len(leave_cards)):
            board.append(card_tools.id_to_card(leave_cards[i]))
            self.enumerate_leave_public_card(
                i, leave_cards, hand, board, opponent_range
            )
            board.pop()

    def calc_win_probablity(self, private_card, public_card, opponent_range):
        new_opponent_range = opponent_range.copy()
        for card in public_card:
            for card2_id in range(52):
                card2 = card_tools.id_to_card(card2_id)
                if card == card2:
                    continue
                new_opponent_range[card_tools.hand_to_id((card, card2))] = 0
        new_opponent_range /= sum(new_opponent_range)
        hands = np.zeros([1326, 7], dtype=int)
        board = np.array(
            [card_tools.card_to_id(card) for card in public_card], dtype=int
        )
        hands[:, :5] = np.repeat(board.reshape([1, 5]), 1326, axis=0)
        hands[:, 5:7] = card_tools.hand_ids
        hands_strength = card_tools.evaluate(hands)
        lbr_hand = np.array(
            [[card_tools.card_to_id(card) for card in (*public_card, *private_card)]],
            dtype=int,
        )
        lbr_strength = card_tools.evaluate(lbr_hand)
        result = (lbr_strength > hands_strength).astype(float) + (
            lbr_strength == hands_strength
        ).astype(float) * 0.5
        np.set_printoptions(threshold=1e6)
        self.outcome += np.dot(result, new_opponent_range)

    def random_init_range(self, cards):
        opponent_range = np.array([1 for i in range(1326)], dtype=float)
        for card in cards:
            for card2_id in range(52):
                card2 = card_tools.id_to_card(card2_id)
                if card == card2:
                    continue
                opponent_range[card_tools.hand_to_id((card, card2))] = 0
        opponent_range /= np.sum(opponent_range)
        return opponent_range


lookup = Lookup()
