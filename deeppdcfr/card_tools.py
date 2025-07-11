import numpy as np
from pathlib import Path


class CardTools:
    ranks = [str(n) for n in range(2, 10)] + list("TJQKA")
    suits = list("cdhs")

    def __init__(self):
        file = Path(__file__).parents[1].absolute() / "matrix" / "texas_lookup.npy"
        self.hand_rank = np.load(file)
        self._card_to_id = {}
        self._id_to_card = []
        counter = 0
        for rank in self.ranks:
            for suit in self.suits:
                card_str = rank + suit
                self._id_to_card.append(card_str)
                self._card_to_id[card_str] = counter
                counter += 1
        self._hand_to_id = {}
        self._id_to_hand = []
        self.hand_ids = np.zeros([1326, 2])
        hand_counter = 0
        for card1 in range(counter):
            for card2 in range(card1 + 1, counter):
                self.hand_ids[hand_counter][0] = card1
                self.hand_ids[hand_counter][1] = card2
                hand_str = (self._id_to_card[card1], self._id_to_card[card2])
                hand_str2 = (self._id_to_card[card2], self._id_to_card[card1])
                self._id_to_hand.append(hand_str)
                self._hand_to_id[hand_str] = hand_counter
                self._hand_to_id[hand_str2] = hand_counter
                hand_counter += 1

    def card_to_id(self, card):
        return self._card_to_id[card]

    def id_to_card(self, cid):
        return self._id_to_card[cid]

    def hand_to_id(self, hand):
        return self._hand_to_id[tuple(hand)]

    def id_to_hand(self, hid):
        return self._id_to_hand[hid]

    def evaluate(self, hands):
        rank = self.hand_rank[hands[:, 0] + 54]
        for c in range(1, hands.shape[1]):
            rank = self.hand_rank[hands[:, c] + rank + 1]
        return rank


card_tools = CardTools()
