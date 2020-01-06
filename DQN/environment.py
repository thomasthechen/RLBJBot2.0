import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import torch
from torch.autograd import Variable

def cmp(a, b):
    return int((a > b)) - int((a < b))

# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10 
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def draw_card(np_random):
    return np_random.choice(deck)


def draw_hand(np_random):
    return [draw_card(np_random), draw_card(np_random)]


def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21

def doubleable_hand(hand):
    return 1 if len(hand) == 2 else 0


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
            return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]

# TODO: There's so much more that needs to be implemented, might want to scrap this class and recreate a new blackjack class
# TODO: Use gym.spaces.Dict to make things more clear in the observation space
# TODO: Here's another implementation: https://github.com/nalkpas/CS230-2018-Project/blob/master/BlackjackSM.py
class BlackjackEnv(gym.Env):
    """
    """
    def __init__(self, natural=False):
        self.double = False
        self.action_space = spaces.Discrete(3)
        self.action_names = ["STAND", "HIT", "DOUBLE"]
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32), 
            spaces.Discrete(11),
            spaces.Discrete(2),
            spaces.Discrete(2)))
        self._seed()

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural
        # Start the first game
        self._reset()        # Number of 
        self.nA = 3

    def reset(self):
        return self._reset()

    def step(self, action):
        return self._step(action)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getActions(self):
        #print(int(doubleable_hand(self.player)))
        #print(" ")
        return torch.tensor([1, 1,  int(doubleable_hand(self.player))]).float()



    # TODO: Implement full 5-action space
    def _step(self, action):
        assert self.action_space.contains(action)

        #print("step")
        if action == 1:  # hit: add a card to players hand and return
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                done = True
                reward = -1
            else:
                done = False
                reward = 0
        
        if action == 2:
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                reward = -2
            done = True

            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))
            reward = 2 * cmp(score(self.player), score(self.dealer))
        
        
        if action == 0:  # stand: play out the dealers hand, and score
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))
            reward = cmp(score(self.player), score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 1:
                reward = 1.5
        
        dropout = self.getActions()
        state = self._get_obs()
        #print(state)
        #print(dropout)
        return state, reward, done, {}, dropout

    def _get_obs(self):
        #print(int(doubleable_hand(self.player)), "str")
        return torch.tensor([sum_hand(self.player), self.dealer[0], int(usable_ace(self.player)), int(doubleable_hand(self.player))]).float()

    def _reset(self):
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)

        # Auto-draw another card if the score is less than 12
        #while sum_hand(self.player) < 12:
            #self.player.append(draw_card(self.np_random))

        return self._get_obs(), self.getActions()
  
  
  