import gym
from dqn2 import Agent
#from utils import plotLearning
import numpy as np
from gym import wrappers
import matplotlib.pyplot as plt
from environment import *
import torch as T

if __name__ == '__main__':
    #env = #gym.make('Blackjack-v0')
    env = BlackjackEnv()
    brain = Agent(gamma=0.99, epsilon = 1.0, alpha = 0.0003, batch_size = 64, n_actions = 3, input_dims = [4])

    scores = []
    avgscore = []
    eps_history = []
    n_games = 1500
    score = 0
    avg_score = -1

    
    for i in range(n_games):
        
        if i % 20 == 0 and i > 0:
            avg_score = np.mean(scores[max(0, i-20):(i+1)])
            
            print('episode', i, 'score', score,
            'average score %.3f' % avg_score,
            'epsilon %.3f' % brain.EPSILON)
        else:
            print('episode', i, 'score', score)
        
        score = 0
        eps_history.append(brain.EPSILON)
        observation, dropout = env.reset()
        done = False
        

        while not done:
            action = brain.chooseAction(observation, dropout)
            observation_, reward, done, info, dropout = env.step(action)
            score += reward
            brain.storeTransition(observation, action, reward, observation_,
            done)
            brain.learn()
            observation = observation_

        scores.append(score)
        avgscore.append(avg_score)

    x = [i+1 for i in range(n_games)]
    filename = 'lunar-lander.png'
    #plotLearning(x, scores, eps_history, filename)

    for j in range(1,12):
        for i in range(2,22): 
            print(brain.Q_eval(T.tensor([j, i, 0, 1]).float())[T.argmax(brain.Q_eval(T.tensor([j, i, 0, 1]).float())).item()], end =" ")
        print("")

    plt.plot(scores)
    plt.plot(avgscore)
    plt.ylabel('Scores')
    plt.show()