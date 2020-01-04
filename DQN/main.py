import gym
from dqn2 import Agent
#from utils import plotLearning
import numpy as np
from gym import wrappers
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make('Blackjack-v0')
    brain = Agent(gamma=0.99, epsilon = 1.0, alpha = 0.0003, batch_size = 64, n_actions = 2, input_dims = [3])

    scores = []
    avgscore = []
    eps_history = []
    n_games = 2000
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
        observation = env.reset()
        done = False

        while not done:
            action = brain.chooseAction(observation)
            observation_, reward, done, info = env.step(action)
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

    
    plt.plot(scores)
    plt.plot(avgscore)
    plt.ylabel('Scores')
    plt.show()