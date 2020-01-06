import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, ALPHA, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)
        self.loss = nn.MSELoss()
   

    def forward(self, observation):
        state = T.Tensor(observation)
        #observation = observation.view(-1)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions

class Agent(object):
    def __init__(self, gamma, epsilon, alpha, input_dims, batch_size, n_actions,
                 max_mem_size=1000000, eps_end=0.001, eps_dec=0.996):
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.EPS_MIN = eps_end
        self.EPS_DEC = eps_dec
        self.ALPHA = alpha
        self.action_space = [i for i in range(n_actions)]
        #print (self.action_space)
        self.n_actions = n_actions
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.Q_eval = DeepQNetwork(alpha, n_actions=self.n_actions,
                              input_dims=input_dims, fc1_dims=256, fc2_dims=256)
        self.state_memory = np.zeros((self.mem_size, *input_dims))
        self.new_state_memory = np.zeros((self.mem_size, *input_dims))
        self.action_memory = np.zeros((self.mem_size, self.n_actions),
                                      dtype=np.uint8)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)

    def storeTransition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        actions = np.zeros(self.n_actions)
        actions[action] = 1.0
        self.action_memory[index] = actions
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = 1 - terminal
        self.mem_cntr += 1

    def chooseAction(self, observation, dropout):
        rand = np.random.random()
        actions = self.Q_eval.forward(observation) * dropout
        
        if rand > self.EPSILON:
            action = T.argmax(actions).item()
        else:
            X = np.ma.masked_equal(T.FloatTensor(self.action_space) * dropout,0)
            action = int( np.random.choice(X))
            
        return action

    def learn(self):
        if self.mem_cntr > self.batch_size:
            self.Q_eval.optimizer.zero_grad()

            max_mem = self.mem_cntr if self.mem_cntr < self.mem_size \
                                    else self.mem_size

            batch = np.random.choice(max_mem, self.batch_size)
            state_batch = self.state_memory[batch]
            action_batch = self.action_memory[batch]
            #print(action_batch)
            action_values = np.array(self.action_space, dtype=np.uint8)
            #print(action_values)
            action_indices = np.dot(action_batch, action_values)
            #print(action_indices)
            reward_batch = self.reward_memory[batch]
            new_state_batch = self.new_state_memory[batch]
            terminal_batch = self.terminal_memory[batch]

            reward_batch = T.Tensor(reward_batch)
            terminal_batch = T.Tensor(terminal_batch)

            q_eval = self.Q_eval.forward(state_batch)
            #q_target = self.Q_eval.forward(state_batch).to(self.Q_eval.device)
            q_target = q_eval.clone()
            q_next = self.Q_eval.forward(new_state_batch)

            batch_index = np.arange(self.batch_size, dtype=np.int32)
            #print(batch_index)
            #print(q_target)

            for i in range(self.batch_size):
                q_target[batch_index[i], action_indices[i]] = reward_batch[i] + (self.GAMMA*T.max(q_next, dim=1)[0]*terminal_batch)[i]
            #q_target[batch_index, action_indices] = reward_batch + \
                               # self.GAMMA*T.max(q_next, dim=1)[0]*terminal_batch

            self.EPSILON = self.EPSILON*self.EPS_DEC if self.EPSILON > \
                           self.EPS_MIN else self.EPS_MIN

            loss = self.Q_eval.loss(q_target, q_eval)
            loss.backward()
            self.Q_eval.optimizer.step()