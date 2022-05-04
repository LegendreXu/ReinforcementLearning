# -*- coding: utf-8 -*-
"""
Created on Mon May  2 18:07:37 2022

@author: 13732
"""

import os
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class ReplayBuffer(object):
    
    ''' Create a replay buffer to store transition history and sample transition for traninig'''
    
    def __init__(self, max_size, input_shape):
        ## max_size: the max size of the buffer
        ## input_shape: the input(state) dimensions
        self.memory_size = max_size
        self.memory_counter = 0
        self.state_memory = np.zeros((self.memory_size, *input_shape))
        self.next_state_memory = np.zeros((self.memory_size, *input_shape))
        self.action_memory = np.zeros(self.memory_size)
        self.reward_memory = np.zeros(self.memory_size)
        self.terminal_memory = np.zeros(self.memory_size, dtype = np.float32)
        
    def store_transition(self, state, action, reward, next_state, done):
        index = self.memory_counter % self.memory_size ## use % to get the recent memory
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward    
        self.next_state_memory[index] = next_state
        self.terminal_memory[index] = 1 - done
        self.memory_counter += 1
        
    def sample_buffer(self, batch_size):
        max_memory = min(self.memory_counter, self.memory_size)
        batch = np.random.choice(max_memory, batch_size)
        
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminals = self.terminal_memory[batch]
        next_states = self.next_state_memory[batch]
        
        return states, actions, rewards, next_states, terminals
    
class Net(nn.Module):
    '''  construct a critic network here for state valuation'''
    def __init__(self, LR, input_dims, fc1_dims, fc2_dims, n_actions):
        super(Net, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        
        self.action_value = nn.Linear(fc2_dims, self.n_actions) ## genearte state_action value
        
        
        self.optimizer = optim.Adam(self.parameters(), lr = LR)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.to(self.device)
    
    def forward(self, state):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        
        actions_value = self.action_value(F.relu(state_value))
        
        return actions_value

class DQNAgent(object):
    ''' our final agent'''
    def __init__(self, LR, input_dims, tau, n_actions, gamma = 0.95,
        max_size = 1000000, fc1_dims = 400, fc2_dims = 300, batch_size = 64, epsilon = 0.9):
        
        ## LR: learning rate for the Critic Network
        ## input_dims: state dimension ( 2 in our hedging problems)
        ## tau: updating parameters
        ## gamma: discount reward rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims)
        self.batch_size = batch_size
        
        self.policy_Q = Net(LR, input_dims, fc1_dims, fc2_dims, n_actions)
        self.target_Q = Net(LR, input_dims, fc1_dims, fc2_dims, n_actions)
        
        self.update_network_parameters(tau = 1)
    
    def epsilon_decay(self):
        self.epsilon *= 0.8
    
    def choose_action(self, observation):
        ## choose action given observation
        if np.random.uniform() < 1 - self.epsilon:        
            self.policy_Q.eval()
            observation = torch.tensor(observation, dtype = torch.float).to(self.policy_Q.device)                             
            actions_value = self.policy_Q.forward(observation)   
            self.policy_Q.train()                         
            #action = torch.max(actions_value, 1)[1].data.numpy()                
            #action = action[0]       
            action = torch.max(actions_value, 0)[1].cpu().detach().numpy()           
            action = int(action)   
        else:
            action = np.random.randint(0, self.policy_Q.n_actions)
        return action
    
    def final_action(self, observation):
        ## choose action given observation. Without exploration!
        
        self.policy_Q.eval()
        observation = torch.tensor(observation, dtype = torch.float).to(self.policy_Q.device)                             
        actions_value = self.policy_Q.forward(observation)   
        self.policy_Q.train()                         
        action = torch.max(actions_value, 0)[1].cpu().detach().numpy()                  
        
        return action
        
    
    def remember(self, state, action, reward, next_state, done):
        ## store transition
        self.memory.store_transition(state, action, reward, next_state, done)
    
    def learn(self):
        if self.memory.memory_counter < self.batch_size:
            ## Only learn when the size of memory buffer is larger than batch size
            return
        
        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)
        reward = torch.tensor(reward, dtype=torch.float).to(self.policy_Q.device)
        done = torch.tensor(done).to(self.policy_Q.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.policy_Q.device)
        state = torch.tensor(state, dtype=torch.float).to(self.policy_Q.device)
        
        ## Since the action is in 1 dimension here, we change it to matrix representation
        action = [action]
        action = torch.tensor(action, dtype=torch.int64).to(self.policy_Q.device)
        action = action.T
        
        self.policy_Q.eval()
        self.target_Q.eval()
        
        q_policy = self.policy_Q(state).gather(1, action)
        q_next = self.target_Q(state).detach()
        q_target = reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        
        self.policy_Q.optimizer.zero_grad()
        loss = F.mse_loss(q_policy, q_target)
        loss.backward()                                                 
        self.policy_Q.optimizer.step()       
        
        self.update_network_parameters()
    
        
    def update_network_parameters(self, tau=None):
        ## update network with decay tau
        if tau is None:
            tau = self.tau
            
        policy_params = self.policy_Q.named_parameters()
        target_params = self.target_Q.named_parameters()
        policy_dict = dict(policy_params)
        target_dict = dict(target_params)
        
        
        for name in policy_dict:
            policy_dict[name] = tau*policy_dict[name].clone() + (1-tau)*target_dict[name].clone()
        
        self.target_Q.load_state_dict(policy_dict)
        
