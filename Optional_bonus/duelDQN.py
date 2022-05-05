# -*- coding: utf-8 -*-
"""
Created on Thu May  5 23:51:18 2022

@author: George Chen
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

class ReplayBuffer():
    
    def __init__(self, memory_size, state_dims, action_dims):
        self.memory_size = memory_size
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.memory_encounter = 0 ##calculate the number of encountered transitions
        self.state_memory = np.zeros((self.memory_size, state_dims),
                                    dtype=np.float32)
        self.next_state_memory = np.zeros((self.memory_size, state_dims),
                                        dtype=np.float32)
        self.action_memory = np.zeros(self.memory_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.done_memory = np.zeros(self.memory_size, dtype=bool)
    
    def store_transition(self, state, action, reward, next_state, done):
        store_index = self.memory_encounter % self.memory_size
        self.state_memory[store_index] = state
        self.next_state_memory[store_index] = next_state
        self.reward_memory[store_index] = reward
        self.action_memory[store_index] = action
        self.done_memory[store_index] = done
        
        self.memory_encounter += 1 ## remember to increase memory encounter after storing 
        
        
    def sample(self, batch_size):
        
        maxi_size = min(self.memory_size, self.memory_encounter)
        sample_batch = np.random.choice(maxi_size, batch_size, replace=False)
        
        states = self.state_memory[sample_batch]
        actions = self.action_memory[sample_batch]
        rewards = self.reward_memory[sample_batch]
        next_states = self.next_state_memory[sample_batch]
        dones = self.done_memory[sample_batch]

        return states, actions, rewards, next_states, dones
                
class Net(nn.Module):
    def __init__(self, lr, state_dims, action_dims, name):
        super(Net, self).__init__()
        self.lr = lr
        self.state_dims = state_dims
        self.action_dims = action_dims
        
        self.fc1 = nn.Linear(state_dims, 256)
        self.V_func = nn.Linear(256, 1)
        self.A_func = nn.Linear(256, self.action_dims)
        
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.checkpoint_file = os.path.join('./', name + '_DuelDQN')
        
    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        
        A = self.A_func(x)
        V = self.V_func(x)
        
        return V, A
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
    
class DDQN_Agent():
    def __init__(self, state_dims, action_dims, lr, tau, gamma=0.99, epsilon=1.0, eps_min = 0.01, eps_dec = 1e-3, batch_size = 64, memory_size = 100000):
        self.policy_Q = Net(lr, state_dims, action_dims, name = 'policy_network')
        self.target_Q = Net(lr, state_dims, action_dims, name = 'target_network')
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min =  eps_min
        self.eps_dec = eps_dec
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(memory_size, state_dims, action_dims)
        self.tau = tau
        self.replace_target = 100
        self.action_space = [i for i in range(action_dims)]
        self.learn_step_counter = 0
        
    def update_network_parameters(self):
       ## update network with decay tau
           
       policy_params = self.policy_Q.named_parameters()
       target_params = self.target_Q.named_parameters()

       policy_dict = dict(policy_params)
       target_dict = dict(target_params)

       for name in policy_dict:
           policy_dict[name] = self.tau*policy_dict[name].clone() + (1-self.tau)*target_dict[name].clone()
       
       self.target_Q.load_state_dict(policy_dict)
    
    def update_target_network(self):
        if self.learn_step_counter % self.replace_target == 0:
            self.target_Q.load_state_dict(self.policy_Q.state_dict())   
    
    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            state = torch.tensor([state],dtype=torch.float).to(self.policy_Q.device)
            v, A = self.policy_Q.forward(state)
            action = torch.argmax(A).item()
        else:
            action = np.random.choice(self.action_space)

        return action
    
    def final_action(self, state):
        state = torch.tensor([state],dtype=torch.float).to(self.policy_Q.device)
        V, A = self.policy_Q.forward(state)
        action = torch.argmax(A).item()

        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.store_transition(state, action, reward, next_state, done)
        
    def decrement_epsilon(self):
        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon - self.eps_dec     
        else:
            self.epsilon = self.eps_min
    
    
            
    def learn(self):
        if self.batch_size > self.buffer.memory_encounter:
            return
        
        #self.update_target_network()
        self.policy_Q.optimizer.zero_grad()

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
                                

        states = torch.tensor(states).to(self.policy_Q.device)
        rewards = torch.tensor(rewards).to(self.policy_Q.device)
        dones = torch.tensor(dones).to(self.policy_Q.device)
        next_states = torch.tensor(next_states).to(self.policy_Q.device)
        actions = torch.tensor(actions).to(self.policy_Q.device)

        indices = np.arange(self.batch_size)

        V_s, A_s = self.policy_Q.forward(states)
        V_s_next, A_s_next = self.target_Q.forward(next_states)

        V_s_policy, A_s_policy = self.policy_Q.forward(next_states)

        Q_pred = torch.add(V_s,
                        (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        Q_next = torch.add(V_s_next,
                        (A_s_next - A_s_next.mean(dim=1, keepdim=True)))

        Q_eval = torch.add(V_s_policy, (A_s_policy - A_s_policy.mean(dim=1,keepdim=True)))

        argmax_actions = torch.argmax(Q_eval, dim=1)

        Q_next[dones] = 0.0
        Q_target = rewards + self.gamma*Q_next[indices, argmax_actions]

        loss = self.policy_Q.loss(Q_target, Q_pred).to(self.policy_Q.device)
        loss.backward()
        self.policy_Q.optimizer.step()
        
        self.decrement_epsilon()
        
        self.update_network_parameters()
        self.learn_step_counter += 1
    
    
    def save_models(self):
        print('... saving best models ...')
        self.policy_Q.save_checkpoint()
        self.target_Q.save_checkpoint()

    def load_models(self):
        print('... loading best models ...')
        self.policy_Q.load_checkpoint()
        self.target_Q.load_checkpoint()

   
       
       

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        