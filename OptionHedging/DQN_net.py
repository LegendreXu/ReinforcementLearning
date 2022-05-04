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
        
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(2, 50)
        self.linear.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, ACTION_NUMS)
        self.out.weight.data.normal(0, 0.1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        actions_value = self.out(x)

class DQN(object):
    def __init__(self):                                                         # 定义DQN的一系列属性
        self.eval_net, self.target_net = Net(), Net()                           # 利用Net创建两个神经网络: 评估网络和目标网络
        self.learn_step_counter = 0                                             # for target updating
        self.memory_counter = 0                                                 # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))             # 初始化记忆库，一行代表一个transition
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)    # 使用Adam优化器 (输入为评估网络的参数和学习率)
        self.loss_func = nn.MSELoss()                                           # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)

    def choose_action(self, x):                                                 # 定义动作选择函数 (x为状态)
        x = torch.unsqueeze(torch.FloatTensor(x), 0)                            # 将x转换成32-bit floating point形式，并在dim=0增加维数为1的维度
        if np.random.uniform() < EPSILON:                                       # 生成一个在[0, 1)内的随机数，如果小于EPSILON，选择最优动作
            actions_value = self.eval_net.forward(x)                            # 通过对评估网络输入状态x，前向传播获得动作值
            action = torch.max(actions_value, 1)[1].data.numpy()                # 输出每一行最大值的索引，并转化为numpy ndarray形式
            action = action[0]                                                  # 输出action的第一个数
        else:                                                                   # 随机选择动作
            action = np.random.randint(0, N_ACTIONS)                            # 这里action随机等于0或1 (N_ACTIONS = 2)
        return action                                                           # 返回选择的动作 (0或1)

    def store_transition(self, s, a, r, s_):                                    # 定义记忆存储函数 (这里输入为一个transition)
        transition = np.hstack((s, [a, r], s_))                                 # 在水平方向上拼接数组
        # 如果记忆库满了，便覆盖旧的数据
        index = self.memory_counter % MEMORY_CAPACITY                           # 获取transition要置入的行数
        self.memory[index, :] = transition                                      # 置入transition
        self.memory_counter += 1                                                # memory_counter自加1

    def learn(self):                                                            # 定义学习函数(记忆库已满后便开始学习)
        # 目标网络参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:                  # 一开始触发，然后每100步触发
            self.target_net.load_state_dict(self.eval_net.state_dict())         # 将评估网络的参数赋给目标网络
        self.learn_step_counter += 1                                            # 学习步数自加1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)            # 在[0, 2000)内随机抽取32个数，可能会重复
        b_memory = self.memory[sample_index, :]                                 # 抽取32个索引对应的32个transition，存入b_memory
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        # 将32个s抽出，转为32-bit floating point形式，并存储到b_s中，b_s为32行4列
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        # 将32个a抽出，转为64-bit integer (signed)形式，并存储到b_a中 (之所以为LongTensor类型，是为了方便后面torch.gather的使用)，b_a为32行1列
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        # 将32个r抽出，转为32-bit floating point形式，并存储到b_s中，b_r为32行1列
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
        # 将32个s_抽出，转为32-bit floating point形式，并存储到b_s中，b_s_为32行4列

        # 获取32个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
        q_eval = self.eval_net(b_s).gather(1, b_a)
        # eval_net(b_s)通过评估网络输出32行每个b_s对应的一系列动作值，然后.gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
        q_next = self.target_net(b_s_).detach()
        # q_next不进行反向传递误差，所以detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(BATCH_SIZE, 1)的形状；最终通过公式得到目标值
        loss = self.loss_func(q_eval, q_target)
        # 输入32个评估值和32个目标值，使用均方损失函数
        self.optimizer.zero_grad()                                      # 清空上一步的残余更新参数值
        loss.backward()                                                 # 误差反向传播, 计算参数更新值
        self.optimizer.step()       
'''