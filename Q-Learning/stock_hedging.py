# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 16:39:49 2022

@author: George
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange

class hedging_strategy():
    
    def __init__(self, period, ALPHA=0.5, GAMMA=1):
        self.ALPHA = ALPHA # step size of Q-learning
        self.GAMMA = GAMMA # discount rate of Q-learning
        self.Actions = [0, 0.5, 1] # possible actions: 0: short 0 stocks, 0.5: short 0.5 stocks, 1: short 1 stocks
        
        self.up = 1.1 # the upside ratio of stock movement in binominal model
        self.down = - 1/self.up # the downside ratio of stock movement in binominal model
        self.period = period
        self.start_price = self.period-1 ## we use a np.array(2*T-1) to store the corresponding price, so the start price should be the T-1 element in the array
        
        
        ### Here we define state as the tuple (price(St), time(t))

    def choose_action(self, state, q_value, epsilon):
        # choose action based on epsilon-greedy algorithm. Here we take epsilon as input so that we can adjust epsilon according to different episodes
        if np.random.binomial(1, epsilon) == 1:
            ## exploration
            action = np.random.choice([0,1,2]) 
        else: 
            ## exploitation
            values_ = q_value[state[0], state[1], :]
            action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])
        return action
        
    def step(self, state, action):
        ## each step, input a state and action, return next state and reward
        movement = np.random.binomial(1, 0.5)
        
        if movement == 1:
            next_price = state[0] + 1
        elif movement == 0:
            next_price = state[0] - 1
        
        St = float(np.where(state[0]>self.start_price, self.up**(state[0]-self.start_price), self.down**(-state[0]+self.start_price))) # stock price at time t
        St_1 = float(np.where(next_price>state[0], St*self.up, St*self.down)) # stock price at time t+1
        
                   
        reward = (St_1 - St) + self.Actions[action]*(St - St_1) ## long_position(1) * (St+1- St) + short_position*(St-St+1)
        t = state[1]+1
        
        next_state = [next_price, t]
        return next_state, reward
    
    def q_learning(self, q_value, epsilon):
        price = self.period-1
        t = 0
        state = [price,t]
        rewards = 0
        
        while state[1] < self.period:
            #action = choose_action(state, q_value)
            action = self.choose_action(state, q_value, epsilon)
            next_state, reward = self.step(state, action)
            rewards += reward
            # Q-Learning update
            if state[1] < self.period-1:
                q_value[state[0],state[1], action] += self.ALPHA * (
                        reward + self.GAMMA * np.max(q_value[next_state[0],next_state[1],:]) -
                        q_value[state[0],state[1], action])
            else:
                ## if t == T-1, means the next state is the terminal state t=T. we set all Q(:, T, :) = 0)
                q_value[state[0],state[1], action] += self.ALPHA * (
                        reward + 0 -
                        q_value[state[0],state[1], action])
                
            state = next_state
        return q_value, rewards
    
    
    # print optimal policy
    def print_optimal_policy(self):
        q_value = self.q_value
        optimal_policy = []
        for i in range(0, np.shape(q_value)[0]):
            optimal_policy.append([])
            for j in range(0, np.shape(q_value)[1]):
                bestAction = np.argmax(q_value[i, j, :])
                if ~np.any(q_value[i, j, :]):
                    ## if q_value is all zero, it means this state(price, time) is not availabel in the binominal model
                    optimal_policy[-1].append('')
                elif bestAction == 0:
                    optimal_policy[-1].append('0')
                elif bestAction == 1:
                    optimal_policy[-1].append('-0.5')
                elif bestAction == 2:
                    optimal_policy[-1].append('-1')
        for row in optimal_policy:
            print(row)
            
    def train(self, episodes = 100000):
        T = self.period
        q_value = np.zeros((2*T-1, T , 3))  
        reward_record = []
        for j in trange(episodes):
            x = (episodes-1) * (1 - 0.1) / episodes
            epsilon = x/(x+j)
            q_value, rewards = self.q_learning(q_value, epsilon)  
            reward_record.append(rewards)
        self.q_value = q_value
        self.reward_record = np.array(reward_record)
        
    def plot_training(self):
        plt.plot(self.reward_record)
        plt.title('rewards against traning episodes')
        plt.xlabel('episodes')
        plt.ylabel('rewards')
    
    def get_q_value(self):
        return self.q_value
    
    def compare_results(self, runs = 1000):
        rewards_optimal = []
        rewards_0 = []
        rewards_half = []
        rewards_1 = []
        rewards_random = []
        
        reward_optimal, reward_0, reward_half, reward_1, reward_random = 0, 0, 0, 0, 0
        
        for i in range(runs):
            
            price = self.period-1
            t = 0
            state = [price,t]
            
            
            while state[1] < self.period:
                movement = np.random.binomial(1, 0.5)
                
                if movement == 1:
                    next_price = state[0] + 1
                elif movement == 0:
                    next_price = state[0] - 1
                
                St = float(np.where(state[0]>self.start_price, self.up**(state[0]-self.start_price), self.down**(-state[0]+self.start_price))) # stock price at time t
                St_1 = float(np.where(next_price>state[0], St*self.up, St*self.down)) # stock price at time t+1
                
                optimal_action =  np.argmax(self.q_value[state[0], state[1], :])
                reward_optimal += ((St_1 - St) + self.Actions[optimal_action]*(St - St_1))
                reward_0 += (St_1 - St)
                reward_half += ((St_1 - St) + self.Actions[1]*(St - St_1))
                reward_1 += ((St_1 - St) + self.Actions[2]*(St - St_1))
                reward_random += ((St_1 - St) + self.Actions[np.random.choice([0,1,2])]*(St - St_1))
                
                t = state[1]+1
                next_state = [next_price, t]
                state = next_state
             
            rewards_optimal.append(reward_optimal/(i+1))
            rewards_0.append(reward_0/(i+1))
            rewards_half.append(reward_half/(i+1))
            rewards_1.append(reward_1/(i+1))
            rewards_random.append(reward_random/(i+1))
        plt.plot(np.array(rewards_optimal), label = 'optimal policy')
        plt.plot(np.array(rewards_0), label = 'always 0')
        plt.plot(np.array(rewards_half), label = 'always 0.5')
        plt.plot(np.array(rewards_1), '--',linewidth = 0.5, label = 'always 1')
        plt.plot(np.array(rewards_random), label = 'random policy')
        plt.title('policy rewards against sampling sizes')
        plt.xlabel('runs')
        plt.ylabel('average rewards')
        plt.legend(loc = 'lower right')
    
if __name__ == '__main__':
    solver = hedging_strategy(1)
    solver.train()
    solver.print_optimal_policy()
    q_value = solver.get_q_value()
    #solver.plot_training()
    solver.compare_results(100000) 
