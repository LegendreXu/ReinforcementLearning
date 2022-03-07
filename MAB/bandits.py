# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 21:40:50 2022

@author: 13732
"""

import pandas as pd
import numpy as np

class Bandit():
    # @k_arm: # of arms
    # @epsilon: probability for exploration in epsilon-greedy algorithm
    # @initial: initial estimation for each action
    # @step_size: constant step size for updating estimations
    # @sample_averages: if True, use sample averages to update estimations instead of constant step size
    # @UCB_param: if not None, use UCB algorithm to select action
    # @gradient: if True, use gradient based bandit algorithm
    # @gradient_baseline: if True, use average reward as baseline for gradient based bandit algorithm
    def __init__(self, stock_data, epsilon=0., step_size=0.1, sample_averages=False, UCB_param=None,
                 gradient=False, gradient_baseline=False, America=False):
        self.k_arm = len(stock_data)
        self.step_size = step_size
        self.sample_averages = sample_averages
        self.indices = np.arange(self.k_arm)
        self.time = 0
        self.UCB_param = UCB_param
        self.gradient = gradient
        self.gradient_baseline = gradient_baseline
        self.average_reward = 0
        self.data = stock_data
        self.epsilon = epsilon
        self.action_count = np.zeros(self.k_arm)
        self.initial = np.zeros(self.k_arm)
        self.q_estimation = np.zeros(self.k_arm)
        self.time = 0
    
    ## process the stock data, generate the daily return    
    def generate_stock_ret(self, America):
        res = []
        for item in self.data.keys():
            sample = self.data[item].close
            temp = sample.apply(np.log).diff()[1:]
            temp = temp.to_frame()
            temp.columns = [item]
            res.append(temp)

        self.df = pd.concat(res, axis=1)
        if America:
            self.df = self.df[self.df.index>'2016-01-01']
         
        self.df.fillna(0, inplace=True)
    
    ## setting initial value, q_estimation
    def initialization(self):
        history = np.mean(self.df[:120], axis=0)
        self.df = self.df[120:]
        self.q_estimation = history.values
        self.time = 0
        
    # get an action for this bandit
    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices)

        if self.UCB_param is not None:
            UCB_estimation = self.q_estimation + \
                self.UCB_param * np.sqrt(np.log(self.time + 1) / (self.action_count + 1e-5))
            q_best = np.max(UCB_estimation)
            return np.random.choice(np.where(UCB_estimation == q_best)[0])

        if self.gradient:
            exp_est = np.exp(self.q_estimation)
            self.action_prob = exp_est / np.sum(exp_est)
            return np.random.choice(self.indices, p=self.action_prob)

        q_best = np.max(self.q_estimation)
        return np.random.choice(np.where(self.q_estimation == q_best)[0])

    # take an action, update estimation for this action
    def step(self, action):
        # generate the reward according to the stock return at time(self.time)
        all_rewards = self.df.iloc[self.time]
        reward = all_rewards[action]
        optimal_action = np.argmax(all_rewards)
        
        self.time += 1
        self.action_count[action] += 1
        self.average_reward += (reward - self.average_reward) / self.time

        if self.sample_averages:
            # update estimation using sample averages
            self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
        elif self.gradient:
            one_hot = np.zeros(self.k)
            one_hot[action] = 1
            if self.gradient_baseline:
                baseline = self.average_reward
            else:
                baseline = 0
            self.q_estimation += self.step_size * (reward - baseline) * (one_hot - self.action_prob)
        else:
            # update estimation with constant step size
            self.q_estimation[action] += self.step_size * (reward - self.q_estimation[action])
        return reward, self.q_estimation, optimal_action

def multi_armed_bandit(stock_data, epsilon=0.1, step_size=0.1, sample_averages=False, UCB_param=None, gradient=False, gradient_baseline=False, America=False):
    bandit = Bandit(stock_data, epsilon, step_size, sample_averages, UCB_param, gradient, gradient_baseline, America)
    bandit.generate_stock_ret(America)
    bandit.initialization()
    n_steps = len(bandit.df) 
    qs = np.zeros((n_steps, bandit.k_arm))
    rewards = np.zeros(n_steps)
    actions = np.zeros(n_steps)
    optimal = np.zeros(n_steps)
   
    for t in range(n_steps):
   
      action = bandit.act()
      actions[t] = action
   
      rewards[t], qs[t], optimal_action = bandit.step(action)
      optimal[t] = action == optimal_action
   
    results = {
        'qs': qs,
        'actions': actions,
        'rewards': rewards,
        'optimal': optimal
    }
   
    return results, bandit.df


def show_result(df, result):
    df0 = df.copy()
    ret_lis = []
    for i in range(len(df0)):
        num = int(result['actions'][i])
        ret_lis.append(df0.iloc[i,num])
    
    ret_df = pd.DataFrame(ret_lis, index=df0.index)
    net_value = ret_df.cumsum().apply(np.exp)
    
    
    import matplotlib.ticker as ticker
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(20,5))
    ax = fig.add_subplot(111)
    i = np.argmax(np.maximum.accumulate(net_value) - net_value) # end of the period
    j = np.argmax(net_value[:i]) # start of period
    ax.plot(net_value)
    ax.plot([net_value.index[i], net_value.index[j]], [net_value.iloc[i], net_value.iloc[j]], 'o', color='Red', markersize=10)
    tick_space = 250
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_space))
    plt.grid()
    plt.show()
    
    return net_value




