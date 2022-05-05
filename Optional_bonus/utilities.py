# -*- coding: utf-8 -*-
"""
Created on Wed May  4 17:33:34 2022

@author: George Chen
"""

import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt

def bs_call(St, K, r, sigma, tau):
    if tau == 0:
        return max(St-K, 0)
    else:
        d1 = ( np.log(St / K) + (r + 1/2*sigma**2)*tau ) / (sigma*np.sqrt(tau))
        d2 = d1 - sigma*np.sqrt(tau)
    return float(St*norm.cdf(d1) - K*np.exp(-r*tau)*norm.cdf(d2))

def delta(St, K, r, sigma, tau):
    if tau == 0:
        return float(np.where(St>=K, 1, 0))
    else:
        d1 = ( np.log(St / K) + (r + 1/2*sigma**2)*tau ) / (sigma*np.sqrt(tau))
    return float(norm.cdf(d1))
  
class HedgingEnv_Continuous:
  ''' construct a hedging environment with continuous action space for DDPG algorithm'''
  def __init__(self,S,K,T,sigma,rf):

    self.S = S
    self.K = K
    self.T = T
    self.sigma = sigma
    self.rf = rf

    self.stock_pos = 0 
    self.S0 = S
    
    self.trading_days = 50
    self.dt = 1.0/self.trading_days
    self.ttm = T

  
  def option_price(self):
    return bs_call(self.S, self.K, self.rf, self.sigma, self.ttm)

  
  def true_delta(self):
    return delta(self.S, self.K, self.rf, self.sigma, self.ttm)
    
  def step(self, action):
    temp_option = self.option_price()
    self.ttm = self.ttm - self.dt
    temp_S = self.S
        
    self.stock_pos = action
    self.S = self.S * np.exp((self.rf - self.sigma * self.sigma / 2) * self.dt + self.sigma * np.random.normal() * np.sqrt(self.dt)) 
    
    option = self.option_price()
    reward = temp_option - option + self.stock_pos * (self.S - temp_S)
    #reward = -reward+0.05*reward*reward
    reward = -abs(reward)
    state = [self.S, self.ttm]
    
    if self.ttm <= 0 or np.isclose(self.ttm, 0):
        done = True
    else:
        done = False
    #return reward, state, self.true_delta
    return  state, reward, done

class HedgingEnv_Discrete:
  ''' construct a hedging environment with discrete action space for DDQN algorithm'''
  def __init__(self,S,K,T,sigma,rf):

    self.S = S
    self.K = K
    self.T = T
    self.sigma = sigma
    self.rf = rf

    self.stock_pos = 0 
    self.S0 = S
    
    self.trading_days = 50
    self.dt = 1.0/self.trading_days
    self.ttm = T
    
    self.actions_num = 100
    self.lower_bound = 0.0
    self.upper_bound = 1.0
    self.action_map = {i: self.lower_bound + i*(self.upper_bound-self.lower_bound)/self.actions_num for i in range(self.actions_num+1)}
  
  def option_price(self):
    return bs_call(self.S, self.K, self.rf, self.sigma, self.ttm)

  
  def true_delta(self):
    return delta(self.S, self.K, self.rf, self.sigma, self.ttm)
    
  def step(self, action):
    temp_option = self.option_price()
    self.ttm = self.ttm - self.dt
    temp_S = self.S
        
    self.stock_pos = self.action_map[int(action)]
    self.S = self.S * np.exp((self.rf - self.sigma * self.sigma / 2) * self.dt + self.sigma * np.random.normal() * np.sqrt(self.dt)) 
    
    option = self.option_price()
    reward = temp_option - option + self.stock_pos * (self.S - temp_S)
    #reward = -reward+0.05*reward*reward
    reward = -abs(reward)
    state = [self.S, self.ttm]
    
    if self.ttm <= 0 or np.isclose(self.ttm, 0):
        done = True
    else:
        done = False
    #return reward, state, self.true_delta
    return  state, reward, done

def episodes_50_return_continuous(agent, S, K, T, sigma, r):
    
    '''
    evaluation function for continuous hedging environment.
    this function will simmulate 50 episodes under the current hedging strategy 
    '''
    
    scores = []
    for i in range(50):
        env = HedgingEnv_Continuous(S, K, T, sigma, r)
        done = False
        score = 0
        state = [S, T]
        while not done:
            action = agent.final_action(state)
            next_state, reward, done = env.step(action)
            score += reward
            state = next_state
            
        scores.append(score)
    
    return np.mean(scores)

def episodes_50_return_discrete(agent, S, K, T, sigma, r):
    
    '''
    evaluation function for discrete hedging environment.
    this function will simmulate 50 episodes under the current hedging strategy 
    '''
    
    scores = []
    for i in range(50):
        env = HedgingEnv_Discrete(S, K, T, sigma, r)
        done = False
        score = 0
        state = [S, T]
        while not done:
            action = agent.final_action(state)
            next_state, reward, done = env.step(action)
            score += reward
            state = next_state
            
        scores.append(score)
    
    return np.mean(scores)