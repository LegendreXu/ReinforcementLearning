# -*- coding: utf-8 -*-
"""
Created on Wed May  4 01:00:39 2022

@author: 13732
"""

from DDPG import DDPGAgent

from utilities import HedgingEnv, episodes_50_return, bs_call, delta, HedgingEnv_DQN, episodes_50_return_DQN

import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import random
import pickle

random.seed(24)

    
agent = DDPGAgent(LR = 0.00025, theta = 0.00025, input_dims=[2], tau=0.001, gamma = 1, batch_size=64, fc1_dims=400, fc2_dims=300 )
#score_history = []

S = 50
K = 50
T = 12/50 
sigma = 0.1
rf = 0

trail_scores = []
for i in range(20000):
    
    env = HedgingEnv(S, K, T, sigma, rf)
    done = False
    score = 0
    state = [S, T]
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, int(done))
        agent.learn()
        score += reward
        state = next_state
    
    #score_history.append(score)
    if i % 100 == 0:
        trial_50 = episodes_50_return(agent, S, K, T, sigma, rf)
        print('episode ', i, 'score %.2f' % score,
              'trail 50 games avergae %.2f' % trial_50)
        trail_scores.append(trial_50)
        if abs(trial_50) < 0.45:
            print('Finish training!!')
            break
plt.plot(trail_scores)
plt.title('50 episodes trail_reward')

open_file = open('./trail_scores.pkl', "wb")
pickle.dump(trail_scores, open_file)
open_file.close()


true_d = []
pred_d = []
for i in range(450, 550):
    St = i/10
    true_d.append(delta(St, K, rf, sigma, T))
    observation = [St, T]
    pred_d.append(float(agent.final_action(observation)))

data = pd.DataFrame([true_d, pred_d]).T
data.rename(columns = {0:'true_delta', 1:'pred_delta'}, inplace=True)
print(data)

def hedging(S, K, r, sigma, T, method, agent=None):
    
    if method == 'delta-hedging':
        hedge_pos = delta(S, K, r, sigma, T)
        
    
    #else:
     #   observation = [S, T]
      #  agent.actor.eval()
       # observation = torch.tensor(observation, dtype = torch.float).to(agent.actor.device)
        #mu = agent.actor(observation).to(agent.actor.device)
        #hedge_pos = float(mu.cpu().detach().numpy())
    
    else:
        observation = [S, T]
        hedge_pos = float(agent.final_action(observation))
    dt = 1/50
    St_1 = S
    portfolio = []
    call_porf = []
    stock_porf = []
    call = []
    stock = []
    for i in range(1, 13):
        tau = (12-i)/50
        tau_1 = (12 - (i-1))/50
        St = St_1*np.exp( (r - 1/2*sigma**2)*dt + sigma*norm.rvs(size=1)*np.sqrt(dt) )
        
        call_t_1 = bs_call(St_1, K, r, sigma, tau_1)
        call_t = bs_call(St, K, r, sigma, tau)
        portfolio.append( np.exp(-r*i/50)*( (call_t_1 - call_t) + hedge_pos * (St - St_1) ) )
        call_porf.append( ( (call_t_1 - call_t) ))
        stock_porf.append( ( (St - St_1) ))
        call.append(call_t)
        stock.append(St)
        St_1 = St
        if method == 'delta-hedging':
            hedge_pos = delta(St, K, r, sigma, tau)
        elif tau >= 0:
            observation = [float(St), tau]
            hedge_pos = float(agent.final_action(observation))
    return portfolio, call_porf, stock_porf,call,stock

t_delta = []
agent_delta = []
for i in range(400, 600):
    St = i/10
    t_delta.append(delta(St, K, rf, sigma, T))
    observation = [St,T]
    agent_delta.append(agent.final_action(observation))
plt.plot(t_delta, label = 'true delta')
plt.plot(agent_delta, label = 'agent delta')
plt.legend(loc='upper left')


p,c,s,call,stock = hedging(S, K, rf, sigma, T, 'delta-hedging')
p,c,s,call,stock = hedging(S, K, rf, sigma, T, 'agent-hedging', agent)
method = 'agent-hedging'

plt.plot(s)
plt.plot(c)
plt.plot(p)

plt.plot(np.cumsum(s))
plt.plot(np.cumsum(c))
plt.plot(np.cumsum(p))

plt.plot(call)
plt.plot(stock)


#torch.save(agent, r'D:\hkust\Course_2022Spring\6010Y\Assignment3\ddpg_agent1.pth')
#torch.save(agent, r'D:\hkust\Course_2022Spring\6010Y\Assignment3\ddpg_agent.pth')

torch.save(agent, r'./ddpg_agent.pth')




'''
for DQN
'''
########
########
########
from DQN_net import DQNAgent
n_actions = 11
DQNagent = DQNAgent(LR = 0.00005, input_dims=[2], tau=0.005, n_actions = n_actions, gamma = 1, batch_size=64, fc1_dims=400, fc2_dims=300)

S = 50
K = 50
T = 12/50 
sigma = 0.1
rf = 0

trail_scores_dqn = []
for i in range(20000):
    if i >0 and i % 200 == 0:
        DQNagent.epsilon_decay()
    env_dqn = HedgingEnv_DQN(S, K, T, sigma, rf)
    done = False
    score = 0
    state = [S, T]
    while not done:
        action = DQNagent.choose_action(state)
        next_state, reward, done = env_dqn.step(action)
        DQNagent.remember(state, action, reward, next_state, int(done))
        DQNagent.learn()
        score += reward
        state = next_state
    
    if i % 100 == 0:
        trial_50 = episodes_50_return_DQN(DQNagent, S, K, T, sigma, rf)
        print('episode ', i, 'score %.2f' % score,
              'trail 50 games avergae %.2f' % trial_50)
        trail_scores_dqn.append(trial_50)
        if abs(trial_50) < 0.45:
            print('Finish training!!')
            break
            
plt.plot(trail_scores_dqn)
plt.title('50 episodes trail_reward')

true_d = []
pred_d = []
for i in range(450, 550):
    St = i/10
    true_d.append(delta(St, K, rf, sigma, T))
    observation = [St, T]
    pred_d.append(float(DQNagent.final_action(observation))/100)

data = pd.DataFrame([true_d, pred_d]).T
data.rename(columns = {0:'true_delta', 1:'pred_delta'}, inplace=True)
print(data)