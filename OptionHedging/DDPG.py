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
    
class OUActionNoise(object):
    ## generate action noise for each exploration time
    def __init__(self, mu, sigma=0.1, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.reset()
        self.sigma = sigma
        
    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size = self.mu.shape)
        self.x_prev = x
        return x
    
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
        
    
    
class CriticNN(nn.Module):
    '''  construct a critic network here for state valuation'''
    def __init__(self, LR, input_dims, fc1_dims, fc2_dims, name):
        super(CriticNN, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        
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
        
        self.action_value = nn.Linear(1, fc2_dims) ## genearte state_action value
        
        f3 = 0.003
        self.q = nn.Linear(self.fc2_dims, 1)
        torch.nn.init.uniform_(self.q.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.q.bias.data, -f3, f3)
        
        self.optimizer = optim.Adam(self.parameters(), lr = LR)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.to(self.device)
    
    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        
        action_value = F.relu(self.action_value(action))
        
        state_action_value = F.relu(torch.add(state_value, action_value))
        state_action_value = self.q(state_action_value)
        
        return state_action_value
    
class ActorNN(nn.Module):
    ''' construct the actor network here'''
    def __init__(self, theta, input_dims, fc1_dims, fc2_dims, name):
        ## theta: learning rate 
        super(ActorNN, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        
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
        
        f3 = 0.003
        self.mu = nn.Linear(self.fc2_dims, 1)
        torch.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.mu.bias.data, -f3, f3)
        
        self.optimizer = optim.Adam(self.parameters(), lr = theta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.to(self.device)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.tanh(self.mu(x))
        
        return torch.abs(x)
    
class DDPGAgent(object):
    ''' our final agent'''
    def __init__(self, LR, theta, input_dims, tau, gamma = 0.95,
        max_size = 1000000, fc1_dims = 400, fc2_dims = 300, batch_size = 64):
        
        ## LR: learning rate for the Critic Network
        ## theta: learning rate for the Actor Network
        ## input_dims: state dimension ( 2 in our hedging problems)
        ## tau: updating parameters
        ## gamma: discount reward rate
        
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims)
        self.batch_size = batch_size
        
        self.actor = ActorNN(theta, input_dims, fc1_dims, fc2_dims,  name = 'Actor')
        self.target_actor = ActorNN(theta, input_dims, fc1_dims, fc2_dims, name = 'TargetActor')
        
        self.critic = CriticNN(LR, input_dims, fc1_dims, fc2_dims,  name = 'Critic')
        self.target_critic = CriticNN(LR, input_dims, fc1_dims, fc2_dims, name = 'TargetCritic')
        
        self.noise = OUActionNoise(mu=np.zeros(1))
        
        self.update_network_parameters(tau = 1)
        
    def choose_action(self, observation):
        ## choose action given observation
        self.actor.eval()
        observation = torch.tensor(observation, dtype = torch.float).to(self.actor.device)
        mu = self.actor(observation).to(self.actor.device)
        #mu_prime = mu + torch.tensor(self.noise(), dtype = torch.float).to(self.actor.device)
        mu_prime = mu + torch.tensor(self.noise(), dtype = torch.float).to(self.actor.device) ## Add noise for exploration
        self.actor.train()
        return max(mu_prime.cpu().detach().numpy(), 0)
    
    def final_action(self, observation):
        ## choose action given observation. Without exploration!
        
        self.actor.eval()
        observation = torch.tensor(observation, dtype = torch.float).to(self.actor.device)
        mu = self.actor(observation).to(self.actor.device)
        self.actor.train()
        return max(mu.cpu().detach().numpy(), 0)
        
    
    def remember(self, state, action, reward, next_state, done):
        ## store transition
        self.memory.store_transition(state, action, reward, next_state, done)
    
    def learn(self):
        if self.memory.memory_counter < self.batch_size:
            ## Only learn when the size of memory buffer is larger than batch size
            return
        
        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)
        reward = torch.tensor(reward, dtype=torch.float).to(self.critic.device)
        done = torch.tensor(done).to(self.critic.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.critic.device)
        state = torch.tensor(state, dtype=torch.float).to(self.critic.device)
        
        ## Since the action is in 1 dimension here, we change it to matrix representation
        action = [action]
        action = torch.tensor(action, dtype=torch.float).to(self.critic.device)
        action = action.T
        
        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()
        
        target_actions = self.target_actor.forward(next_state)
        critic_value_ = self.target_critic.forward(next_state, target_actions)
        critic_value = self.critic.forward(state, action)
        
        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j]*done[j])
        target = torch.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)
        
        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()
        
        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()
        
        self.update_network_parameters()
    
        
    def update_network_parameters(self, tau=None):
        ## update network with decay tau
        if tau is None:
            tau = self.tau
            
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()
        
        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)
        
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + (1-tau)*target_critic_state_dict[name].clone()
        
        self.target_critic.load_state_dict(critic_state_dict)
        
        
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + (1-tau)*target_actor_state_dict[name].clone()
    
        self.target_actor.load_state_dict(actor_state_dict)
        

