import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
from config import device
from models.base_model import BaseRoutingModel

# Add these constants back for compatibility with GCN.py
BATCH_SIZE = 192
GAMMA = 0.999
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 75000
TARGET_UPDATE = 10
LR = 0.001

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.random_memory = [] # For exploration samples
        self.position = 0
        self.num_pushes = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.random_memory.append(None)
        self.memory[self.position] = Transition(*args)

        if self.num_pushes < 2*self.capacity:
            self.random_memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        self.num_pushes += 1

    def sample(self, batch_size):
        return random.sample(self.memory, min(batch_size, len(self.memory)))

    def regular_sample(self, batch_size):
        regular_memory_batch_size = int(batch_size*.7)
        random_memory_batch_size = batch_size - regular_memory_batch_size
        
        # Make sure we don't try to sample more than available
        regular_memory_batch_size = min(regular_memory_batch_size, len(self.memory))
        random_memory_batch_size = min(random_memory_batch_size, len(self.random_memory))
        
        buffer_sample = random.sample(self.memory, regular_memory_batch_size)
        if random_memory_batch_size > 0:
            buffer_sample += random.sample(self.random_memory, random_memory_batch_size)
        return buffer_sample

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, inputs, outputs, hidden_size=None):
        super(DQN, self).__init__()
        if hidden_size is None:
            hidden_size = max(64, int((inputs+outputs)/2))
        
        self.fc1 = nn.Linear(inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, outputs)
        
        # Better initialization for improved learning
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.kaiming_normal_(self.fc4.weight)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

class Agent(BaseRoutingModel):
    def __init__(self, outputs, policy_net, target_net, config=None):
        # Create an empty dict if config is None
        config_dict = config or {}
        super().__init__(None, config_dict)
        self.n_actions = outputs
        self.policy_net = policy_net.to(device)
        self.target_net = target_net.to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Use config_dict instead of config
        self.batch_size = config_dict.get('batch_size', BATCH_SIZE)
        self.gamma = config_dict.get('gamma', GAMMA)
        self.eps_start = config_dict.get('eps_start', EPS_START)
        self.eps_end = config_dict.get('eps_end', EPS_END)
        self.target_update = config_dict.get('target_update', TARGET_UPDATE)
        self.epsilon_decay = config_dict.get('epsilon_decay', EPS_DECAY)
        self.lr = config_dict.get('learning_rate', LR)
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.episode_durations = []
        self.losses = []
        self.rewards = []
        
    def forward(self, state):
        """Forward pass through policy network"""
        return self.policy_net(state)
        
    def select_action(self, state, evaluation=False):
        """Select action using epsilon-greedy policy"""
        if evaluation:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
                
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.epsilon_decay)

        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)

    def predict(self, state):
        """Predict best action (no exploration)"""
        with torch.no_grad():
            return self.target_net(state).max(1)[1].view(1, 1)

    def optimize_model(self):
        """Perform one step of optimization"""
        if len(self.memory) < self.batch_size:
            return
            
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Create mask for non-final states
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), 
            device=device, dtype=torch.bool)
            
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(len(transitions), device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
            
        # Compute expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.losses.append(loss.item())
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to stabilize training
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        return loss.item()

class MultiAgent:
    def __init__(self, env, config=None):
        self.env = env
        self.config = config or {}
        self.num_agents = int(self.env.observation_space.nvec[0])
        self.agents = []
        self.nodes = list(self.env.graph.nodes)
        self.initialize()
        
    def initialize(self):
        """Initialize an agent for each node in the graph"""
        for i in range(len(self.nodes)):
            # For each node: inputs = one-hot target, outputs = number of neighbors
            num_inputs = self.num_agents
            num_outputs = len(list(self.env.graph.neighbors(self.nodes[i])))
            
            # Create policy and target networks
            policy_net = DQN(num_inputs, num_outputs).to(device)
            target_net = DQN(num_inputs, num_outputs).to(device)
            
            # Create agent
            self.agents.append(
                Agent(
                    num_outputs,
                    policy_net,
                    target_net,
                    self.config
                )
            )

    def _format_input(self, input):
        """Convert destination node to one-hot encoding"""
        try:
            arr = torch.zeros((1, self.num_agents), device=device)
            arr[0][self.nodes.index(input)] = 1
            return arr
        except Exception as e:
            print(f"Error in _format_input: {e}")
            print(f"Input: {input}, Nodes: {self.nodes}")
            raise

    def run(self, episodes=100):
        """Train the multi-agent system"""
        for i in range(episodes):
            # Reset environment
            obs, done = self.env.reset(), False
            curr_agent = self.nodes.index(obs[0])
            state = self._format_input(obs[1])
            total_reward = 0
            
            # Run episode
            while not done:
                # Select action
                action = self.agents[curr_agent].select_action(state)
                
                # Take step in environment
                obs, reward, done, infos = self.env.step(action.item())
                total_reward += reward
                
                # Convert reward to tensor
                reward = torch.tensor([reward], device=device)

                # Prepare next state (unless terminal)
                if not done:
                    next_state = self._format_input(obs[1])
                else:
                    next_state = None

                # Store transition in memory
                self.agents[curr_agent].memory.push(state, action, next_state, reward)

                # Move to next state
                state = next_state

                # Perform optimization step
                self.agents[curr_agent].optimize_model()

                # Update current agent
                curr_agent = self.nodes.index(obs[0])
                
            # Update target networks periodically
            if i % self.config.get('target_update', TARGET_UPDATE) == 0:
                for j in range(len(self.agents)):
                    self.agents[j].target_net.load_state_dict(
                        self.agents[j].policy_net.state_dict())
                        
            # Print progress
            if i % 100 == 0:
                print(f"Episode {i}/{episodes}, Reward: {total_reward:.2f}")

    def test(self, num_episodes=350):
        """Evaluate the multi-agent system"""
        good = 0
        bad = 0
        
        for _ in range(num_episodes):
            # Reset environment
            obs, done = self.env.reset(), False
            curr_node = obs[0]
            curr_agent = self.nodes.index(obs[0])
            obs = self._format_input(obs[1])
            total_reward = 0
            
            # Run episode
            while not done:
                with torch.no_grad():
                    # Select best action
                    action = self.agents[curr_agent].predict(obs)
                    
                    # Handle case where action is invalid
                    neighbors = list(self.env.graph.neighbors(curr_node))
                    if action.item() >= len(neighbors):
                        action = torch.tensor(
                            [[random.randint(0, len(neighbors)-1)]],
                            device=device
                        )
                    
                    # Take step in environment
                    obs, reward, done, infos = self.env.step(action.item(), train_mode=None)
                    total_reward += reward
                    
                    # Update current agent
                    curr_node = obs[0]
                    curr_agent = self.nodes.index(obs[0])
                    obs = self._format_input(obs[1])
            
            # Count successes and failures
            if reward == 1.01 or reward == -1.51:
                good += 1
            else:
                bad += 1
                
        success_rate = good / float(good + bad) if (good + bad) > 0 else 0
        print(f"Success rate: {success_rate:.4f} ({good}/{good+bad})")
        return success_rate
