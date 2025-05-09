import torch
import numpy as np
from multiprocessing import Pool
import copy

class ParallelEnv:
    """Process multiple environments in parallel for faster training"""
    
    def __init__(self, env_creator, num_envs=4, config=None):
        # Fix: Check if the env_creator accepts a config parameter
        try:
            # Try to create one environment with config
            self.envs = [env_creator(config) for _ in range(num_envs)]
        except TypeError:
            # If env_creator doesn't accept config, call it without arguments
            self.envs = [env_creator() for _ in range(num_envs)]
        
        self.num_envs = num_envs
        self.states = None
        self.reset()
    
    def reset(self):
        """Reset all environments"""
        self.states = [env.reset() for env in self.envs]
        return self.states
    
    def step(self, actions):
        """Take steps in all environments with given actions"""
        results = []
        
        # Execute steps sequentially (more reliable than Pool with custom environments)
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            next_state, reward, done, info = env.step(action)
            results.append((next_state, reward, done, info))
            
        # Unpack results
        next_states, rewards, dones, infos = zip(*results)
        
        # Reset environments that are done
        for i, done in enumerate(dones):
            if done:
                self.states[i] = self.envs[i].reset()
            else:
                self.states[i] = next_states[i]
                
        return next_states, rewards, dones, infos