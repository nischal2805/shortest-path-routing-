import itertools
from collections import defaultdict
import json
import os
import time

class HyperparameterTuner:
    """Framework for hyperparameter tuning"""
    
    def __init__(self, param_grid, env_creator, model_creator, 
                 episodes_per_config=1000, eval_episodes=200):
        self.param_grid = param_grid
        self.env_creator = env_creator
        self.model_creator = model_creator
        self.episodes_per_config = episodes_per_config
        self.eval_episodes = eval_episodes
        self.results = defaultdict(list)
        
    def generate_configs(self):
        """Generate all possible configurations from the parameter grid"""
        keys = self.param_grid.keys()
        values = list(self.param_grid.values())
        for config_values in itertools.product(*values):
            yield dict(zip(keys, config_values))
    
    def run(self, save_dir="hyperparameter_results"):
        """Run hyperparameter tuning"""
        os.makedirs(save_dir, exist_ok=True)
        
        for config in self.generate_configs():
            print(f"Testing configuration: {config}")
            
            # Create environment and model with this config
            env = self.env_creator(config)
            model = self.model_creator(env, config)
            
            # Train model
            start_time = time.time()
            model.train(self.episodes_per_config)
            train_time = time.time() - start_time
            
            # Evaluate model
            success_rate, avg_reward = self._evaluate(model, env)
            
            # Record results
            config_str = json.dumps(config, sort_keys=True)
            self.results[config_str].append({
                'success_rate': success_rate,
                'avg_reward': avg_reward,
                'train_time': train_time
            })
            
            # Save interim results
            self._save_results(save_dir)
        
        return self._get_best_config()
    
    def _evaluate(self, model, env):
        """Evaluate a model configuration"""
        total_reward = 0
        successes = 0
        
        for _ in range(self.eval_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = model.select_action(state, evaluation=True)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                
            total_reward += episode_reward
            if episode_reward > 0:  # Assuming positive reward means success
                successes += 1
                
        return successes / self.eval_episodes, total_reward / self.eval_episodes
    
    def _save_results(self, save_dir):
        """Save tuning results to a file"""
        with open(f"{save_dir}/results_{int(time.time())}.json", 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def _get_best_config(self):
        """Get the best configuration based on success rate and reward"""
        best_config = None
        best_score = float('-inf')
        
        for config_str, results in self.results.items():
            avg_success = sum(r['success_rate'] for r in results) / len(results)
            avg_reward = sum(r['avg_reward'] for r in results) / len(results)
            
            # Score is a combination of success rate and reward
            score = avg_success * 10 + avg_reward
            
            if score > best_score:
                best_score = score
                best_config = json.loads(config_str)
                
        return best_config