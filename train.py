from utils.error_handling import ErrorHandler
from environment.reward import RewardFunction
from utils.hyperparameter_tuning import HyperparameterTuner
from environment.parallel_env import ParallelEnv
import time
import torch
import os
import networkx as nx
import matplotlib.pyplot as plt
from config import device

# Import existing modules
from helper.graph import compute_flow_value
from environment.util import create_graph, get_flows, adjust_lat_band
from environment.env import Env as link_hop_env
from models.DQN import MultiAgent, DQN
from models.GCN import GCN_Agent, GCN
from environment.GCN_env import Env

# Create training directory
directory = "training_data"
if not os.path.exists(directory):
    os.makedirs(directory)

# Create models directory
models_dir = "saved_models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Default reward threshold for evaluation
lwr = -1.51

def save_model(model, model_name):
    """
    Save a model to disk
    
    Args:
        model: The model to save
        model_name: Name to use for the saved model
    """
    model_path = f"{models_dir}/{model_name}.pt"
    
    if hasattr(model, 'policy_net'):
        # For agents that have a policy network
        torch.save(model.policy_net.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    elif isinstance(model, MultiAgent):
        # For MultiAgent, save all agent networks
        os.makedirs(f"{models_dir}/{model_name}", exist_ok=True)
        for i, agent in enumerate(model.agents):
            agent_path = f"{models_dir}/{model_name}/agent_{i}.pt"
            torch.save(agent.policy_net.state_dict(), agent_path)
        print(f"MultiAgent model saved to {models_dir}/{model_name}/")
    else:
        # For regular networks
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

def load_model(model, model_name):
    """
    Load a model from disk
    
    Args:
        model: The model to load weights into
        model_name: Name of the saved model
    
    Returns:
        The loaded model
    """
    model_path = f"{models_dir}/{model_name}.pt"
    
    if os.path.exists(model_path):
        if hasattr(model, 'policy_net'):
            # For agents that have a policy network
            model.policy_net.load_state_dict(torch.load(model_path))
            model.target_net.load_state_dict(model.policy_net.state_dict())
            print(f"Loaded model from {model_path}")
        elif isinstance(model, MultiAgent):
            # For MultiAgent, load all agent networks
            agent_dir = f"{models_dir}/{model_name}"
            if os.path.exists(agent_dir):
                for i, agent in enumerate(model.agents):
                    agent_path = f"{agent_dir}/agent_{i}.pt"
                    if os.path.exists(agent_path):
                        agent.policy_net.load_state_dict(torch.load(agent_path))
                        agent.target_net.load_state_dict(agent.policy_net.state_dict())
                print(f"Loaded MultiAgent model from {agent_dir}/")
        else:
            # For regular networks
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded model from {model_path}")
    else:
        print(f"No saved model found at {model_path}")
    
    return model

def plot_metrics(metrics, title, save_path=None):
    """Plot training metrics and optionally save to file"""
    plt.figure(figsize=(12, 6))
    for name, values in metrics.items():
        if len(values) > 0:
            plt.plot(values, label=name)
    
    plt.title(title)
    plt.xlabel('Steps/Episodes')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def ma(train_model=True, evaluate_model=True, num_episodes=10000, load_saved=False):
    """Train and/or evaluate the Multi-Agent DQN model"""
    t0 = time.time()
    print("\n=== Multi-Agent DQN ===")
    
    # Create environment
    env = link_hop_env(f"{directory}/ma_dqn.csv", G)
    env.graph = adjust_lat_band(env.graph, flows)
    
    # Create model
    model = MultiAgent(env)
    
    # Load saved model if requested
    if load_saved:
        model = load_model(model, "ma_dqn")
    
    # Train model
    if train_model:
        print("Training model...")
        metrics = {'rewards': [], 'success_rate': []}
        
        # Training loop with periodic evaluation
        for i in range(0, num_episodes, 5000):
            # Train for 5000 episodes
            model.run(5000)
            
            # Save model
            save_model(model, "ma_dqn")
            
            # Periodic evaluation
            if evaluate_model:
                success_rate = model.test(100)
                metrics['success_rate'].append(success_rate)
                print(f"Episode {i+5000}/{num_episodes}, Success rate: {success_rate:.4f}")
        
        # Plot metrics
        plot_metrics(metrics, "Multi-Agent DQN Training", f"{directory}/ma_dqn_training.png")
    
    # Final evaluation
    if evaluate_model:
        print("\nFinal evaluation:")
        model.test(2000)
    
    t1 = time.time()
    total = t1-t0
    print(f"Multi Agent {round(total/60, 2)} mins")
    
    return model

def gcn(train_model=True, evaluate_model=True, num_episodes=10000, load_saved=False):
    """Train and/or evaluate the Single-Agent GCN model"""
    print("\n=== Single-Agent GCN ===")
    
    num_nodes = 50
    max_neighbors = 50
    
    # Setup environment
    environment = Env(f"{directory}/gcn.csv",
                    num_nodes_in_graph=num_nodes, 
                    max_neighbors=max_neighbors,
                    graph=G)
    
    # Create model
    policy_net = GCN(num_nodes, max_neighbors).to(device)
    target_net = GCN(num_nodes, max_neighbors).to(device)
    
    gcn_agent = GCN_Agent(
        outputs=num_nodes,
        policy_net=policy_net,
        target_net=target_net,
        num_nodes=num_nodes,
        env=environment
    )
    
    # Load saved model if requested
    if load_saved:
        gcn_agent = load_model(gcn_agent, "sa_gcn")
    
    # Train model
    if train_model:
        print("Training model...")
        start_time = time.time()
        gcn_agent.run(num_episodes)
        
        # Save model
        save_model(gcn_agent, "sa_gcn")
        
        print(f'GCN training completed in {(time.time()-start_time)/60:.2f} mins')
        
        # Plot metrics if available
        if hasattr(gcn_agent, 'metrics') and len(gcn_agent.metrics) > 0:
            plot_metrics(gcn_agent.metrics, "GCN Training Metrics", f"{directory}/gcn_training.png")
    
    # Evaluate model
    if evaluate_model:
        # Implement evaluation for GCN
        # This depends on how your GCN model evaluation is structured
        pass
    
    return gcn_agent

def ecmp():
    """Run ECMP baseline model"""
    t0 = time.time()
    env = link_hop_env(directory + "/" + "ecmp_150" + ".csv", G)
    env.graph.remove_nodes_from(nodes_to_remove)

    good = 0
    bad = 0
    reward = 0
    for _ in range(10_000):
        obs, done = env.reset(), False
        
        paths = nx.all_shortest_paths(env.graph, obs[0], obs[1])
        path = []
        b = -1

        for p in paths:
            if compute_flow_value(env.graph, tuple(p)) > b:
                b = compute_flow_value(env.graph, tuple(p))
                path = p

        for i in range(1, len(path)):
            action = env.neighbors.index(path[i])
            obs, reward, done, infos = env.step(action)

        if reward == 1.01 or reward == lwr:
            good += 1
        else:
            bad += 1

    t1 = time.time()
    total = t1-t0
    print(f"ecmp {round(total/60, 2)} mins")
    print(f"ecmp % Routed: {good / float(good + bad)}")

def spf():
    """Run Shortest Path First baseline model"""
    t0 = time.time()
    env = link_hop_env(directory + "/" + "spf_150" + ".csv", G)
    env.graph.remove_nodes_from(nodes_to_remove)
    
    good = 0
    bad = 0
    for _ in range(10_000):
        obs, done = env.reset(), False
        
        try:
            path = nx.shortest_path(env.graph, obs[0], obs[1])
            
            for i in range(1, len(path)):
                action = env.neighbors.index(path[i])
                obs, reward, done, infos = env.step(action)
            
            if reward == 1.01 or reward == lwr:
                good += 1
            else:
                bad += 1
                
        except nx.NetworkXNoPath:
            bad += 1
    
    t1 = time.time()
    total = t1-t0
    print(f"spf {round(total/60, 2)} mins")
    print(f"spf % Routed: {good / float(good + bad)}")

# Main execution
if __name__ == "__main__":
    # Create graph and generate flows
    G = create_graph(50, 100, "50nodes.brite")
    flows = get_flows(G, 70)
    nodes_to_remove = []
    
    # Skip multi-agent training and only train GCN with fewer episodes
    print("\nSkipping Multi-Agent DQN (already trained)")
    print("\nTraining only Single-Agent GCN...")
    
    # Reduce episodes from 10000 to a smaller number
    # Choose based on your hardware capabilities
    gcn_model = gcn(train_model=True, evaluate_model=True, num_episodes=1000)  # Reduced to 1000
