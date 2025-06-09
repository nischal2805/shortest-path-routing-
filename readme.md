# Shortest Path Routing with Reinforcement Learning

This is a Reinforcement Learning model designed to optimize network routing by finding the fastest paths through networks using intelligent agent-based decision making.

## Project Overview

This project implements a reinforcement learning environment for network routing optimization. The system uses a gym-compatible environment where an RL agent learns to navigate through network graphs to find optimal paths between source and destination nodes, considering factors like latency and bandwidth.

## Features

- **Custom Gym Environment**: Implements a custom OpenAI Gym environment for network routing
- **Graph-based Network Simulation**: Uses NetworkX for representing and manipulating network topologies
- **Multi-objective Optimization**: Considers both latency and bandwidth in routing decisions
- **Training Data Collection**: Automatically saves training metrics and performance data
- **Flexible Graph Input**: Supports custom network graphs or auto-generated topologies

## Project Structure

```
d:\main-sr\
├── environment/
│   ├── env.py              # Main RL environment implementation
│   └── util.py             # Utility functions for graph operations
├── helper/
│   └── graph.py            # Graph helper functions
├── training_data/          # Directory for storing training results
└── readme.md
```

## Requirements

- Python 3.7+
- gym
- networkx
- pandas
- numpy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/nischal2805/Shortest-path-routing.git
cd Shortest-path-routing
```

2. Install required dependencies:
```bash
pip install gym networkx pandas numpy
```

## How to Run

### Basic Usage

1. **Import the environment:**
```python
from environment.env import Env

# Initialize environment with a save file for training data
env = Env(save_file="training_data/results.csv")
```

2. **Train an agent:**
```python
# Reset environment to get initial state
state = env.reset()

# Training loop
for episode in range(1000):
    state = env.reset()
    done = False
    
    while not done:
        # Choose action (replace with your RL algorithm)
        action = env.action_space.sample()  # Random action for demo
        
        # Take step in environment
        next_state, reward, done, info = env.step(action)
        state = next_state
```

3. **Use custom graph:**
```python
import networkx as nx

# Create custom network topology
custom_graph = nx.Graph()
custom_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 3)])

# Initialize environment with custom graph
env = Env(save_file="custom_results.csv", graph=custom_graph)
```

### Training Modes

The environment supports both training and testing modes:

```python
# Training mode (default)
state, reward, done, info = env.step(action, train_mode=True)

# Testing mode
state, reward, done, info = env.step(action, train_mode=False)
```

## Environment Details

### Observation Space
- **Type**: MultiDiscrete([num_nodes, num_nodes])
- **Description**: [current_node, target_node]

### Action Space
- **Type**: Discrete(max_neighbors)
- **Description**: Index of neighbor node to move to

### Reward Function
The environment provides multi-objective rewards considering:
- **Path completion**: Reward for reaching the target
- **Latency**: Penalty for longer paths
- **Bandwidth**: Reward for higher bandwidth paths

### Output Files
- `{save_file}`: Training episode results (steps, reward, latency, bandwidth)
- `{save_file}_test.csv`: Testing episode results
- `training_data/step_data.csv`: Step-by-step training data

## Example Complete Script

```python
from environment.env import Env
import numpy as np

# Initialize environment
env = Env(save_file="training_data/my_experiment.csv")

# Simple training loop
num_episodes = 100

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        # Get valid actions
        valid_actions = state[2]['valid_actions'] if len(state) > 2 else [1] * env.max_neighbors
        
        # Choose random valid action (replace with your RL algorithm)
        valid_indices = [i for i, valid in enumerate(valid_actions) if valid == 1]
        action = np.random.choice(valid_indices)
        
        # Take step
        state, reward, done, info = env.step(action)
        total_reward += reward
    
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

print("Training completed! Check training_data/my_experiment.csv for results.")
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).
