import networkx as nx

class RewardFunction:
    """Improved reward function with shaping to guide learning"""
    
    def __init__(self, success_reward=1.0, failure_penalty=-1.5, 
                 distance_factor=0.05, congestion_factor=0.1,
                 step_penalty=0.01):
        self.success_reward = success_reward
        self.failure_penalty = failure_penalty
        self.distance_factor = distance_factor
        self.congestion_factor = congestion_factor
        self.step_penalty = step_penalty
        
    def calculate(self, graph, current_node, target_node, path, 
                  done=False, max_steps=None):
        """
        Calculate reward with shaping components
        
        Args:
            graph: The network graph
            current_node: Current node
            target_node: Target node
            path: Path taken so far
            done: Whether episode is done
            max_steps: Maximum steps allowed (for timeout)
            
        Returns:
            tuple: (reward, is_terminal)
        """
        # Check for success
        if current_node == target_node:
            # Success - reached target
            path_length = sum(graph[path[i]][path[i+1]]["weight"] 
                             for i in range(len(path)-1))
                
            # Calculate optimal path length
            try:
                optimal_path = nx.shortest_path(graph, path[0], target_node, weight="weight")
                optimal_length = sum(graph[optimal_path[i]][optimal_path[i+1]]["weight"] 
                                   for i in range(len(optimal_path)-1))
                efficiency = optimal_length / max(0.001, path_length)
            except:
                efficiency = 1.0
                
            # Calculate congestion
            try:
                edge_congestion = max(0, min(1, 1 - min(graph[path[-2]][path[-1]]["capacity"], 1.0)))
            except:
                edge_congestion = 0
                
            # Combine rewards
            reward = self.success_reward * (0.7 + 0.3 * efficiency) - (self.congestion_factor * edge_congestion)
            return reward, True
            
        # Check for timeout
        if max_steps and len(path) >= max_steps:
            return self.failure_penalty, True
            
        # Check for loops
        if path.count(current_node) > 1:
            # Penalize loops
            return self.failure_penalty * 0.5, False
            
        # Progress reward - are we getting closer?
        if len(path) >= 2:
            previous_node = path[-2]
            
            # Calculate distances to target
            try:
                current_distance = nx.shortest_path_length(
                    graph, current_node, target_node, weight="weight")
                previous_distance = nx.shortest_path_length(
                    graph, previous_node, target_node, weight="weight")
                
                if current_distance < previous_distance:
                    # Getting closer
                    progress_reward = self.distance_factor * (previous_distance - current_distance)
                    return -self.step_penalty + progress_reward, False
                else:
                    # Getting further away
                    return -self.step_penalty * 2, False
            except:
                # No path to target
                return self.failure_penalty, True
                
        # Default small negative reward
        return -self.step_penalty, False