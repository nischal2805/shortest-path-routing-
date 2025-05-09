import logging
import traceback
import os
import networkx as nx  # Add missing import

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=f"{log_dir}/training.log",
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class ErrorHandler:
    """Centralized error handling with logging and recovery"""
    
    @staticmethod
    def handle_training_error(func):
        """Decorator for handling training errors"""
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(f"Error in {func.__name__}: {str(e)}")
                logging.error(traceback.format_exc())
                
                # Try to recover training state if possible
                try:
                    model = args[0]  # Assuming first arg is the model
                    model.save("error_recovery_checkpoint.pt")
                    logging.info("Created recovery checkpoint")
                except:
                    logging.error("Failed to create recovery checkpoint")
                
                # Re-raise with more context
                raise RuntimeError(f"Training error in {func.__name__}: {str(e)}") from e
        return wrapper
    
    @staticmethod
    def validate_graph(graph):
        """Validate a graph structure"""
        if graph is None:
            raise ValueError("Graph cannot be None")
            
        if graph.number_of_nodes() == 0:
            raise ValueError("Graph has no nodes")
            
        # Check connectivity
        if not nx.is_connected(graph):
            logging.warning("Graph is not connected - routing may fail")
        
        return True
    
    @staticmethod
    def validate_environment_step(state, reward, done, info):
        """Validate environment step output"""
        if state is None:
            raise ValueError("Environment returned None state")
        
        if reward is None:
            raise ValueError("Environment returned None reward")
            
        if not isinstance(done, bool):
            raise ValueError(f"Environment returned non-boolean done value: {done}")
            
        return True