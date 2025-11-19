"""
train.py

Train PPO agent to optimize qubit placement.
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
from qubit_env import QubitPlacementEnv
import os


class TrainingCallback(BaseCallback):
    """
    Custom callback to track training metrics.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_noises = []
        
    def _on_step(self) -> bool:
        # Check if any environment just finished an episode
        dones = self.locals.get('dones', [])
        infos = self.locals.get('infos', [])
        
        for i, done in enumerate(dones):
            if done and len(infos) > i:
                info = infos[i]
                
                # Track total noise if available
                if 'total_noise' in info:
                    self.episode_noises.append(info['total_noise'])
        
        return True



def train_agent(n_qubits=10, 
                compute_capacity=5,
                total_timesteps=50000,
                n_envs=4,
                model_name='ppo_qubit_placement'):
    """
    Train PPO agent on qubit placement task.
    
    Args:
        n_qubits: Number of qubits
        compute_capacity: Max qubits in compute region
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        model_name: Name for saving model
        
    Returns:
        Trained PPO model
    """
    print("=" * 60)
    print("Training RL Agent for Qubit Placement")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Qubits: {n_qubits}")
    print(f"  - Compute capacity: {compute_capacity}")
    print(f"  - Total timesteps: {total_timesteps}")
    print(f"  - Parallel environments: {n_envs}")
    print("=" * 60)
    
    # Create vectorized environments (multiple parallel envs for faster training)
    env = make_vec_env(
        QubitPlacementEnv,
        n_envs=n_envs,
        env_kwargs={
            'n_qubits': n_qubits,
            'compute_capacity': compute_capacity,
            'noise_compute': 1.0,
            'noise_storage': 0.1,
            'noise_move': 0.5,
            'circuit_depth': 20,
            'gate_density': 0.3
        }
    )
    
    print("\n✓ Environments created")
    
    # Create PPO agent
    model = PPO(
        'MlpPolicy',              # Multi-layer perceptron policy
        env,
        learning_rate=3e-4,       # Standard learning rate
        n_steps=2048,             # Steps per update
        batch_size=64,            # Batch size for training
        n_epochs=10,              # Training epochs per update
        gamma=0.99,               # Discount factor
        gae_lambda=0.95,          # GAE parameter
        clip_range=0.2,           # PPO clipping
        verbose=1,                # Print training info
        tensorboard_log='../tensorboard_logs/'
    )
    
    print("\n✓ PPO agent created")
    print("\nStarting training...")
    print("-" * 60)
    
    # Create callback for tracking
    callback = TrainingCallback()
    
    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    print("-" * 60)
    print("\n✓ Training complete!")
    
    # Save the trained model
    models_dir = '../models'
    os.makedirs(models_dir, exist_ok=True)
    model_path = f'{models_dir}/{model_name}'
    model.save(model_path)
    print(f"\n✓ Model saved to: {model_path}.zip")
    
    # Clean up
    env.close()
    
    return model, callback


def plot_training_progress(callback, save_path='../results/training_progress.png'):
    """
    Plot training metrics.
    
    Args:
        callback: TrainingCallback with logged metrics
        save_path: Where to save plot
    """
    if len(callback.episode_rewards) == 0:
        print("No episode data to plot")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot episode rewards
    axes[0].plot(callback.episode_rewards, alpha=0.6, label='Episode Reward')
    
    # Moving average for smoothing
    window = min(50, len(callback.episode_rewards) // 10)
    if window > 1:
        rewards_smooth = np.convolve(
            callback.episode_rewards, 
            np.ones(window)/window, 
            mode='valid'
        )
        axes[0].plot(rewards_smooth, 'r-', linewidth=2, label=f'Moving Avg ({window})')
    
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Training Progress: Episode Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot total noise per episode
    if len(callback.episode_noises) > 0:
        axes[1].plot(callback.episode_noises, alpha=0.6, label='Total Noise')
        
        # Moving average
        if window > 1:
            noise_smooth = np.convolve(
                callback.episode_noises,
                np.ones(window)/window,
                mode='valid'
            )
            axes[1].plot(noise_smooth, 'r-', linewidth=2, label=f'Moving Avg ({window})')
        
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Total Noise')
        axes[1].set_title('Training Progress: Total Noise per Episode')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Training plot saved to: {save_path}")
    
    plt.close()


if __name__ == "__main__":
    # Train the agent
    model, callback = train_agent(
        n_qubits=10,
        compute_capacity=5,
        total_timesteps=50000,  # Adjust based on time available
        n_envs=4,
        model_name='ppo_qubit_v1'
    )
    
    # Plot training progress
    plot_training_progress(callback)
    
    print("\n" + "=" * 60)
    print("✅ Training complete! Model saved.")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run evaluate.py to test the trained agent")
    print("  2. Compare against random and greedy baselines")
    print("  3. Visualize placement decisions")
