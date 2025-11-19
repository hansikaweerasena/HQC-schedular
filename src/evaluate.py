"""
evaluate.py

Test the trained RL agent vs. baselines (random, greedy) on new circuits.
"""

import numpy as np
from stable_baselines3 import PPO
from qubit_env import QubitPlacementEnv
import matplotlib.pyplot as plt

def run_episode(env, model=None, mode='rl'):
    obs, info = env.reset()
    total_reward = 0.0
    total_noise = 0.0
    steps = 0

    done = False
    while not done:
        if mode == 'rl' and model is not None:
            action, _ = model.predict(obs, deterministic=True)
        elif mode == 'random':
            action = env.action_space.sample()
        elif mode == 'greedy':
            # KEEP qubits in compute if needed for next gate; otherwise, keep last state.
            locations = env.qubit_locations.copy()
            active_next = obs[env.n_qubits:2*env.n_qubits]
            action = np.zeros(env.n_qubits, dtype=np.int32)
            for q in range(env.n_qubits):
                # If will need gate next and currently in storage, move to compute
                if active_next[q] > 0 and locations[q] == 0:
                    action[q] = 1  # move to compute
                # If NOT needed, but currently in compute and compute is full, move to storage (not sophisticated for demo)
                elif active_next[q] == 0 and locations[q] == 1:
                    action[q] = 1  # move to storage
                else:
                    action[q] = 0  # stay
        else:
            raise ValueError("Unknown mode")
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_noise = info.get('total_noise', total_noise)
        steps += 1
        done = terminated or truncated
    return total_reward, total_noise, steps

def evaluate_agent(model_path, n_episodes=20):
    print("\n=== Evaluation on Random Circuits ===")
    env = QubitPlacementEnv(n_qubits=10, compute_capacity=5)
    model = PPO.load(model_path)

    scores = {'rl': [], 'random': [], 'greedy': []}
    noises = {'rl': [], 'random': [], 'greedy': []}

    for _ in range(n_episodes):
        for key in ['rl', 'random', 'greedy']:
            reward, noise, steps = run_episode(env, model if key=='rl' else None, mode=key)
            scores[key].append(reward)
            noises[key].append(noise)

    print("Avg Total Reward:")
    for k in scores:
        print(f"  {k.capitalize():<8}: {np.mean(scores[k]):.2f} (std {np.std(scores[k]):.2f})")
    print("Avg Total Noise:")
    for k in noises:
        print(f"  {k.capitalize():<8}: {np.mean(noises[k]):.2f} (std {np.std(noises[k]):.2f})")

    # Plot for visual comparison
    plt.figure(figsize=(8,5))
    plt.boxplot([noises['rl'], noises['greedy'], noises['random']], labels=['RL agent', 'Greedy', 'Random'])
    plt.title('Total Noise over 20 Test Circuits')
    plt.ylabel('Total Noise')
    plt.savefig('../results/eval_noise_boxplot.png')
    plt.show()
    print("\nBoxplot saved to ../results/eval_noise_boxplot.png")

if __name__ == "__main__":
    model_path = '../models/ppo_qubit_v1.zip'
    evaluate_agent(model_path)
