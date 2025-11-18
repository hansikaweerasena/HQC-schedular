import qiskit
import gymnasium as gym
from stable_baselines3 import PPO
import torch
import numpy as np
import matplotlib.pyplot as plt

print("✓ All imports successful!")
print(f"Qiskit version: {qiskit.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")

# Quick test
from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
print("\n✓ Qiskit circuit creation works!")

env = gym.make('CartPole-v1')
print("✓ Gymnasium environment works!")

model = PPO('MlpPolicy', env, verbose=0)
print("✓ Stable-Baselines3 PPO works!")

print("\n🎉 Setup complete! Ready to build POC.")
