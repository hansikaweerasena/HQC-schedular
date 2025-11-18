# RL-Based Quantum Qubit Placement Optimizer

**Reinforcement learning framework for optimizing qubit placement between compute and storage regions in heterogeneous quantum computers.**

---

## Overview

This project develops a novel architecture-level optimization framework for quantum computers with heterogeneous qubit regions:
- **Compute qubits**: Fast gate execution but high noise and short coherence times
- **Storage qubits**: Slow/no gates but low noise and long coherence times

The core challenge: **dynamically decide where each logical qubit should be placed at each time step to minimize total decoherence while satisfying gate execution constraints.**

We formulate this as a sequential decision problem and use **Proximal Policy Optimization (PPO)** reinforcement learning to learn an optimal placement policy.

---

## Problem Formulation

### Input
- Quantum circuit represented as time-layered graph (parallel gates grouped into layers)
- Hardware constraints (compute capacity, noise parameters, movement costs)

### Optimization Goal
Minimize total noise = dwell noise + movement noise across circuit execution

### Constraints
- Compute region has limited capacity (max C qubits)
- Gates can only execute when qubits are in compute region
- Moving qubits between regions incurs noise and latency

### Approach
Train RL agent to learn placement policy that maps circuit state → qubit movement decisions

---

## Requirements

### Python Version
- Python 3.10+

### Dependencies
```
qiskit>=1.0.0          # Quantum circuit generation and manipulation
stable-baselines3>=2.2  # PPO reinforcement learning
gymnasium>=0.29         # RL environment interface
torch>=2.1              # Neural network backend
numpy>=1.24             # Numerical operations
matplotlib>=3.8         # Visualization
```

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
# venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Project Structure

```
quantum-rl-poc/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── test_install.py             # Verify installation works
│
├── src/
│   ├── circuit_generator.py   # Generate random quantum circuits and extract time layers
│   ├── qubit_env.py            # Gymnasium environment for qubit placement problem
│   ├── train.py                # Train PPO agent on placement task
│   └── evaluate.py             # Evaluate trained policy vs baselines
│
├── models/                      # Saved trained RL models
├── results/                     # Evaluation plots and metrics
└── venv/                        # Virtual environment (not committed)
```

### File Descriptions

#### Core Implementation

- **`src/circuit_generator.py`**: Converts Qiskit quantum circuits into time-layered representations where each layer contains gates that can execute in parallel.

- **`src/qubit_env.py`**: Custom Gymnasium environment that simulates qubit placement dynamics, enforces hardware constraints, and calculates noise-based rewards.

- **`src/train.py`**: Training pipeline that generates circuit datasets, trains PPO agent, and saves trained models.

- **`src/evaluate.py`**: Evaluation suite comparing trained RL policy against random and greedy baselines on test circuits.

#### Utilities

- **`test_install.py`**: Verification script to check all dependencies are installed correctly.

- **`requirements.txt`**: Pinned package versions for reproducibility.

---

## Quick Start

### 1. Verify Installation
```bash
python test_install.py
```
Expected output: `🎉 Setup complete! Ready to build POC.`

### 2. Generate and Test Circuits
```bash
cd src
python circuit_generator.py
```
Expected output: Prints circuit layers showing parallel gate execution structure

### 3. Test Environment
```bash
python qubit_env.py
```
Expected output: Simulates episode with random actions, prints noise accumulation

### 4. Train RL Agent
```bash
python train.py
```
Expected output: PPO training progress, saves model to `../models/`

### 5. Evaluate Results
```bash
python evaluate.py
```
Expected output: Comparison plot showing RL vs baselines saved to `../results/`

---

## Current Implementation (POC Phase)

### Simplifying Assumptions
- **Small scale**: 10 logical qubits, circuits with 15-25 layers
- **Uniform noise model**: Constant dwell noise per region, fixed movement cost
- **Instantaneous movement**: No transfer latency between regions
- **Compute capacity**: Fixed at 5 qubits maximum in compute region
- **Random circuits**: Training on randomly generated Clifford+T circuits

### What Works
✅ End-to-end RL training pipeline  
✅ Valid constraint enforcement (capacity, gate requirements)  
✅ PPO agent learns to reduce noise vs random baseline  
✅ Baseline comparisons (random, greedy policies)  
✅ Visualization and evaluation metrics  

### Next Steps (Incremental Complexity)
- [ ] Add transfer latency (movement takes multiple time steps)
- [ ] Implement gate-specific noise models (CNOT vs single-qubit)
- [ ] Scale to 50+ qubits with Graph Neural Network policy
- [ ] Test on structured circuits (QAOA, VQE, quantum chemistry)
- [ ] Compare against classical compiler optimization (Qiskit, TKET)

---

## Methodology

### 1. Circuit Representation
Quantum circuits are converted to time-layered directed acyclic graphs (DAGs) using Qiskit's transpiler. Each layer represents gates that can execute in parallel.

### 2. RL Formulation

**State**: 
- Current qubit locations (compute=1, storage=0)
- Active qubits in next layer
- Current layer index

**Action**: 
- Per-qubit binary decision: stay=0, move=1

**Reward**: 
- Negative noise (minimize)
- Large penalty for constraint violations (gates on qubits not in compute)

**Policy**: 
- Multi-layer perceptron (MLP) trained with PPO
- Maps state → action probabilities

### 3. Training
- Generate 100 random circuits (10 qubits, depth 15-25)
- Train PPO for 50k-100k timesteps
- Use experience replay and gradient clipping for stability

### 4. Evaluation
Compare trained policy against baselines:
- **Random**: Randomly move qubits each layer
- **Greedy**: Keep all qubits in compute (ignores noise optimization)

---

## Hardware Requirements

- **CPU**: Any modern processor (M1 Mac, Intel/AMD x86)
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: Not required (CPU training sufficient for POC scale)
- **Storage**: <1GB for code + models

**Training time**: 5-15 minutes on M1 MacBook Pro for POC configuration

---

## Research Context

This work addresses a gap in quantum compilation: **existing compilers optimize qubit routing and gate scheduling, but do not consider heterogeneous memory hierarchies.**

As quantum computers scale, architectural heterogeneity (fast compute vs slow storage qubits) will become essential for managing decoherence. This project explores RL-based compilation techniques for such architectures.

**Target venue**: Quantum computing architecture conferences (ISCA, MICRO, ASPLOS) or quantum compilation workshops (QCE, TQC)

---

## Contributing

This is a research project under active development. Contributions, suggestions, and discussions are welcome.

### Development Workflow
1. Create feature branch: `git checkout -b feature-name`
2. Make changes and test: `python test_install.py && python src/train.py`
3. Commit with clear message: `git commit -m "Add transfer latency model"`
4. Push and create pull request

---

## License

MIT License - see LICENSE file for details

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{quantum_rl_placement_2025,
  author = {[Your Name]},
  title = {RL-Based Quantum Qubit Placement Optimizer},
  year = {2025},
  url = {https://github.com/yourusername/quantum-rl-poc}
}
```

---

## Contact

- **Author**: [Your Name]
- **Email**: [your.email@university.edu]
- **GitHub**: [@yourusername](https://github.com/yourusername)

---

**Status**: Proof-of-concept phase (Phase 1 complete, scaling to realistic systems in progress)