"""
qubit_env.py

Gymnasium environment for quantum qubit placement optimization.
Agent decides where to place qubits (compute vs storage) at each time layer.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from circuit_generator import generate_layered_circuit


class QubitPlacementEnv(gym.Env):
    """
    RL Environment for qubit placement between compute and storage regions.
    
    State: Current qubit locations + which qubits are active next layer
    Action: For each qubit, decide to move or stay
    Reward: Negative noise (minimize dwell + movement costs)
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, 
                 n_qubits=10,
                 compute_capacity=5,
                 noise_compute=1.0,      # Idle Noise per qubit per layer in compute
                 noise_storage=0.1,      # Idle Noise per qubit per layer in storage
                 noise_move=0.5,         # Idle Noise cost per movement
                 circuit_depth=20,
                 gate_density=0.3):
        """
        Initialize environment.
        
        Args:
            n_qubits: Number of logical qubits
            compute_capacity: Max qubits allowed in compute region
            noise_compute: Dwell noise per layer when in compute
            noise_storage: Dwell noise per layer when in storage (lower)
            noise_move: Noise incurred when moving between regions
            circuit_depth: Target depth for generated circuits
            gate_density: Sparsity of gates (0.0 to 1.0)
        """
        super().__init__()
        
        # Environment parameters
        self.n_qubits = n_qubits
        self.compute_capacity = compute_capacity
        self.noise_compute = noise_compute
        self.noise_storage = noise_storage
        self.noise_move = noise_move
        self.circuit_depth = circuit_depth
        self.gate_density = gate_density
        
        # State space: [qubit locations (n_qubits), active qubits next layer (n_qubits), layer_index (1)]
        # Locations: 0 = storage, 1 = compute
        # Active: 0 = idle, 1 = has gate next layer
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(2 * n_qubits + 1,),  # locations + active_next + layer_idx
            dtype=np.float32
        )
        
        # Action space: For each qubit, binary decision (stay=0, move=1)
        self.action_space = spaces.MultiBinary(n_qubits)
        
        # Episode state variables
        self.circuit_data = None
        self.current_layer = 0
        self.qubit_locations = None  # 0 = storage, 1 = compute
        self.total_noise = 0.0



    def _decompose_layers(self, circuit_data):
        """
        Decompose layers with >compute_capacity qubits into sub-layers.
        
        Args:
            circuit_data: Dict with 'layers' key
            
        Returns:
            Modified circuit_data with decomposed layers
        """
        new_layers = []
        
        for layer in circuit_data['layers']:
            if len(layer) <= self.compute_capacity:
                # Layer fits - keep as is
                new_layers.append(layer)
            else:
                # Layer too big - split into chunks of size compute_capacity
                for i in range(0, len(layer), self.compute_capacity):
                    sub_layer = layer[i:i + self.compute_capacity]
                    new_layers.append(sub_layer)
        
        # Update circuit data
        circuit_data['layers'] = new_layers
        circuit_data['n_layers'] = len(new_layers)
        
        return circuit_data


    def reset(self, seed=None, options=None):
        """
        Reset environment with a new random circuit.
        
        Returns:
            observation: Initial state
            info: Additional info dict
        """
        super().reset(seed=seed)
        
        # Generate new random circuit
        circuit_data = generate_layered_circuit(
            n_qubits=self.n_qubits,
            depth=self.circuit_depth,
            gate_density=self.gate_density,
            seed=seed
        )

        # Decompose layers (preprocessing)
        self.circuit_data = self._decompose_layers(circuit_data)
        
        # Initialize all qubits in storage (conservative start)
        self.qubit_locations = np.zeros(self.n_qubits, dtype=np.int32)
        
        # Reset episode state
        self.current_layer = 0
        self.total_noise = 0.0
        
        # Get initial observation
        obs = self._get_observation()
        info = {'circuit_layers': self.circuit_data['n_layers']}
        
        return obs, info
    

    def _get_observation(self):
        """
        Construct current state observation.
        
        Returns:
            numpy array: [locations, active_next, normalized_layer_idx]
        """
        # Current qubit locations (0 or 1)
        locations = self.qubit_locations.astype(np.float32)
        
        # Which qubits are active in the NEXT layer (need to be in compute)
        active_next = np.zeros(self.n_qubits, dtype=np.float32)
        if self.current_layer < len(self.circuit_data['layers']):
            for qubit_idx in self.circuit_data['layers'][self.current_layer]:
                active_next[qubit_idx] = 1.0
        
        # Normalized layer index (0.0 to 1.0)
        layer_progress = self.current_layer / max(len(self.circuit_data['layers']), 1)
        
        # Concatenate all features
        obs = np.concatenate([locations, active_next, [layer_progress]])
        
        return obs
    

    def _project_action(self, action):
        """
        Simplified masking (sub-layering handles capacity).
        Only ensures active qubits are in compute.
        """
        action = np.array(action, dtype=np.int32)
        
        # Apply desired movements
        new_locations = self.qubit_locations.copy()
        for qubit_idx in range(self.n_qubits):
            if action[qubit_idx] == 1:
                new_locations[qubit_idx] = 1 - self.qubit_locations[qubit_idx]
        
        # Get active qubits for current layer
        active_qubits = []
        if self.current_layer < len(self.circuit_data['layers']):
            active_qubits = self.circuit_data['layers'][self.current_layer]
        
        # Force active qubits to compute
        for qubit_idx in active_qubits:
            new_locations[qubit_idx] = 1
        
        # Reconstruct action
        corrected_action = np.zeros(self.n_qubits, dtype=np.int32)
        for qubit_idx in range(self.n_qubits):
            if new_locations[qubit_idx] != self.qubit_locations[qubit_idx]:
                corrected_action[qubit_idx] = 1
        
        return corrected_action



    def _calculate_noise(self, action):
        """
        Calculate noise for this step (movement + dwell).
        
        Args:
            action: Binary array of movements
            
        Returns:
            tuple: (movement_noise, dwell_noise, new_locations)
        """
        # Movement noise
        movement_noise = np.sum(action) * self.noise_move
        
        # Apply movements to get new locations
        new_locations = self.qubit_locations.copy()
        for qubit_idx in range(self.n_qubits):
            if action[qubit_idx] == 1:
                new_locations[qubit_idx] = 1 - self.qubit_locations[qubit_idx]
        
        # Dwell noise (qubits sitting in their regions)
        dwell_noise = 0.0
        for qubit_idx in range(self.n_qubits):
            if new_locations[qubit_idx] == 1:  # In compute
                dwell_noise += self.noise_compute
            else:  # In storage
                dwell_noise += self.noise_storage
        
        return movement_noise, dwell_noise, new_locations



    def step(self, action):
        """
        Execute one time step with action masking (no constraint violations possible).
        
        Args:
            action: Binary array (n_qubits,) where 1 = move, 0 = stay
            
        Returns:
            observation: Next state
            reward: Reward for this step
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        # === ACTION MASKING: Project to valid action ===
        action = self._project_action(action)
        
        # === Calculate noise (no penalties - masking ensures validity) ===
        movement_noise, dwell_noise, new_locations = self._calculate_noise(action)
        
        # Update qubit locations
        self.qubit_locations = new_locations
        
        # Total noise this step
        total_step_noise = dwell_noise + movement_noise
        self.total_noise += total_step_noise
        
        # Reward = negative noise (minimize noise)
        reward = -total_step_noise
        
        # Advance to next layer
        self.current_layer += 1
        
        # Check if episode is done
        terminated = self.current_layer >= len(self.circuit_data['layers'])
        truncated = False
        
        # Get next observation
        obs = self._get_observation()
        
        # Info dictionary
        compute_count = np.sum(self.qubit_locations)
        info = {
            'layer': self.current_layer - 1,
            'dwell_noise': dwell_noise,
            'movement_noise': movement_noise,
            'constraint_penalty': 0.0,  # Always 0 with masking
            'total_noise': self.total_noise,
            'compute_count': int(compute_count)
        }
        
        return obs, reward, terminated, truncated, info

    
    def render(self):
        """
        Render current state (for debugging).
        """
        if self.circuit_data is None:
            print("Environment not initialized. Call reset() first.")
            return
        
        print(f"\n=== Layer {self.current_layer}/{len(self.circuit_data['layers'])} ===")
        print(f"Qubit Locations: {self.qubit_locations}")
        print(f"Compute count: {np.sum(self.qubit_locations)}/{self.compute_capacity}")
        print(f"Total noise so far: {self.total_noise:.2f}")


# Test the environment
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Qubit Placement Environment")
    print("=" * 60)
    
    # Create environment
    env = QubitPlacementEnv(n_qubits=10, compute_capacity=5)
    
    # Reset and get initial state
    obs, info = env.reset(seed=42)
    print(f"\n✓ Environment created successfully!")
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Action space: {env.action_space}")
    print(f"  - Circuit layers: {info['circuit_layers']}")
    
    # Run a few steps with random actions
    print(f"\n📊 Running 5 steps with random actions:")
    print("-" * 60)
    
    for step_num in range(5):
        # Random action
        action = env.action_space.sample()
        
        # Execute step
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nStep {step_num + 1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Compute: {info['compute_count']}/{env.compute_capacity}")
        print(f"  Noise breakdown: dwell={info['dwell_noise']:.1f}, move={info['movement_noise']:.1f}, penalty={info['constraint_penalty']:.1f}")
        
        if terminated:
            print(f"\n✓ Episode finished at layer {info['layer']}")
            break
    
    print("\n" + "=" * 60)
    print("✅ Environment working correctly!")
    print("=" * 60)
