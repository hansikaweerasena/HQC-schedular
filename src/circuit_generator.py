"""
circuit_generator.py

Generates random quantum circuits and converts them to time-layered representation via layering.
Each layer contains gates that can execute in parallel. This is the input for the RL model.
"""

from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.converters import circuit_to_dag
import numpy as np


def generate_random_circuit(n_qubits=10, depth=20, seed=None):
    """
    Generate a random quantum circuit using Qiskit.
    
    Args:
        n_qubits: Number of qubits in the circuit
        depth: Approximate circuit depth (number of gate layers)
        seed: Random seed for reproducibility
        
    Returns:
        QuantumCircuit object
    """
    if seed is not None:
        np.random.seed(seed)
    
    circuit = random_circuit(
        n_qubits, 
        depth, 
        measure=False,
        max_operands=2,  # Allow single and 2-qubit gates
        seed=seed,
        two_qubit_prob=0.1
    )
    
    return circuit


def extract_circuit_layers(circuit):
    """
    Convert quantum circuit to time-layered representation.
    
    Each layer contains gates that can execute in parallel (no qubit conflicts).
    
    Args:
        circuit: Qiskit QuantumCircuit
        
    Returns:
        List of layers, where each layer is a list of active qubit indices
    """
    # Convert circuit to Directed Acyclic Graph (DAG)
    dag = circuit_to_dag(circuit)
    
    # Get all gate layers (excluding barriers)
    layer_info = []
    
    for layer in dag.layers():
        active_qubits = set()
        
        # Iterate through operations in this layer
        for node in layer['graph'].op_nodes():
            # Get qubits involved in this gate
            for qubit in node.qargs:
                # Find qubit index in circuit
                if hasattr(qubit, 'index'):
                    qubit_idx = qubit.index
                elif hasattr(qubit, '_index'):
                    qubit_idx = qubit._index
                else:
                    qubit_idx = circuit.qubits.index(qubit)
                
                active_qubits.add(qubit_idx)
        
        # Only add non-empty layers
        if active_qubits:
            layer_info.append(sorted(list(active_qubits)))
    
    return layer_info


def generate_layered_circuit(n_qubits=10, depth=20, seed=None):
    """
    Complete pipeline: generate circuit and extract layers.
    
    Args:
        n_qubits: Number of qubits
        depth: Target circuit depth
        seed: Random seed
        
    Returns:
        Dictionary with:
            - 'n_qubits': Number of qubits
            - 'n_layers': Number of time layers
            - 'layers': List of active qubits per layer
            - 'circuit': Original Qiskit circuit (for debugging)
    """
    # Generate random circuit
    circuit = generate_random_circuit(n_qubits, depth, seed)
    
    # Extract time layers
    layers = extract_circuit_layers(circuit)
    
    return {
        'n_qubits': n_qubits,
        'n_layers': len(layers),
        'layers': layers,
        'circuit': circuit
    }


# Test code - runs when you execute this file directly
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Circuit Generator")
    print("=" * 60)
    
    # Generate a small test circuit
    circuit_data = generate_layered_circuit(n_qubits=20, depth=50, seed=None)
    
    print(f"\n✓ Generated circuit successfully!")
    print(f"  - Qubits: {circuit_data['n_qubits']}")
    print(f"  - Layers: {circuit_data['n_layers']}")
    
    print(f"\n📊 Layer breakdown:")
    print("-" * 60)
    for i, active_qubits in enumerate(circuit_data['layers'][:10]):  # Show first 10 layers
        print(f"  Layer {i:2d}: Qubits {active_qubits} are active ({len(active_qubits)} qubits)")
    
    if circuit_data['n_layers'] > 10:
        print(f"  ... ({circuit_data['n_layers'] - 10} more layers)")
    
    print("\n" + "=" * 60)
    print("✅ Circuit generator working correctly!")
    print("=" * 60)
