"""
circuit_generator.py

Generates random quantum circuits and converts them to time-layered representation.
Each layer contains gates that can execute in parallel.
"""

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
import numpy as np


def generate_random_circuit_custom(n_qubits=10, depth=20, gate_density=0.3, seed=None):
    """
    Generate a custom random circuit with controllable sparsity.
    - This creates more realistic circuits where not all qubits are active every layer.
    
    Args:
        n_qubits: Number of qubits
        depth: Number of layers
        gate_density: Probability of gate on each qubit per layer (0.0 to 1.0)
        seed: Random seed
        
    Returns:
        QuantumCircuit
    """
    if seed is not None:
        np.random.seed(seed)
    
    qc = QuantumCircuit(n_qubits)
    
    for layer_idx in range(depth):
        # Randomly select qubits to be active this layer
        active_qubits = []
        for q in range(n_qubits):
            if np.random.random() < gate_density:
                active_qubits.append(q)
        
        # Add gates to active qubits
        np.random.shuffle(active_qubits)
        
        # Add 2-qubit gates (pairs of qubits)
        for i in range(0, len(active_qubits) - 1, 2):
            q1, q2 = active_qubits[i], active_qubits[i + 1]
            gate_type = np.random.choice(['cx', 'cz', 'swap'])
            
            if gate_type == 'cx':
                qc.cx(q1, q2)
            elif gate_type == 'cz':
                qc.cz(q1, q2)
            else:
                qc.swap(q1, q2)
        
        # Add single-qubit gate to any leftover qubit
        if len(active_qubits) % 2 == 1:
            q = active_qubits[-1]
            gate_type = np.random.choice(['h', 'x', 'y', 'z', 's', 't'])
            
            if gate_type == 'h':
                qc.h(q)
            elif gate_type == 'x':
                qc.x(q)
            elif gate_type == 'y':
                qc.y(q)
            elif gate_type == 'z':
                qc.z(q)
            elif gate_type == 's':
                qc.s(q)
            else:  # 't'
                qc.t(q)
        
        # Add barrier to force layer separation
        if active_qubits:
            qc.barrier(*active_qubits)
    
    return qc


def extract_circuit_layers(circuit):
    """
    Convert quantum circuit to time-layered representation.
    - Each layer contains gates that can execute in parallel (no qubit conflicts).
    
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

            # Skip barriers and other non-gate operations
            if node.op.name in ['barrier', 'measure', 'reset']:
                continue

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


def generate_layered_circuit(n_qubits=10, depth=20, gate_density=0.3, seed=None):
    """
    Complete pipeline: generate circuit and extract layers.
    
    Args:
        n_qubits: Number of qubits
        depth: Target circuit depth (number of layers)
        gate_density: Probability of gate per qubit per layer (0.0 to 1.0)
        seed: Random seed
        
    Returns:
        Dictionary with:
            - 'n_qubits': Number of qubits
            - 'n_layers': Number of time layers
            - 'layers': List of active qubits per layer
            - 'circuit': Original Qiskit circuit
    """
    # Generate circuit with controlled sparsity
    circuit = generate_random_circuit_custom(n_qubits, depth, gate_density, seed)
    
    # Extract time layers
    layers = extract_circuit_layers(circuit)
    
    return {
        'n_qubits': n_qubits,
        'n_layers': len(layers),
        'layers': layers,
        'circuit': circuit
    }


# Test code
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Circuit Generator")
    print("=" * 60)
    
    # Generate test circuit with 30% gate density
    circuit_data = generate_layered_circuit(
        n_qubits=10, 
        depth=20, 
        gate_density=0.3,  # 30% chance of gate per qubit
        seed=102
    )
    
    print(f"\n✓ Generated circuit successfully!")
    print(f"  - Qubits: {circuit_data['n_qubits']}")
    print(f"  - Layers: {circuit_data['n_layers']}")
    
    # Calculate statistics
    layer_sizes = [len(layer) for layer in circuit_data['layers']]
    avg_active = np.mean(layer_sizes)
    
    print(f"  - Avg active qubits per layer: {avg_active:.1f}")
    print(f"  - Min active: {min(layer_sizes)}, Max active: {max(layer_sizes)}")
    
    print(f"\n📊 Layer breakdown:")
    print("-" * 60)
    for i, active_qubits in enumerate(circuit_data['layers'][:15]):
        print(f"  Layer {i:2d}: Qubits {active_qubits} ({len(active_qubits)} active)")
    
    if circuit_data['n_layers'] > 15:
        print(f"  ... ({circuit_data['n_layers'] - 15} more layers)")
    
    print("\n" + "=" * 60)
    print("✅ Circuit generator working correctly!")
    print("=" * 60)
