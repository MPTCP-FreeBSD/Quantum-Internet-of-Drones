import math
import time
import pandas as pd
from itertools import combinations
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, state_fidelity, partial_trace
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error, amplitude_damping_error, pauli_error, phase_damping_error
from qiskit.quantum_info import Kraus
import numpy as np
import itertools
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import (
    DensityMatrix, 
    state_fidelity, 
    partial_trace, 
    Kraus,
    random_unitary,
    Statevector
)
from qiskit_aer.noise import (
    NoiseModel, 
    depolarizing_error, 
    amplitude_damping_error, 
    thermal_relaxation_error,
    phase_damping_error,
    pauli_error
)
from typing import List, Tuple, Dict, Any


# Import your custom teleportation circuit builder
from utils.ibm_lab_util import build_qc  # Ensure this file and function exist

def init_qc():
    """Initialize teleportation circuit with input state."""
    qr = QuantumRegister(3, name="q")
    cr = ClassicalRegister(3, name="c")
    teleportation_circuit = build_qc(qr, cr)

    state_prep = QuantumCircuit(qr, cr)
    state_prep.rx(math.pi / 4, qr[0])
    state_prep.barrier()

    return state_prep.compose(teleportation_circuit), qr, cr, qr[2]



def create_pauli_crosstalk_kraus( strength: float, pauli_type: str = 'XX') -> Kraus:
    """Create Pauli-based crosstalk Kraus operators."""
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    
    pauli_dict = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
    
    if len(pauli_type) != 2:
        raise ValueError("pauli_type must be 2-character string like 'XX', 'XY', etc.")
        
    pauli1, pauli2 = pauli_type[0], pauli_type[1]
    pauli_op = np.kron(pauli_dict[pauli1], pauli_dict[pauli2])
    
    K0 = np.sqrt(1 - strength) * np.eye(4)
    K1 = np.sqrt(strength) * pauli_op
    
    # Ensure CPTP condition
    norm_check = K0.conj().T @ K0 + K1.conj().T @ K1
    if not np.allclose(norm_check, np.eye(4), atol=1e-10):
        # Renormalize
        total_norm = np.trace(norm_check)
        K0 = K0 / np.sqrt(total_norm)
        K1 = K1 / np.sqrt(total_norm)
        
    return Kraus([K0, K1])

def create_zz_crosstalk_kraus( strength: float, coupling_angle: float = 0.0) -> Kraus:
    """Create ZZ-coupling crosstalk (common in superconducting qubits)."""
    # ZZ interaction: exp(-i * strength * Z⊗Z * coupling_angle)
    ZZ = np.kron(np.array([[1, 0], [0, -1]]), np.array([[1, 0], [0, -1]]))
    
    # Unitary evolution under ZZ coupling
    zz_unitary = np.array([
        [np.exp(-1j * strength * coupling_angle), 0, 0, 0],
        [0, np.exp(1j * strength * coupling_angle), 0, 0],
        [0, 0, np.exp(1j * strength * coupling_angle), 0],
        [0, 0, 0, np.exp(-1j * strength * coupling_angle)]
    ])
    
    # Convert to Kraus (unitary channel has single Kraus operator)
    return Kraus([zz_unitary])

def create_amplitude_phase_crosstalk_error( amp_strength: float, phase_strength: float):
    """Create combined amplitude and phase crosstalk using Qiskit's built-in error models."""
    # Use Qiskit's built-in amplitude and phase damping errors
    amp_error = amplitude_damping_error(amp_strength)
    phase_error = phase_damping_error(phase_strength)
    
    # For 2-qubit crosstalk, we'll create correlated errors
    # First, create single-qubit errors for each qubit
    amp_error_2q = amp_error.tensor(amp_error)  # Independent amplitude damping on both qubits
    phase_error_2q = phase_error.tensor(phase_error)  # Independent phase damping on both qubits
    
    # Combine them (this creates a mixed error model)
    combined_error = amp_error_2q.compose(phase_error_2q)
    
    return combined_error

def create_depolarizing_crosstalk_error( strength: float):
    """Create depolarizing crosstalk error."""
    return depolarizing_error(strength, 2)

def create_random_crosstalk_kraus( strength: float, num_operators: int = 4) -> Kraus:
    """Create random crosstalk using random unitary matrices."""
    kraus_ops = []
    remaining_strength = 1.0
    
    for i in range(num_operators - 1):
        # Random strength for this operator
        op_strength = np.random.uniform(0, remaining_strength * strength)
        remaining_strength -= op_strength / strength
        
        # Random 4x4 unitary
        random_u = random_unitary(4).data
        kraus_ops.append(np.sqrt(op_strength) * random_u)
    
    # Last operator gets remaining strength
    final_strength = remaining_strength * strength
    if final_strength > 0:
        random_u = random_unitary(4).data
        kraus_ops.append(np.sqrt(final_strength) * random_u)
    
    # Identity component
    kraus_ops.insert(0, np.sqrt(1 - strength) * np.eye(4))
    
    return Kraus(kraus_ops)

def build_crosstalk_noise_model(crosstalk_config):
    """
    Build noise model with configurable crosstalk on specified qubit pairs.
    
    Args:
        crosstalk_config: Dictionary with crosstalk parameters
    """
    noise_model = NoiseModel()
    
    # Base noise parameters
    base_1q_error = crosstalk_config.get('base_1q_error', 0.001)
    base_2q_error = crosstalk_config.get('base_2q_error', 0.01)
    thermal_t1 = crosstalk_config.get('t1_time', 50e-6)
    thermal_t2 = crosstalk_config.get('t2_time', 30e-6)
    gate_time = crosstalk_config.get('gate_time', 100e-9)
    
    # Single-qubit errors
    if base_1q_error > 0:
        depol_1q = depolarizing_error(base_1q_error, 1)
        thermal_1q = thermal_relaxation_error(thermal_t1, thermal_t2, gate_time)
        combined_1q = depol_1q.compose(thermal_1q)
        noise_model.add_all_qubit_quantum_error(combined_1q, ['x', 'y', 'z', 'h', 'rx', 'ry', 'rz'])
    
    # Base two-qubit errors
    if base_2q_error > 0:
        depol_2q = depolarizing_error(base_2q_error, 2)
        noise_model.add_all_qubit_quantum_error(depol_2q, ['cx', 'cz'])
    
    # Add crosstalk to specified pairs
    crosstalk_type = crosstalk_config.get('type', 'pauli')
    crosstalk_strength = crosstalk_config.get('strength', 0.01)
    

    if crosstalk_type == 'pauli':
        pauli_type = crosstalk_config.get('pauli_type', 'XX')
        crosstalk_kraus = create_pauli_crosstalk_kraus(crosstalk_strength, pauli_type)
    elif crosstalk_type == 'zz':
        coupling_angle = crosstalk_config.get('coupling_angle', np.pi/4)
        crosstalk_kraus = create_zz_crosstalk_kraus(crosstalk_strength, coupling_angle)
    elif crosstalk_type == 'amp_phase':
        # Use the new method that returns a proper Qiskit error
        amp_strength = crosstalk_config.get('amp_strength', crosstalk_strength/2)
        phase_strength = crosstalk_config.get('phase_strength', crosstalk_strength/2)
        crosstalk_error = create_amplitude_phase_crosstalk_error(amp_strength, phase_strength)
        # Apply the error directly
        noise_model.add_all_qubit_quantum_error(crosstalk_error, ['cx', 'cz'])
        print(f"  → Applied {crosstalk_type} crosstalk (amp={amp_strength:.4f}, phase={phase_strength:.4f}) to all qubits")

    elif crosstalk_type == 'depolarizing':
        crosstalk_error = create_depolarizing_crosstalk_error(crosstalk_strength)
        noise_model.add_all_qubit_quantum_error(crosstalk_error, ['cx', 'cz'])
        print(f"  → Applied {crosstalk_type} crosstalk (strength={crosstalk_strength:.4f}) to all qubits")

    elif crosstalk_type == 'random':
        num_ops = crosstalk_config.get('num_operators', 4)
        crosstalk_kraus = create_random_crosstalk_kraus(crosstalk_strength, num_ops)
    else:
        raise ValueError(f"Unknown crosstalk type: {crosstalk_type}")
    
    # Apply crosstalk to 2-qubit gates on this pair (only for Kraus-based methods)
    if crosstalk_type in ['pauli', 'zz', 'random']:
        noise_model.add_all_qubit_quantum_error(crosstalk_kraus, ['cx', 'cz'],)
        print(f"  → Applied {crosstalk_type} crosstalk (strength={crosstalk_strength:.4f}) to all qubits")
    
    return noise_model


def compute_throughput(fidelity, teleportation_time, shots):
    raw_throughput = shots / teleportation_time
    effective_throughput = fidelity * raw_throughput
    return raw_throughput, effective_throughput

# # Example
# shots = 1000
# teleportation_time = 10e-6  # seconds
# fidelity = 0.92

# raw, effective = compute_throughput(fidelity, teleportation_time, shots)
# print(f"Raw Throughput:       {raw:.2e} qubits/sec")
# print(f"Effective Throughput: {effective:.2e} qubits/sec")


def evaluate_teleportation_segment(noise_model, shots=1000):
    """Run teleportation with noise and return fidelity, latency, and throughput."""
    # Step 1: Get teleportation circuit
    teleport_circuit, qr, cr, b = init_qc()
    teleport_circuit.draw("mpl", cregbundle=False)

    # Step 2: Simulators
    ideal_sim = AerSimulator(method="density_matrix")
    noisy_sim = AerSimulator(noise_model=noise_model, method="density_matrix")
    
    # Save statevectors
    ideal_circuit = teleport_circuit.copy()


    noisy_circuit = transpile(teleport_circuit, noisy_sim)


    # new - valid for density_matrix method
    ideal_circuit.save_density_matrix()
    noisy_circuit.save_density_matrix()

    ideal_result = ideal_sim.run(ideal_circuit).result()
    noisy_result = noisy_sim.run(noisy_circuit).result()

    from qiskit.quantum_info import DensityMatrix

    ideal_dm = DensityMatrix(ideal_result.data(0)['density_matrix'])
    noisy_dm = DensityMatrix(noisy_result.data(0)['density_matrix'])


    ideal_result = ideal_sim.run(ideal_circuit).result()
    noisy_result = noisy_sim.run(noisy_circuit).result()


    # Step 3: Fidelity
    # Step 3: Fidelity (using partial trace on density matrices)
    ideal_b = partial_trace(ideal_dm, [0, 1])
    noisy_b = partial_trace(noisy_dm, [0, 1])

    fidelity_b = state_fidelity(ideal_b, noisy_b)
    fidelity_full = state_fidelity(ideal_dm, noisy_dm)


    # Step 4: Latency and Throughput
    measured_circuit = teleport_circuit.copy()
    measured_circuit.measure_all()

    start = time.time()
    result = noisy_sim.run(measured_circuit, shots=shots).result()
    end = time.time()

    teleportation_time = 10e-6  # 10 microseconds
    throughput = shots / teleportation_time

    raw, effective = compute_throughput(fidelity_full, teleportation_time, shots)

    return fidelity_b, fidelity_full, teleportation_time, raw, effective

# -------------------------------
# Main Execution
# -------------------------------

def main():
    crosstalk_strengths=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3, 0.5, 0.8, 1]
    crosstalk_types=['amp_phase', 'pauli', 'depolarizing']
    results=[]
    
    for crosstalk_strength in crosstalk_strengths:
        for crosstalk_type in crosstalk_types:

            print("crosstalk_strength",crosstalk_strength)
            print("crosstalk_type",crosstalk_type)

            crosstalk_config = {
                'type': crosstalk_type,
                'strength': crosstalk_strength,
                'base_1q_error': 0.001,
                'base_2q_error': 0.005,
                'pauli_type': 'XX',  # for Pauli crosstalk
                'coupling_angle': np.pi/4,  # for ZZ crosstalk
                'amp_strength': crosstalk_strength/2,  # for amp_phase crosstalk
                'phase_strength': crosstalk_strength/2
            }
            noise_model = build_crosstalk_noise_model(crosstalk_config)

            fidelity_b, fidelity_full, latency, raw, effective = evaluate_teleportation_segment(noise_model)

            results.append({
                'fidelity_qubit_b_only': fidelity_b,
                'fidelity_full_state': fidelity_full,
                'latency_sec': latency,
                'raw_throughput_qubits_per_sec': raw,
                'effective_throughput_qubits_per_sec': effective,
            })

            print(f"  → Fidelity(B): {fidelity_b:.4f}, Full: {fidelity_full:.4f}, Latency: {latency:.4e}s, Raw Throughput: {raw:.2e} q/s, Effective Throughput: {effective:.2e} q/s")

    df = pd.DataFrame(results)
    df.to_csv("quantum_network_metrics.csv", index=False)
    df.to_excel("quantum_network_metrics.xlsx", index=False)
    print("\n✅ Results saved to CSV and Excel.")

if __name__ == "__main__":
    main()
