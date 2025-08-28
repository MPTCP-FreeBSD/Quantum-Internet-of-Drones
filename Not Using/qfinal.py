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

# No changes needed for these helper functions for creating Kraus/Error objects
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
    
    return Kraus([K0, K1])

def create_zz_crosstalk_kraus( strength: float, coupling_angle: float = 0.0) -> Kraus:
    """Create ZZ-coupling crosstalk (common in superconducting qubits)."""
    theta = strength * coupling_angle
    
    zz_unitary = np.array([
        [np.exp(-1j * theta), 0, 0, 0],
        [0, np.exp(1j * theta), 0, 0],
        [0, 0, np.exp(1j * theta), 0],
        [0, 0, 0, np.exp(-1j * theta)]
    ])
    
    return Kraus([zz_unitary])

def create_amplitude_phase_crosstalk_error( amp_strength: float, phase_strength: float):
    """Create combined amplitude and phase crosstalk using Qiskit's built-in error models."""
    amp_error_1q = amplitude_damping_error(amp_strength)
    phase_error_1q = phase_damping_error(phase_strength)
    
    amp_error_2q = amp_error_1q.tensor(amp_error_1q)
    phase_error_2q = phase_error_1q.tensor(phase_error_1q)
    
    combined_error = amp_error_2q.compose(phase_error_2q)
    
    return combined_error

def create_depolarizing_crosstalk_error( strength: float):
    """Create depolarizing crosstalk error."""
    return depolarizing_error(strength, 2)

def create_random_crosstalk_kraus( strength: float, num_operators: int = 4) -> Kraus:
    """Create random crosstalk using random unitary matrices."""
    kraus_ops = []
    
    random_probs = np.random.dirichlet(np.ones(num_operators - 1)) * strength
    
    for prob in random_probs:
        random_u = random_unitary(4).data
        kraus_ops.append(np.sqrt(prob) * random_u)
    
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
    
    # --- Single-qubit errors (Depolarizing and Thermal Relaxation) ---
    # Applied to ALL single-qubit gates on ALL qubits (q[0], q[1], q[2])
    # Why: Every qubit interacts with its environment, leading to decoherence.
    # Depolarizing error models information loss due to random mixing with the maximally mixed state.
    # Thermal relaxation models energy decay (T1) and dephasing (T2) over time.
    # These errors affect gates like Hadamard (h) on qr[0] and qr[1], Rx on qr[0],
    # and conditional X/Z gates on qr[2].
    if base_1q_error > 0:
        depol_1q = depolarizing_error(base_1q_error, 1)
        thermal_1q = thermal_relaxation_error(thermal_t1, thermal_t2, gate_time)
        combined_1q = depol_1q.compose(thermal_1q)
        noise_model.add_all_qubit_quantum_error(combined_1q, ['x', 'y', 'z', 'h', 'rx', 'ry', 'rz'])
        print(f"  → Applied base 1-qubit depolarizing ({base_1q_error:.4f}) and thermal relaxation (T1={thermal_t1*1e6}us, T2={thermal_t2*1e6}us) to ALL single-qubit gates on all qubits.")
    
    # --- Base two-qubit errors (Depolarizing) ---
    # Applied to ALL two-qubit gates on ALL valid qubit pairs.
    # In this teleportation circuit, these are the CNOT gates: cx(qr[1], qr[2]) and cx(qr[0], qr[1]).
    # Why: Two-qubit gates are more complex and resource-intensive, making them inherently more prone to errors
    # due to imperfections in control pulses and increased susceptibility to environmental noise during their operation.
    if base_2q_error > 0:
        depol_2q = depolarizing_error(base_2q_error, 2)
        noise_model.add_all_qubit_quantum_error(depol_2q, ['cx', 'cz'])
        print(f"  → Applied base 2-qubit depolarizing ({base_2q_error:.4f}) to ALL two-qubit gates (cx, cz).")
    
    # --- Crosstalk Noise ---
    # These errors model unintended interactions between qubits.
    # Applied as 2-qubit errors to ALL two-qubit gates (cx, cz) in the circuit.
    # Why: Crosstalk often manifests during multi-qubit gate operations. When a gate is performed on
    # one pair of qubits (e.g., qr[0] and qr[1]), the control signals or parasitic physical couplings
    # might inadvertently affect the state of a nearby qubit (e.g., qr[2]) or introduce correlated errors
    # between the qubits being operated on. By applying these errors to all CX gates, we simulate a general
    # presence of such correlated errors during the most interactive parts of the circuit.
    
    crosstalk_type = crosstalk_config.get('type', 'pauli')
    crosstalk_strength = crosstalk_config.get('strength', 0.01)
    
    crosstalk_error_obj = None

    if crosstalk_type == 'pauli':
        pauli_type = crosstalk_config.get('pauli_type', 'XX')
        crosstalk_error_obj = create_pauli_crosstalk_kraus(crosstalk_strength, pauli_type)
        print(f"  → Created {crosstalk_type} crosstalk (strength={crosstalk_strength:.4f}, type={pauli_type}).")
    elif crosstalk_type == 'zz':
        coupling_angle = crosstalk_config.get('coupling_angle', np.pi/4)
        crosstalk_error_obj = create_zz_crosstalk_kraus(crosstalk_strength, coupling_angle)
        print(f"  → Created {crosstalk_type} crosstalk (strength={crosstalk_strength:.4f}, angle={coupling_angle:.4f} rad).")
    elif crosstalk_type == 'amp_phase':
        amp_strength = crosstalk_config.get('amp_strength', crosstalk_strength/2)
        phase_strength = crosstalk_config.get('phase_strength', crosstalk_strength/2)
        crosstalk_error_obj = create_amplitude_phase_crosstalk_error(amp_strength, phase_strength)
        print(f"  → Created combined amplitude/phase crosstalk (amp={amp_strength:.4f}, phase={phase_strength:.4f}).")
    elif crosstalk_type == 'depolarizing':
        crosstalk_error_obj = create_depolarizing_crosstalk_error(crosstalk_strength)
        print(f"  → Created 2-qubit {crosstalk_type} crosstalk (strength={crosstalk_strength:.4f}).")
    elif crosstalk_type == 'random':
        num_ops = crosstalk_config.get('num_operators', 4)
        crosstalk_error_obj = create_random_crosstalk_kraus(crosstalk_strength, num_ops)
        print(f"  → Created {crosstalk_type} crosstalk (strength={crosstalk_strength:.4f}, operators={num_ops}).")
    else:
        raise ValueError(f"Unknown crosstalk type: {crosstalk_type}")
    
    # Apply the generated crosstalk error to ALL two-qubit gates (CX, CZ)
    if crosstalk_error_obj:
        noise_model.add_all_qubit_quantum_error(crosstalk_error_obj, ['cx', 'cz'])
        print(f"  → Applied {crosstalk_type} crosstalk error to all 'cx' and 'cz' gates in the circuit.")
    
    return noise_model


def compute_throughput(fidelity, teleportation_time, shots):
    raw_throughput = shots / teleportation_time
    effective_throughput = fidelity * raw_throughput
    return raw_throughput, effective_throughput

def calculate_quantum_throughput(fidelity, gate_time_us=0.1, readout_time_us=1.0, 
                                fidelity_threshold=0.95, circuit_depth=4):
    """
    Calculate quantum throughput metrics.
    
    Args:
        fidelity: State fidelity after crosstalk
        gate_time_us: Gate execution time in microseconds
        readout_time_us: Readout time in microseconds
        fidelity_threshold: Minimum acceptable fidelity
        circuit_depth: Number of gates in circuit
    
    Returns:
        Dictionary with throughput metrics
    """
    # Basic timing
    total_time_us = circuit_depth * gate_time_us + readout_time_us
    
    # Effective operations per second (only counting successful operations)
    if fidelity >= fidelity_threshold:
        success_rate = fidelity  # Simplified model
        effective_ops_per_sec = (1e6 / total_time_us) * success_rate
    else:
        effective_ops_per_sec = 0  # Below threshold
    
    # Quantum volume approximation (simplified)
    quantum_volume = min(circuit_depth, 2**circuit_depth * fidelity)
    
    # Information throughput (bits/second)
    # Assumes each operation processes log2(circuit_depth) bits of quantum information
    info_throughput = effective_ops_per_sec * np.log2(max(circuit_depth, 2))
    
    # Error-corrected throughput (accounts for error correction overhead)
    if fidelity > 0.999:
        ec_overhead = 1.1  # Minimal overhead for high fidelity
    elif fidelity > 0.99:
        ec_overhead = 5.0  # Moderate overhead
    elif fidelity > 0.95:
        ec_overhead = 50.0  # High overhead
    else:
        ec_overhead = np.inf  # Error correction impossible
    
    ec_throughput = effective_ops_per_sec / ec_overhead if ec_overhead != np.inf else 0
    
    return {
        'total_time_us': total_time_us,
        'raw_ops_per_sec': 1e6 / total_time_us,
        'effective_ops_per_sec': effective_ops_per_sec,
        'quantum_volume': quantum_volume,
        'info_throughput_bits_per_sec': info_throughput,
        'error_corrected_throughput': ec_throughput,
        'success_rate': fidelity if fidelity >= fidelity_threshold else 0
    }

def evaluate_teleportation_segment(noise_model, shots=1000):
    """Run teleportation with noise and return fidelity, latency, and throughput."""
    # Step 1: Get teleportation circuit
    teleport_circuit, qr, cr, b = init_qc() # b is now qr[2]

    # Step 2: Simulators
    ideal_sim = AerSimulator(method="density_matrix")
    noisy_sim = AerSimulator(noise_model=noise_model, method="density_matrix")
    
    ideal_circuit_dm = teleport_circuit.copy()
    noisy_circuit_dm = teleport_circuit.copy()
    
    ideal_circuit_dm.save_density_matrix()
    noisy_circuit_dm.save_density_matrix()

    noisy_circuit_dm = transpile(noisy_circuit_dm, noisy_sim)

    ideal_result_dm = ideal_sim.run(ideal_circuit_dm).result()
    noisy_result_dm = noisy_sim.run(noisy_circuit_dm).result()

    ideal_dm = DensityMatrix(ideal_result_dm.data(0)['density_matrix'])
    noisy_dm = DensityMatrix(noisy_result_dm.data(0)['density_matrix'])

    # Step 3: Fidelity (using partial trace on density matrices)
    # The target qubit is qr[2] (qubit 'b').
    # We need to trace out qubits qr[0] ('s') and qr[1] ('a').
    # The `partial_trace` function takes the indices of qubits to *keep*.
    # Alternatively, it takes the indices of qubits to *trace out*.
    # If the input DM is 3-qubit, and we want qr[2] (index 2), we trace out [0, 1].
    ideal_b = partial_trace(ideal_dm, [0, 1]) # Trace out qr[0] and qr[1]
    noisy_b = partial_trace(noisy_dm, [0, 1]) # Trace out qr[0] and qr[1]

    fidelity_b = state_fidelity(ideal_b, noisy_b)
    fidelity_full = state_fidelity(ideal_dm, noisy_dm) # Fidelity of the full 3-qubit state

    # Step 4: Latency and Throughput
    teleportation_time = 10e-6  # Fixed 10 microseconds

    raw, effective = compute_throughput(fidelity_full, teleportation_time, shots)
    result_thrpt = calculate_quantum_throughput(fidelity_full)

    raw = result_thrpt['raw_ops_per_sec']
    effective = result_thrpt['effective_ops_per_sec']

    return fidelity_b, fidelity_full, teleportation_time, raw, effective

# -------------------------------
# Main Execution
# -------------------------------

def main():
    crosstalk_strengths=[0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,  0.3, 0.5, 0.8, 1]
    crosstalk_types=['amp_phase', 'pauli', 'depolarizing', 'zz', 'random'] 
    # crosstalk_types=['depolarizing'] 
    results=[]
    
    print("Starting teleportation simulation with various crosstalk noise models...")

    for crosstalk_strength in crosstalk_strengths:
        for crosstalk_type in crosstalk_types:
            print(f"\nSimulating with: Crosstalk Strength = {crosstalk_strength}, Type = {crosstalk_type}")

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

            shots_for_throughput = 1000 
            fidelity_b, fidelity_full, latency, raw, effective = evaluate_teleportation_segment(
                noise_model, shots=shots_for_throughput
            )

            results.append({
                'crosstalk_type': crosstalk_type,
                'crosstalk_strength': crosstalk_strength,
                'fidelity_qubit_b_only': fidelity_b,
                'fidelity_full_state': fidelity_full,
                'latency_sec': latency,
                'raw_throughput_qubits_per_sec': raw,
                'effective_throughput_qubits_per_sec': effective,
            })

            print(f"  → Results: Fidelity(B): {fidelity_b:.4f}, Full: {fidelity_full:.4f}, Latency: {latency:.4e}s, Raw Throughput: {raw:.2e} q/s, Effective Throughput: {effective:.2e} q/s")

    df = pd.DataFrame(results)
    df.to_csv("quantum_network_metrics_refined.csv", index=False)
    df.to_excel("quantum_network_metrics_refined.xlsx", index=False)
    print("\n✅ Results saved to CSV and Excel.")




def compute(cross_strength,network_topology_type):
    crosstalk_strengths=[cross_strength]
    crosstalk_types=['amp_phase', 'pauli', 'depolarizing', 'zz', 'random'] 
    # crosstalk_types=['depolarizing'] 
    results=[]
    
    print("Starting teleportation simulation with various crosstalk noise models...")

    for crosstalk_strength in crosstalk_strengths:
        for crosstalk_type in crosstalk_types:
            print(f"\nSimulating with: Crosstalk Strength = {crosstalk_strength}, Type = {crosstalk_type}")

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

            shots_for_throughput = 1000 
            fidelity_b, fidelity_full, latency, raw, effective = evaluate_teleportation_segment(
                noise_model, shots=shots_for_throughput
            )

            results.append({
                'crosstalk_type': crosstalk_type,
                'crosstalk_strength': crosstalk_strength,
                'fidelity_qubit_b_only': fidelity_b,
                'fidelity_full_state': fidelity_full,
                'latency_sec': latency,
                'raw_throughput_qubits_per_sec': raw,
                'effective_throughput_qubits_per_sec': effective,
            })

            print(f"  → Results: Fidelity(B): {fidelity_b:.4f}, Full: {fidelity_full:.4f}, Latency: {latency:.4e}s, Raw Throughput: {raw:.2e} q/s, Effective Throughput: {effective:.2e} q/s")

    df = pd.DataFrame(results)
    df.to_csv(f"quantum_network_metrics_refined_{network_topology_type}.csv", index=False)
    df.to_excel(f"quantum_network_metrics_refined_{network_topology_type}.xlsx", index=False)
    print("\n✅ Results saved to CSV and Excel.")

if __name__ == "__main__":
    main()
    
    net_crosstalk = {"Bus":0.467,
                     "Ring":0.167,
                     "Star":0.267,
                     "Mesh":0}
    
    for topology, ct_str in net_crosstalk.items():
        print(f"Topology: {topology}, Crosstalk: {ct_str}")
    



    import matplotlib.pyplot as plt

    def plot_results(network_topology_type):
        df = pd.read_csv(f"quantum_network_metrics_refined_{network_topology_type}.csv")

        # --- Fidelity Plot ---
        plt.figure(figsize=(4, 3))
        for ctype in df['crosstalk_type'].unique():
            subset = df[df['crosstalk_type'] == ctype]
            plt.plot(subset['crosstalk_strength'], subset['fidelity_full_state'], marker='o', label=ctype)
        plt.xlabel("Crosstalk Strength")
        plt.ylabel("Full State Fidelity")
        plt.title("Fidelity vs Crosstalk Strength")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"fidelity_vs_crosstalk_refined_{network_topology_type}.png", dpi=500)
        # plt.show() 

        # --- Effective Throughput Plot ---
        plt.figure(figsize=(4, 3))
        # Format x-axis in scientific notation with 2 decimal places
        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))  # 'e' for exponential
        for ctype in df['crosstalk_type'].unique():
            subset = df[df['crosstalk_type'] == ctype]
            plt.plot(subset['crosstalk_strength'], subset['effective_throughput_qubits_per_sec'], marker='s', label=ctype)
        plt.xlabel("Crosstalk Strength")
        plt.ylabel("Effective Throughput (ops/sec)")
        plt.title("Effective Throughput vs Crosstalk Strength")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"effective_throughput_vs_crosstalk_refined_{network_topology_type}.png", dpi=500)
        # plt.show() 

    plot_results()