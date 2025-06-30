import math
import time
import pandas as pd
from itertools import combinations
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, state_fidelity, partial_trace
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error, amplitude_damping_error, pauli_error
from qiskit.quantum_info import Kraus
import numpy as np
import itertools


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




def create_cross_kraus_error(crosstalk_strength=0.02):
    """Safely creates CPTP 2-qubit crosstalk Kraus channel."""
    I = np.eye(4)
    X = np.array([[0, 1], [1, 0]])
    XX = np.kron(X, X)

    K0 = np.sqrt(1 - crosstalk_strength) * I
    K1 = np.sqrt(crosstalk_strength) * XX

    # Verify normalization
    norm_check = K0.conj().T @ K0 + K1.conj().T @ K1
    if not np.allclose(norm_check, np.eye(4), atol=1e-6):
        print("[Warning] Kraus operators not CPTP. Auto-normalizing.")
        # Normalize entire channel
        scale = np.sqrt(np.linalg.norm(norm_check))
        K0 = K0 / scale
        K1 = K1 / scale

    return Kraus([K0, K1])

def build_noise_model(cross_talk_level=0, num_qubits=3):
    """
    Builds a realistic noise model with depolarizing, amplitude damping, thermal, and custom 2-qubit crosstalk noise.

    Parameters:
        cross_talk_level (int): Level from 0–10
        num_qubits (int): Number of qubits in teleportation circuit (default=3)

    Returns:
        NoiseModel: Qiskit-compatible noise model
    """
    if not (0 <= cross_talk_level <= 10):
        raise ValueError("cross_talk_level must be between 0 and 10.")

    noise_model = NoiseModel()

    # Base physical noise parameters
    thermal_t1 = 100e-3  # T1 in seconds
    thermal_t2 = 80e-3   # T2 in seconds
    gate_time = 50e-9    # Gate duration in seconds

    # Noise scaling
    base_1q_error = 0.02 + 0.003 * cross_talk_level
    base_2q_error = 0.04 + 0.006 * cross_talk_level
    amp_damp_prob = 0.05 + 0.005 * cross_talk_level
    crosstalk_strength = 0.005 * cross_talk_level if cross_talk_level > 0 else 0.0

    # --- Single-Qubit Errors ---
    depol_1q = depolarizing_error(base_1q_error, 1)
    thermal_1q = thermal_relaxation_error(thermal_t1, thermal_t2, gate_time)
    amp_damp_1q = amplitude_damping_error(amp_damp_prob)
    combined_1q = depol_1q.compose(thermal_1q).compose(amp_damp_1q)
    noise_model.add_all_qubit_quantum_error(combined_1q, ['x', 'y', 'z', 'u1', 'u2', 'u3', 'h'])

    # --- Base 2-Qubit Depolarizing Error ---
    depol_2q = depolarizing_error(base_2q_error, 2)
    noise_model.add_all_qubit_quantum_error(depol_2q, ['cx'])

    # --- Add Custom Crosstalk if Requested ---
    if crosstalk_strength > 0:
        cross_kraus = create_cross_kraus_error(crosstalk_strength)
        all_pairs = list(itertools.combinations(range(num_qubits), 2))
        used_pairs = all_pairs[:min(cross_talk_level, len(all_pairs))]

        for q1, q2 in used_pairs:
            # Add crosstalk Kraus noise in addition to depolarizing error
            noise_model.add_quantum_error(cross_kraus, ['cx'], [q1, q2])
            print(f"  ➤ Crosstalk applied on (q{q1}, q{q2})")

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
    nodes = [f'N{i}' for i in range(5)]
    results = []
    all_segments = []

    for src, dst in combinations(nodes, 2):
        segments = [f"{src}-R1", "R1-R2", f"R2-{dst}"]
        all_segments += segments

    unique_segments = list(set(all_segments))

    segment_cross_talk_dict = {}
    for segment in all_segments:
        segment_cross_talk_dict[segment] = segment_cross_talk_dict.get(segment, 0) + 1

    print("Unique Segments and Cross-talk Levels:")
    for seg, count in segment_cross_talk_dict.items():
        print(f"  {seg}: {count}")

    for src, dst in combinations(nodes, 2):
        segments = [f"{src}-R1", "R1-R2", f"R2-{dst}"]

        for segment in segments:
            level = segment_cross_talk_dict[segment]
            print(f"Evaluating segment: {segment} with cross-talk level {level}")


            noise_model = build_noise_model(cross_talk_level=level, num_qubits=3)

            fidelity_b, fidelity_full, latency, raw, effective = evaluate_teleportation_segment(noise_model)

            results.append({
                'link': f"{src}-{dst}",
                'segment': segment,
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
