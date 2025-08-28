import math
import time
import pandas as pd
from itertools import combinations
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, state_fidelity, partial_trace
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error, amplitude_damping_error, pauli_error

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

def build_noise_model(cross_talk_level=0):
    """
    Build a realistic noise model with tunable cross-talk.

    Parameters:
        cross_talk_level (int): Cross-talk level (0–10). Higher => more noise.

    Returns:
        NoiseModel: Configured Qiskit noise model.
    """
    if not (0 <= cross_talk_level <= 10):
        raise ValueError("cross_talk_level must be between 0 and 10.")

    noise_model = NoiseModel()

    # Constants
    thermal_t1 = 100e-3  # T1 = 100 ms
    thermal_t2 = 80e-3   # T2 = 80 ms
    gate_time = 50e-9    # 50 ns

    # Base error rates (adjusted by cross-talk)
    base_1q_error = 0.02 + 0.003 * cross_talk_level
    base_2q_error = 0.04 + 0.006 * cross_talk_level
    amp_damp_prob = 0.05 + 0.005 * cross_talk_level
    crosstalk_prob = 0.005 * cross_talk_level if cross_talk_level > 0 else 0.0

    # ----- 1-Qubit Errors -----
    depol_1q = depolarizing_error(base_1q_error, 1)
    thermal_1q = thermal_relaxation_error(thermal_t1, thermal_t2, gate_time)
    amp_damp_1q = amplitude_damping_error(amp_damp_prob)
    crosstalk_1q = pauli_error([('X', crosstalk_prob), ('I', 1 - crosstalk_prob)]) if crosstalk_prob > 0 else None

    combined_1q = depol_1q.compose(thermal_1q).compose(amp_damp_1q)
    if crosstalk_1q:
        combined_1q = combined_1q.compose(crosstalk_1q)

    noise_model.add_all_qubit_quantum_error(
        combined_1q, ['u1', 'u2', 'u3', 'x', 'z', 'h']
    )

    # ----- 2-Qubit Errors -----
    depol_2q = depolarizing_error(base_2q_error, 2)
    crosstalk_2q = depolarizing_error(crosstalk_prob, 2) if crosstalk_prob > 0 else None

    combined_2q = depol_2q
    if crosstalk_2q:
        combined_2q = combined_2q.compose(crosstalk_2q)

    noise_model.add_all_qubit_quantum_error(combined_2q, ['cx'])

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
    ideal_sim = AerSimulator(method="statevector")
    noisy_sim = AerSimulator(noise_model=noise_model, method="statevector")
    
    # Save statevectors
    ideal_circuit = teleport_circuit.copy()
    ideal_circuit.save_statevector()

    noisy_circuit = transpile(teleport_circuit, noisy_sim)
    noisy_circuit.save_statevector()

    ideal_result = ideal_sim.run(ideal_circuit).result()
    noisy_result = noisy_sim.run(noisy_circuit).result()

    ideal_sv = Statevector(ideal_result.get_statevector())
    noisy_sv = Statevector(noisy_result.get_statevector())

    # Step 3: Fidelity
    ideal_b = partial_trace(ideal_sv, [0, 1])
    noisy_b = partial_trace(noisy_sv, [0, 1])

    fidelity_b = state_fidelity(ideal_b, noisy_b)
    fidelity_full = state_fidelity(ideal_sv, noisy_sv)

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

            if level == 2:
                continue

            noise_model = build_noise_model(cross_talk_level=level)
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
