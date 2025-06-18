import math
import time
import pandas as pd
from itertools import combinations
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, state_fidelity, partial_trace
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error, amplitude_damping_error,pauli_error

# Import your custom teleportation circuit builder
from utils.ibm_lab_util import build_qc  # Make sure this is available


def init_qc():
    """Initialize teleportation circuit with input state."""
    qr = QuantumRegister(3, name="q")
    cr = ClassicalRegister(3, name="c")
    teleportation_circuit = build_qc(qr, cr)

    # Prepare input state on qubit s (index 0)
    state_prep = QuantumCircuit(qr, cr)
    state_prep.rx(math.pi / 4, qr[0])
    state_prep.barrier()

    return state_prep.compose(teleportation_circuit), qr, cr, qr[2]  # Return full circuit and Bob's qubit


def build_noise_model(cross_talk_level=0):
    """
    Construct a realistic noise model for simulation.
    
    Parameters:
        cross_talk_level (int): Level of cross-talk (0-10). Higher values imply more simultaneous gate interference.
    
    Returns:
        NoiseModel: A configured Qiskit NoiseModel with cross-talk adaptation.
    """
    if not (0 <= cross_talk_level <= 10):
        raise ValueError("cross_talk_level must be between 0 and 10.")

    noise_model = NoiseModel()

    # Base error rates
    base_single_qubit_error = 0.02 + 0.003 * cross_talk_level
    base_two_qubit_error = 0.04 + 0.006 * cross_talk_level
    thermal_t1 = 100e-3  # T1 = 100 ms
    thermal_t2 = 80e-3   # T2 = 80 ms
    gate_time = 50e-9    # 50 ns gate time

    # Apply realistic single-qubit noise
    depol_error_1q = depolarizing_error(base_single_qubit_error, 1)
    thermal_error_1q = thermal_relaxation_error(thermal_t1, thermal_t2, gate_time)
    combined_1q_error = depol_error_1q.compose(thermal_error_1q)

    noise_model.add_all_qubit_quantum_error(combined_1q_error, ['u1', 'u2', 'u3', 'x', 'z', 'h'])

    # Apply realistic two-qubit noise with cross-talk influence
    depol_error_2q = depolarizing_error(base_two_qubit_error, 2)
    noise_model.add_all_qubit_quantum_error(depol_error_2q, ['cx'])

    # Amplitude damping (often relevant in photonic or mobile scenarios like drones)
    damping_error = amplitude_damping_error(0.05 + 0.005 * cross_talk_level)
    noise_model.add_all_qubit_quantum_error(damping_error, ['x', 'z'])

    # Optional: simulate simultaneous gate execution cross-talk as Pauli noise
    # Optional: simulate simultaneous gate execution cross-talk as Pauli noise
    if cross_talk_level > 0:
        crosstalk_prob = 0.005 * cross_talk_level

        # 1-qubit cross-talk error
        crosstalk_1q_error = pauli_error([('X', crosstalk_prob), ('I', 1 - crosstalk_prob)])
        noise_model.add_all_qubit_quantum_error(crosstalk_1q_error, ['x', 'h', 'z'])

        # 2-qubit cross-talk error (affecting both qubits)
        crosstalk_2q_error = depolarizing_error(crosstalk_prob, 2)
        noise_model.add_all_qubit_quantum_error(crosstalk_2q_error, ['cx'])


    return noise_model


def evaluate_teleportation_segment(noise_model, shots=1000):
    """Run teleportation with noise and return fidelity, latency, and throughput."""
    # Step 1: Get teleportation circuit
    teleport_circuit, qr, cr, b = init_qc()

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

    qunatum_teleporttion_time_for_1bell_pair = 10e-6  # 10 microseconds
    throughput = shots / qunatum_teleporttion_time_for_1bell_pair

    return fidelity_b, fidelity_full, qunatum_teleporttion_time_for_1bell_pair, throughput


# -------------------------------
# Main Script to Loop All Links
# -------------------------------




nodes = [f'N{i}' for i in range(5)]
noise_model = 2
results = []

all_segments = []


for src, dst in combinations(nodes, 2):  # Total 10 links
    print("=== combinations: %d, %d",src,dst)
    segments = [f"{src}-R1", "R1-R2", f"R2-{dst}"]
    all_segments+=segments

# node_comm = (('N0','N1'),('N2','N3'))
# for src, dst in node_comm:  # Total 10 links
#     print("=== combinations: %d, %d",src,dst)
#     segments = [f"{src}-R1", "R1-R2", f"R2-{dst}"]
#     all_segments+=segments

print("all_segments: ",all_segments)

print("len(all_segments): ",len(all_segments))

unique_segments = list(set(all_segments))

print("unique_segments: ",unique_segments)

print("len(unique_segments): ",len(unique_segments))

segment_cross_talk_dict = {}

for segment in unique_segments:
    for othersegment in all_segments:
        if segment == othersegment:
            if segment_cross_talk_dict.get(segment) is None:
                segment_cross_talk_dict[segment] = 0
            else:
                segment_cross_talk_dict[segment]+=1


print()
print("segment_cross_talk_dict",segment_cross_talk_dict)






results = []

for src, dst in combinations(nodes, 2):  # Total 10 links
    segments = [f"{src}-R1", "R1-R2", f"R2-{dst}"]

    for segment in segments:
        print("cross-talk level:" , segment_cross_talk_dict[segment])
        noise_model = build_noise_model(cross_talk_level=segment_cross_talk_dict[segment])
        fidelity_b, fidelity_full, latency, throughput = evaluate_teleportation_segment(noise_model)
        results.append({
            'link': f"{src}-{dst}",
            'segment': segment,
            'fidelity_qubit_b_only': fidelity_b,
            'fidelity_full_state': fidelity_full,
            'latency_sec': latency,
            'throughput_qubits_per_sec': throughput
        })
        print(f"Evaluated: {segment} | Fidelity(b): {fidelity_b:.4f} | Fidelity(full state): {fidelity_full:.4f}| throughput_qubits_per_sec: {throughput:.4f} | Latency: {latency:.4f}s")

# Save to CSV and Excel
df = pd.DataFrame(results)
df.to_csv("quantum_network_metrics.csv", index=False)


print("\n✅ Teleportation metrics saved to:")
print("  • quantum_network_metrics.csv")

