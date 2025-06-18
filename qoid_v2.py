import math
import time
import pandas as pd
from itertools import combinations
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, state_fidelity, partial_trace
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error, amplitude_damping_error

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


def build_noise_model():
    """Construct a realistic noise model for simulation."""
    noise_model = NoiseModel()
    depol_error = depolarizing_error(0.05, 1)
    noise_model.add_all_qubit_quantum_error(depol_error, ['u1', 'u2', 'u3'])

    thermal_error = thermal_relaxation_error(0.1, 0.1, 1)
    noise_model.add_all_qubit_quantum_error(thermal_error, ['h', 'x', 'z'])

    depol_2qubit_error = depolarizing_error(0.05, 2)
    noise_model.add_all_qubit_quantum_error(depol_2qubit_error, ['cx'])

    amplitude_damping = amplitude_damping_error(0.1)
    noise_model.add_all_qubit_quantum_error(amplitude_damping, ['x', 'z'])

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

    qunatum_teleporttion_time_for_1bell_pair = 10 ^ 6  # 10 microseconds
    throughput = shots / qunatum_teleporttion_time_for_1bell_pair

    return fidelity_b, fidelity_full, qunatum_teleporttion_time_for_1bell_pair, throughput


# -------------------------------
# Main Script to Loop All Links
# -------------------------------

nodes = [f'N{i}' for i in range(5)]

results = []

for src, dst in combinations(nodes, 2):  # Total 10 links
    segments = [f"{src}-R1", "R1-R2", f"R2-{dst}"]

    for segment in segments:
        noise_model = build_noise_model()
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

