import math
import random
import time
import pandas as pd
from itertools import combinations
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Statevector, state_fidelity, partial_trace
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error, amplitude_damping_error

from utils.ibm_lab_util import build_qc  # Ensure this module is available

def cn_squared(h, v=21.0, A=1.7e-14):
    return (
        0.00594 * (v / 27.0) ** 2 * (10 ** -5 * h) ** 10 * np.exp(-h / 1000.0)
        + 2.7e-16 * np.exp(-h / 1500.0)
        + A * np.exp(-h / 100.0)
    )

def phase_noise_variance(cn2, L, wavelength_nm):
    k = 2 * np.pi / (wavelength_nm * 1e-9)
    return 1.23 * cn2 * (k ** (7 / 6)) * (L ** (11 / 6))

def build_noise_model(base_error_scale=1.0):
    noise_model = NoiseModel()
    depol_error = depolarizing_error(0.05 * base_error_scale, 1)
    noise_model.add_all_qubit_quantum_error(depol_error, ['u1', 'u2', 'u3'])

    thermal_error = thermal_relaxation_error(0.1 * base_error_scale, 0.1 * base_error_scale, 1)
    noise_model.add_all_qubit_quantum_error(thermal_error, ['h', 'x', 'z'])

    depol_2qubit_error = depolarizing_error(0.05 * base_error_scale, 2)
    noise_model.add_all_qubit_quantum_error(depol_2qubit_error, ['cx'])

    amplitude_damping = amplitude_damping_error(0.1 * base_error_scale)
    noise_model.add_all_qubit_quantum_error(amplitude_damping, ['x', 'z'])

    return noise_model

def init_qc():
    qr = QuantumRegister(3, name="q")
    cr = ClassicalRegister(3, name="c")
    teleportation_circuit = build_qc(qr, cr)
    state_prep = QuantumCircuit(qr, cr)
    state_prep.rx(math.pi / 4, qr[0])
    state_prep.barrier()
    return state_prep.compose(teleportation_circuit), qr, cr, qr[2]

def evaluate_segment(segment, active_links, base_error_scale=1.0, shots=1000, wavelength_nm=800):
    # Compute cross-talk penalty: count how many other segments overlap (spatially/temporally)
    print("[link for link in active_links]",[link for link in active_links])
    print("segment",segment)
    print("sum(segment in link for link in active_links if link != segment)",sum(segment in link for link in active_links if link != segment))
    
    cross_talk_factor = sum(segment in link for link in active_links if link != segment)
    cross_talk_factor = max(0, sum(1 for link in active_links if link == segment) - 1)
    cross_talk_factor = random.randint(1,3)
    cross_talk_factor = 1


    print("cross_talk_factor: ",cross_talk_factor)

    print("$$"*20)

    # Increase error scale linearly with number of simultaneous links using the same segment
    noise_model = build_noise_model(base_error_scale=base_error_scale * (1 + 0.2 * cross_talk_factor))

    teleport_circuit, qr, cr, b = init_qc()
    ideal_sim = AerSimulator(method="statevector")
    noisy_sim = AerSimulator(noise_model=noise_model, method="statevector")

    ideal_circuit = teleport_circuit.copy()
    ideal_circuit.save_statevector()
    noisy_circuit = transpile(teleport_circuit, noisy_sim)
    noisy_circuit.save_statevector()

    ideal_result = ideal_sim.run(ideal_circuit).result()
    noisy_result = noisy_sim.run(noisy_circuit).result()

    ideal_sv = Statevector(ideal_result.get_statevector())
    noisy_sv = Statevector(noisy_result.get_statevector())

    ideal_b = partial_trace(ideal_sv, [0, 1])
    noisy_b = partial_trace(noisy_sv, [0, 1])

    fidelity_b = state_fidelity(ideal_b, noisy_b)
    fidelity_full = state_fidelity(ideal_sv, noisy_sv)

    measured_circuit = teleport_circuit.copy()
    measured_circuit.measure_all()
    start = time.time()
    result = noisy_sim.run(measured_circuit, shots=shots).result()
    end = time.time()

    teleport_time = 1e-5  # 10 microseconds
    throughput = shots / teleport_time

    return fidelity_b, fidelity_full, teleport_time, throughput, cross_talk_factor

if __name__ == "__main__":
    nodes = [f'N{i}' for i in range(5)]
    results = []
    simultaneous_pairs = [('N0', 'N4'), ('N1', 'N2'), ('N3', 'N4')]  # Define active comms

    for src, dst in simultaneous_pairs:
        link_id = f"{src}-{dst}"
        segments = [f"{src}-R1", "R1-R2", f"R2-{dst}"]
        active_segments = set()
        for pair in simultaneous_pairs:
            active_segments.update([f"{pair[0]}-R1", "R1-R2", f"R2-{pair[1]}"])

        for segment in segments:
            fidelity_b, fidelity_full, latency, throughput, crosstalk = evaluate_segment(
                segment, active_segments
            )
            
            results.append({
                'link': link_id,
                'segment': segment,
                'fidelity_qubit_b_only': fidelity_b,
                'fidelity_full_state': fidelity_full,
                'latency_sec': latency,
                'throughput_qubits_per_sec': throughput,
                'cross_talk_level': crosstalk
            })

            print(f"{segment} | Crosstalk: {crosstalk} | Fidelity(b): {fidelity_b:.4f} | Throughput: {throughput:.2e}")

    df = pd.DataFrame(results)
    df.to_csv("quantum_network_metrics_crosstalk.csv", index=False)
    print("\n✅ Results saved to quantum_network_metrics_crosstalk.csv")
