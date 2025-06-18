from itertools import combinations
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import state_fidelity
from qiskit.visualization import plot_histogram
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error, pauli_error,amplitude_damping_error

def evaluate_teleportation_segment():


    from utils.ibm_lab_util import build_qc  # Assuming build_qc is defined in this module

    import matplotlib.pyplot as plt
    import time
    import math



    def init_qc():
        qr = QuantumRegister(3, name="q")
        cr = ClassicalRegister(3, name="c")

        teleportation_circuit = build_qc(qr, cr)
        s, a, b = qr

        # Prepare input state
        state_prep = QuantumCircuit(qr, cr)
        state_prep.rx(math.pi / 4, s)
        state_prep.barrier()

        # Combine circuits
        full_circuit = state_prep.compose(teleportation_circuit)

        return full_circuit, qr, cr, b

    # -------------------------
    # Step 2: Setup
    # -------------------------

    noise_model = NoiseModel()

    # Add depolarizing error with 5% probability for all single qubit gates (u1, u2, u3)
    depol_error = depolarizing_error(0.05, 1)
    noise_model.add_all_qubit_quantum_error(depol_error, ['u1', 'u2', 'u3'])

    # Add thermal relaxation error (relaxation and dephasing model) ONLY to single-qubit gates
    thermal_error = thermal_relaxation_error(0.1, 0.1, 1)  # 10% error probability for relaxation and dephasing
    noise_model.add_all_qubit_quantum_error(thermal_error, ['h', 'x', 'z'])  # Apply to single qubit gates

    # Add depolarizing error to the 2-qubit gates (such as cx)
    depol_2qubit_error = depolarizing_error(0.05, 2)  # 5% depolarizing error on 2-qubit gates
    noise_model.add_all_qubit_quantum_error(depol_2qubit_error, ['cx'])

    # Add amplitude damping (relaxation error) to simulate energy loss for single-qubit gates
    amplitude_damping = amplitude_damping_error(0.1)  # 10% chance of relaxation
    noise_model.add_all_qubit_quantum_error(amplitude_damping, ['x', 'z'])


    # Initialize circuit
    teleport_circuit, qr, cr, b = init_qc()

    # Create simulators
    ideal_sim = AerSimulator(method="statevector")
    noisy_sim = AerSimulator(noise_model=noise_model, method="statevector")

    # Save statevectors
    ideal_circuit = teleport_circuit.copy()
    ideal_circuit.save_statevector()

    noisy_circuit = transpile(teleport_circuit, noisy_sim)
    noisy_circuit.save_statevector()

    # -------------------------
    # Step 3: Fidelity
    # -------------------------

    # Run ideal simulation
    ideal_result = ideal_sim.run(ideal_circuit).result()


    # Run noisy simulation
    noisy_result = noisy_sim.run(noisy_circuit).result()

    # fidelity = state_fidelity(ideal_sv, noisy_sv)
    # print(f"Quantum Fidelity: {fidelity:.4f}")

    # print(ideal_sv)
    # print(noisy_sv)

    from qiskit.quantum_info import Statevector, partial_trace

    # Get statevectors
    ideal_sv = Statevector(ideal_result.get_statevector())
    noisy_sv = Statevector(noisy_result.get_statevector())

    # Qubits: s=0, a=1, b=2
    # We want to compare only the state of qubit b (index 2)

    # Trace out qubits s (0) and a (1), leaving only qubit b (2)
    ideal_b = partial_trace(ideal_sv, [0, 1])
    noisy_b = partial_trace(noisy_sv, [0, 1])

    # Now compute fidelity of just qubit b
    fidelity = state_fidelity(ideal_b, noisy_b)
    print(f"Quantum Fidelity (qubit b only): {fidelity:.4f}")


    fidelity = state_fidelity(ideal_sv, noisy_sv)
    print(f"Quantum Fidelity state_fidelity(ideal_sv, noisy_sv): {fidelity:.4f}")


    # -------------------------
    # Step 4: Latency and Throughput (Measurement-Based)
    # -------------------------

    # Add measurement and run multiple shots
    shots = 1000
    measured_circuit = teleport_circuit.copy()
    measured_circuit.measure_all()

    start = time.time()
    result = noisy_sim.run(measured_circuit, shots=shots).result()
    end = time.time()

    latency = end - start
    throughput = shots / latency

    print(f"Quantum Latency (sec): {latency:.4f}")
    print(f"Quantum Throughput (qubits/sec): {throughput:.2f}")

    # -------------------------
    # Step 5: Visualization
    # -------------------------

    counts = result.get_counts()
    print(counts)
    # plot_histogram(counts)
    # plt.title("Teleportation Measurement Results (Noisy Simulation)")
    # plt.show()
    return fidelity, latency, throughput

nodes = [f'N{i}' for i in range(5)]
link_results = []

# Iterate over all node pairs
for src, dst in combinations(nodes, 2):
    # 2 repeaters → 3 segments: src–R1, R1–R2, R2–dst
    segment_names = [f"{src}-R1", "R1-R2", f"R2-{dst}"]

    for segment in segment_names:
        print(f"Evaluating segment: {segment} from {src} to {dst}")
        # fidelity, latency, throughput = evaluate_teleportation_segment()
        # link_results.append({
        #     'link': f"{src}-{dst}",
        #     'segment': segment,
        #     'fidelity': fidelity,
        #     'latency': latency,
        #     'throughput': throughput
        # })