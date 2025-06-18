import numpy as np
import pandas as pd
import time
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, Aer
from qiskit.quantum_info import Statevector, state_fidelity, partial_trace
from qiskit.providers.aer.noise.errors import depolarizing_error

# ---- Atmospheric Turbulence Model ----
def cn_squared(h, v=21.0, A=1.7e-14):
    return (
        0.00594 * (v / 27.0) ** 2 * (10 ** -5 * h) ** 10 * np.exp(-h / 1000.0)
        + 2.7e-16 * np.exp(-h / 1500.0)
        + A * np.exp(-h / 100.0)
    )

def phase_noise_variance(cn2, L, wavelength_nm):
    k = 2 * np.pi / (wavelength_nm * 1e-9)
    sigma2 = 1.23 * cn2 * (k ** (7 / 6)) * (L ** (11 / 6))
    return sigma2

def apply_turbulence_noise(circuit, qubit, sigma):
    theta = np.random.normal(0, np.sqrt(sigma))
    circuit.rz(theta, qubit)

def apply_crosstalk_noise(circuit, qubit, active_links, alpha=0.01):
    lambda_c = alpha * (active_links - 1)
    if lambda_c > 0:
        noise_instr = depolarizing_error(lambda_c, 1).to_instruction()
        circuit.append(noise_instr, [qubit])

# ---- Teleportation Circuit with Noise ----
def create_teleportation_with_crosstalk(sigma_turb, active_links, alpha):
    qr = QuantumRegister(3)
    cr = ClassicalRegister(2)
    qc = QuantumCircuit(qr, cr)

    qc.initialize([1 / np.sqrt(2), 1 / np.sqrt(2)], 0)

    qc.h(1)
    qc.cx(1, 2)

    apply_turbulence_noise(qc, 1, sigma_turb)
    apply_turbulence_noise(qc, 2, sigma_turb)
    apply_crosstalk_noise(qc, 1, active_links, alpha)
    apply_crosstalk_noise(qc, 2, active_links, alpha)

    qc.cx(0, 1)
    qc.h(0)
    qc.measure(0, 0)
    qc.measure(1, 1)
    qc.x(2).c_if(cr, 1)
    qc.z(2).c_if(cr, 2)

    return qc, qr

# ---- Simulation for Node Pairs ----
def simulate_multi_node_teleportation(pairs, altitude_m=500, wavelength_nm=800, alpha=0.01):
    backend = Aer.get_backend("aer_simulator")
    expected = Statevector([1/np.sqrt(2), 1/np.sqrt(2)])
    segment_length = 1000
    cn2 = cn_squared(altitude_m)
    sigma2 = phase_noise_variance(cn2, segment_length, wavelength_nm)

    all_results = []

    for idx, (src, dst) in enumerate(pairs):
        active_links = len(pairs)
        qc, qr = create_teleportation_with_crosstalk(sigma2, active_links, alpha)
        circ = transpile(qc, backend)

        t0 = time.perf_counter()
        result = backend.run(circ).result()
        t1 = time.perf_counter()

        final_state = result.get_statevector(circ)
        teleported = partial_trace(final_state, [0, 1])
        fidelity = state_fidelity(teleported, expected)
        elapsed_time = t1 - t0

        all_results.append({
            "Source": src,
            "Destination": dst,
            "Fidelity": fidelity,
            "Time (s)": elapsed_time,
            "Quantum Throughput (qps)": 1 / elapsed_time if elapsed_time > 0 else 0
        })

    df = pd.DataFrame(all_results)
    df.to_csv("crosstalk_teleportation_results.csv", index=False)
    print("Results saved to 'crosstalk_teleportation_results.csv'")
    return df

# ---- Example Use ----
if __name__ == "__main__":
    example_pairs = [
        ("A", "B"),
        ("C", "D"),
        ("E", "F")
    ]
    simulate_multi_node_teleportation(example_pairs, altitude_m=500, wavelength_nm=800, alpha=0.01)
