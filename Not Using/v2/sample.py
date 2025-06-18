from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, assemble
from qiskit_aer import AerSimulator
from qiskit.quantum_info import state_fidelity
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
from qiskit.quantum_info import state_fidelity
import numpy as np

def create_bell_pair():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc

def measure_bell_pair():
    qc = QuantumCircuit(2, 2)
    qc.measure([0, 1], [0, 1])
    return qc

def create_noise_model():
    noise_model = NoiseModel()
    
    # Depolarizing + Crosstalk style model
    dep_err = depolarizing_error(0.02, 1)
    cx_err = depolarizing_error(0.03, 2)
    
    noise_model.add_all_qubit_quantum_error(dep_err, ['h'])
    noise_model.add_all_qubit_quantum_error(cx_err, ['cx'])
    
    return noise_model

def simulate_link(noise_model, shots=100):
    backend = Aer.get_backend('aer_simulator')
    
    # Create entangled pair
    bell_circuit = create_bell_pair()
    bell_circuit.save_statevector()
    
    # Transpile
    bell_circuit = transpile(bell_circuit, backend)
    qobj = assemble(bell_circuit)
    
    # Simulate
    result = backend.run(qobj, noise_model=noise_model).result()
    sv = result.get_statevector()
    
    # Fidelity with ideal Bell state
    bell_state = [1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]
    fidelity = state_fidelity(sv, bell_state)
    
    return fidelity

# === Simulate All Star Links ===
noise_model = create_noise_model()
successful = 0
threshold = 0.9
runs_per_link = 100

for link in ['0-1', '0-2', '0-3', '0-4']:
    link_successes = 0
    for _ in range(runs_per_link):
        fid = simulate_link(noise_model)
        if fid >= threshold:
            link_successes += 1
    print(f"Link {link} success rate: {link_successes}/{runs_per_link}")
    successful += link_successes

# Total throughput estimate
total_throughput = successful / (len(['0-1', '0-2', '0-3', '0-4']) * runs_per_link)
print(f"Estimated throughput: {total_throughput * 100:.2f}%")
