
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error, amplitude_damping_error,pauli_error


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


noisemodel = build_noise_model(cross_talk_level=1)

noisemodel = build_noise_model(cross_talk_level=10)


