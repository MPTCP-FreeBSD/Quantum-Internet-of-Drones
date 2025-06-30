import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix, state_fidelity, Statevector
from qiskit_aer.noise import NoiseModel
from qiskit.quantum_info import Kraus
import pandas as pd

class CrosstalkDemonstration:
    """Demonstrate different types of quantum crosstalk and their effects."""
    
    def __init__(self):
        self.simulator = AerSimulator(method="density_matrix")
    
    def create_pauli_crosstalk_demo(self, pauli_type='XX', strength=0.05):
        """Demonstrate Pauli crosstalk effects."""
        print(f"\n🔬 PAULI CROSSTALK DEMO ({pauli_type}, strength={strength})")
        print("="*60)
        
        # Create a simple circuit: Apply X gate to qubit 0, measure both qubits
        qc = QuantumCircuit(2)
        qc.x(0)  # Should only affect qubit 0
        qc.save_density_matrix()
        
        # Pauli matrices
        I = np.eye(2)
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])
        pauli_dict = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
        
        # Create crosstalk Kraus operators
        pauli1, pauli2 = pauli_type[0], pauli_type[1]
        pauli_op = np.kron(pauli_dict[pauli1], pauli_dict[pauli2])
        
        K0 = np.sqrt(1 - strength) * np.eye(4)
        K1 = np.sqrt(strength) * pauli_op
        crosstalk_kraus = Kraus([K0, K1])
        
        # Create noise model
        noise_model = NoiseModel()
        noise_model.add_quantum_error(crosstalk_kraus, ['x'], [0, 1])
        
        # Run ideal and noisy simulations
        ideal_result = self.simulator.run(qc).result()
        noisy_result = AerSimulator(noise_model=noise_model, method="density_matrix").run(qc).result()
        
        ideal_dm = DensityMatrix(ideal_result.data(0)['density_matrix'])
        noisy_dm = DensityMatrix(noisy_result.data(0)['density_matrix'])
        
        # Analyze results
        print(f"Expected ideal state: |10⟩ (qubit 0 = |1⟩, qubit 1 = |0⟩)")
        print(f"Ideal density matrix diagonal: {np.diag(ideal_dm.data).real}")
        print(f"Noisy density matrix diagonal: {np.diag(noisy_dm.data).real}")
        print(f"State fidelity: {state_fidelity(ideal_dm, noisy_dm):.4f}")
        
        # Show what happened to each computational basis state
        basis_states = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
        print(f"\nProbability distribution:")
        for i, state in enumerate(basis_states):
            ideal_prob = np.diag(ideal_dm.data)[i].real
            noisy_prob = np.diag(noisy_dm.data)[i].real
            print(f"  {state}: ideal={ideal_prob:.4f}, noisy={noisy_prob:.4f}, change={noisy_prob-ideal_prob:+.4f}")
        
        if pauli_type == 'XX':
            print(f"\n💡 Analysis: XX crosstalk causes correlated bit-flips.")
            print(f"   When we flip qubit 0, crosstalk also affects qubit 1.")
            print(f"   This creates unwanted |01⟩ and |11⟩ components.")
            
        elif pauli_type == 'ZZ':
            print(f"\n💡 Analysis: ZZ crosstalk causes conditional phase shifts.")
            print(f"   Different computational basis states accumulate different phases.")
            
        return ideal_dm, noisy_dm
    
    def create_zz_coupling_demo(self, coupling_strength=0.1, evolution_time=1.0):
        """Demonstrate ZZ coupling in superconducting qubits."""
        print(f"\n🔬 ZZ COUPLING DEMO (strength={coupling_strength}, time={evolution_time})")
        print("="*60)
        
        # Create superposition states on both qubits
        qc = QuantumCircuit(2)
        qc.h(0)  # |+⟩ = (|0⟩ + |1⟩)/√2
        qc.h(1)  # |+⟩ = (|0⟩ + |1⟩)/√2
        # Combined state: |++⟩ = (|00⟩ + |01⟩ + |10⟩ + |11⟩)/2
        qc.save_density_matrix()
        
        # ZZ coupling Hamiltonian: H = J * Z⊗Z
        # Evolution: U = exp(-i * J * t * Z⊗Z)
        ZZ = np.kron(np.array([[1, 0], [0, -1]]), np.array([[1, 0], [0, -1]]))
        zz_unitary = np.array([
            [np.exp(-1j * coupling_strength * evolution_time), 0, 0, 0],
            [0, np.exp(1j * coupling_strength * evolution_time), 0, 0],
            [0, 0, np.exp(1j * coupling_strength * evolution_time), 0],
            [0, 0, 0, np.exp(-1j * coupling_strength * evolution_time)]
        ])
        
        # Create noise model with ZZ coupling
        zz_kraus = Kraus([zz_unitary])
        noise_model = NoiseModel()
        noise_model.add_quantum_error(zz_kraus, ['h'], [0, 1])  # Apply during both H gates
        
        # Simulate
        ideal_result = self.simulator.run(qc).result()
        noisy_result = AerSimulator(noise_model=noise_model, method="density_matrix").run(qc).result()
        
        ideal_dm = DensityMatrix(ideal_result.data(0)['density_matrix'])
        noisy_dm = DensityMatrix(noisy_result.data(0)['density_matrix'])
        
        print(f"Expected ideal amplitudes: all equal (0.5 each)")
        print(f"Ideal state amplitudes: {np.sqrt(np.diag(ideal_dm.data).real)}")
        print(f"Noisy state amplitudes: {np.sqrt(np.diag(noisy_dm.data).real)}")
        print(f"State fidelity: {state_fidelity(ideal_dm, noisy_dm):.4f}")
        
        # Show phase differences
        ideal_phases = np.angle(np.diag(ideal_dm.data))
        noisy_phases = np.angle(np.diag(noisy_dm.data))
        print(f"\nPhase analysis:")
        basis_states = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
        for i, state in enumerate(basis_states):
            phase_diff = noisy_phases[i] - ideal_phases[i]
            print(f"  {state}: phase difference = {phase_diff:.4f} rad")
        
        print(f"\n💡 Analysis: ZZ coupling creates conditional phase shifts.")
        print(f"   |00⟩ and |11⟩ get opposite phases compared to |01⟩ and |10⟩.")
        print(f"   This breaks the symmetry of the |++⟩ state.")
        
        return ideal_dm, noisy_dm
    
    def create_amplitude_phase_demo(self, amp_strength=0.02, phase_strength=0.03):
        """Demonstrate combined amplitude and phase crosstalk."""
        print(f"\n🔬 AMPLITUDE-PHASE CROSSTALK DEMO (amp={amp_strength}, phase={phase_strength})")
        print("="*60)
        
        # Create Bell state
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)  # |Φ+⟩ = (|00⟩ + |11⟩)/√2
        qc.save_density_matrix()
        
        # Amplitude damping: |1⟩ → √(1-γ)|1⟩ + √γ|0⟩
        # Phase damping: Random phase accumulation
        
        # Combined Kraus operators
        sqrt_gamma = np.sqrt(amp_strength)
        sqrt_phi = np.sqrt(phase_strength)
        
        # Amplitude damping operators
        A0 = np.array([[1, 0], [0, np.sqrt(1-amp_strength)]])
        A1 = np.array([[0, sqrt_gamma], [0, 0]])
        
        # Phase damping operators  
        P0 = np.array([[1, 0], [0, np.sqrt(1-phase_strength)]])
        P1 = np.array([[0, 0], [0, sqrt_phi]])
        
        # 2-qubit Kraus operators (simplified model)
        K0 = np.sqrt(1 - amp_strength - phase_strength) * np.eye(4)
        K1 = np.sqrt(amp_strength) * np.kron(A1, np.eye(2))  # Amplitude damping on qubit 0
        K2 = np.sqrt(amp_strength) * np.kron(np.eye(2), A1)  # Amplitude damping on qubit 1
        K3 = np.sqrt(phase_strength) * np.kron(P1, P1)       # Correlated phase damping
        
        amp_phase_kraus = Kraus([K0, K1, K2, K3])
        
        # Create noise model
        noise_model = NoiseModel()
        noise_model.add_quantum_error(amp_phase_kraus, ['cx'], [0, 1])
        
        # Simulate
        ideal_result = self.simulator.run(qc).result()
        noisy_result = AerSimulator(noise_model=noise_model, method="density_matrix").run(qc).result()
        
        ideal_dm = DensityMatrix(ideal_result.data(0)['density_matrix'])
        noisy_dm = DensityMatrix(noisy_result.data(0)['density_matrix'])
        
        # Calculate purity (measure of mixedness)
        ideal_purity = np.trace(ideal_dm.data @ ideal_dm.data).real
        noisy_purity = np.trace(noisy_dm.data @ noisy_dm.data).real
        
        print(f"Expected Bell state: (|00⟩ + |11⟩)/√2")
        print(f"Ideal purity: {ideal_purity:.4f} (1.0 = pure state)")
        print(f"Noisy purity: {noisy_purity:.4f}")
        print(f"Purity loss: {ideal_purity - noisy_purity:.4f}")
        print(f"State fidelity: {state_fidelity(ideal_dm, noisy_dm):.4f}")
        
        print(f"\nPopulation analysis:")
        basis_states = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
        for i, state in enumerate(basis_states):
            ideal_pop = np.diag(ideal_dm.data)[i].real
            noisy_pop = np.diag(noisy_dm.data)[i].real
            print(f"  {state}: ideal={ideal_pop:.4f}, noisy={noisy_pop:.4f}, change={noisy_pop-ideal_pop:+.4f}")
        
        print(f"\n💡 Analysis: Amplitude-phase crosstalk breaks entanglement.")
        print(f"   Amplitude damping: |11⟩ → |01⟩ or |10⟩ (energy loss)")
        print(f"   Phase damping: Random phases destroy coherence")
        print(f"   Combined effect: Mixed state with reduced purity")
        
        return ideal_dm, noisy_dm
    
    def compare_crosstalk_effects(self):
        """Compare all crosstalk types on the same circuit."""
        print(f"\n🔬 CROSSTALK COMPARISON ON GHZ STATE")
        print("="*80)
        
        # Create 3-qubit GHZ state
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)  # |GHZ⟩ = (|000⟩ + |111⟩)/√2
        qc.save_density_matrix()
        
        # Ideal case
        ideal_result = self.simulator.run(qc).result()
        ideal_dm = DensityMatrix(ideal_result.data(0)['density_matrix'])
        ideal_purity = np.trace(ideal_dm.data @ ideal_dm.data).real
        
        results = []
        crosstalk_configs = [
            ('No Crosstalk', None),
            ('XX Pauli', 'XX'),
            ('ZZ Pauli', 'ZZ'),
            ('YY Pauli', 'YY'),
            ('ZZ Coupling', 'ZZ_coupling'),
            ('Amp-Phase', 'amp_phase')
        ]
        
        for name, config_type in crosstalk_configs:
            if config_type is None:
                # Ideal case
                noisy_dm = ideal_dm
            else:
                # Create appropriate noise model
                noise_model = NoiseModel()
                strength = 0.02
                
                if config_type in ['XX', 'ZZ', 'YY']:
                    # Pauli crosstalk
                    I = np.eye(2)
                    X = np.array([[0, 1], [1, 0]])
                    Y = np.array([[0, -1j], [1j, 0]])
                    Z = np.array([[1, 0], [0, -1]])
                    pauli_dict = {'X': X, 'Y': Y, 'Z': Z}
                    
                    pauli_op = np.kron(pauli_dict[config_type[0]], pauli_dict[config_type[1]])
                    K0 = np.sqrt(1 - strength) * np.eye(4)
                    K1 = np.sqrt(strength) * pauli_op
                    kraus = Kraus([K0, K1])
                    
                    # Apply to adjacent pairs
                    noise_model.add_quantum_error(kraus, ['cx'], [0, 1])
                    noise_model.add_quantum_error(kraus, ['cx'], [1, 2])
                    
                elif config_type == 'ZZ_coupling':
                    # ZZ coupling
                    coupling_angle = np.pi/4
                    zz_unitary = np.array([
                        [np.exp(-1j * strength * coupling_angle), 0, 0, 0],
                        [0, np.exp(1j * strength * coupling_angle), 0, 0],
                        [0, 0, np.exp(1j * strength * coupling_angle), 0],
                        [0, 0, 0, np.exp(-1j * strength * coupling_angle)]
                    ])
                    kraus = Kraus([zz_unitary])
                    noise_model.add_quantum_error(kraus, ['cx'], [0, 1])
                    noise_model.add_quantum_error(kraus, ['cx'], [1, 2])
                    
                elif config_type == 'amp_phase':
                    # Amplitude-phase crosstalk
                    amp_str, phase_str = strength/2, strength/2
                    K0 = np.sqrt(1 - amp_str - phase_str) * np.eye(4)
                    
                    # Simplified amplitude damping
                    A1 = np.array([[0, np.sqrt(amp_str)], [0, 0]])
                    K1 = np.kron(A1, np.eye(2))
                    K2 = np.kron(np.eye(2), A1)
                    
                    # Phase damping
                    P1 = np.array([[0, 0], [0, np.sqrt(phase_str)]])
                    K3 = np.kron(P1, P1)
                    
                    kraus = Kraus([K0, K1, K2, K3])
                    noise_model.add_quantum_error(kraus, ['cx'], [0, 1])
                    noise_model.add_quantum_error(kraus, ['cx'], [1, 2])
                
                # Run noisy simulation
                noisy_sim = AerSimulator(noise_model=noise_model, method="density_matrix")
                noisy_result = noisy_sim.run(qc).result()
                noisy_dm = DensityMatrix(noisy_result.data(0)['density_matrix'])
            
            # Calculate metrics
            fidelity = state_fidelity(ideal_dm, noisy_dm)
            purity = np.trace(noisy_dm.data @ noisy_dm.data).real
            purity_loss = ideal_purity - purity
            
            # Population in correct GHZ subspace (|000⟩ + |111⟩)
            ghz_population = (np.diag(noisy_dm.data)[0] + np.diag(noisy_dm.data)[7]).real
            
            results.append({
                'Crosstalk Type': name,
                'Fidelity': fidelity,
                'Purity': purity,
                'Purity Loss': purity_loss,
                'GHZ Population': ghz_population
            })
            
            print(f"{name:15}: Fidelity={fidelity:.4f}, Purity={purity:.4f}, GHZ Pop={ghz_population:.4f}")
        
        # Create comparison DataFrame
        df = pd.DataFrame(results)
        print(f"\n📊 Summary Table:")
        print(df.to_string(index=False, float_format='%.4f'))
        
        return df

def main():
    """Run all crosstalk demonstrations."""
    demo = CrosstalkDemonstration()
    
    print("🚀 QUANTUM CROSSTALK EFFECTS DEMONSTRATION")
    print("="*80)
    
    # 1. Pauli crosstalk demos
    demo.create_pauli_crosstalk_demo('XX', 0.05)
    demo.create_pauli_crosstalk_demo('ZZ', 0.05)
    demo.create_pauli_crosstalk_demo('YY', 0.03)
    
    # 2. ZZ coupling demo
    demo.create_zz_coupling_demo(0.1, 1.0)
    
    # 3. Amplitude-phase crosstalk demo
    demo.create_amplitude_phase_demo(0.02, 0.03)
    
    # 4. Comprehensive comparison
    comparison_df = demo.compare_crosstalk_effects()
    
    print(f"\n✅ All demonstrations completed!")
    print(f"\n🎯 Key Takeaways:")
    print(f"   • XX crosstalk: Creates correlated bit-flip errors")
    print(f"   • ZZ crosstalk: Causes conditional phase accumulation")  
    print(f"   • YY crosstalk: Mixed bit-flip and phase errors")
    print(f"   • ZZ coupling: Always-on interaction in superconducting qubits")
    print(f"   • Amplitude-phase: Combines energy loss and dephasing")
    print(f"   • All types reduce fidelity and purity of quantum states")

if __name__ == "__main__":
    main()