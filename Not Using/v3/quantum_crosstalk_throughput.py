import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix, state_fidelity, Statevector
from qiskit_aer.noise import NoiseModel
from qiskit.quantum_info import Kraus
import pandas as pd
import time

class CrosstalkDemonstration:
    """Demonstrate different types of quantum crosstalk and their effects on throughput."""
    
    def __init__(self):
        self.simulator = AerSimulator(method="density_matrix")
    
    def calculate_quantum_throughput(self, fidelity, gate_time_us=0.1, readout_time_us=1.0, 
                                   fidelity_threshold=0.95, circuit_depth=10):
        """
        Calculate quantum throughput metrics.
        
        Args:
            fidelity: State fidelity after crosstalk
            gate_time_us: Gate execution time in microseconds
            readout_time_us: Readout time in microseconds
            fidelity_threshold: Minimum acceptable fidelity
            circuit_depth: Number of gates in circuit
        
        Returns:
            Dictionary with throughput metrics
        """
        # Basic timing
        total_time_us = circuit_depth * gate_time_us + readout_time_us
        
        # Effective operations per second (only counting successful operations)
        if fidelity >= fidelity_threshold:
            success_rate = fidelity  # Simplified model
            effective_ops_per_sec = (1e6 / total_time_us) * success_rate
        else:
            effective_ops_per_sec = 0  # Below threshold
        
        # Quantum volume approximation (simplified)
        quantum_volume = min(circuit_depth, 2**circuit_depth * fidelity)
        
        # Information throughput (bits/second)
        # Assumes each operation processes log2(circuit_depth) bits of quantum information
        info_throughput = effective_ops_per_sec * np.log2(max(circuit_depth, 2))
        
        # Error-corrected throughput (accounts for error correction overhead)
        if fidelity > 0.999:
            ec_overhead = 1.1  # Minimal overhead for high fidelity
        elif fidelity > 0.99:
            ec_overhead = 5.0  # Moderate overhead
        elif fidelity > 0.95:
            ec_overhead = 50.0  # High overhead
        else:
            ec_overhead = np.inf  # Error correction impossible
        
        ec_throughput = effective_ops_per_sec / ec_overhead if ec_overhead != np.inf else 0
        
        return {
            'total_time_us': total_time_us,
            'raw_ops_per_sec': 1e6 / total_time_us,
            'effective_ops_per_sec': effective_ops_per_sec,
            'quantum_volume': quantum_volume,
            'info_throughput_bits_per_sec': info_throughput,
            'error_corrected_throughput': ec_throughput,
            'success_rate': fidelity if fidelity >= fidelity_threshold else 0
        }
    
    def create_pauli_crosstalk_demo(self, pauli_type='XX', strength=0.05):
        """Demonstrate Pauli crosstalk effects with throughput analysis."""
        print(f"\n🔬 PAULI CROSSTALK DEMO ({pauli_type}, strength={strength})")
        print("="*60)
        
        # Create a circuit with a 2-qubit gate to demonstrate crosstalk
        qc = QuantumCircuit(2)
        qc.h(0)  # Put qubit 0 in superposition
        qc.cx(0, 1)  # CNOT gate - this is where crosstalk occurs
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
        
        # Create noise model - add crosstalk to 2-qubit gates
        noise_model = NoiseModel()
        noise_model.add_quantum_error(crosstalk_kraus, ['cx'], [0, 1])
        
        # Run ideal and noisy simulations
        ideal_result = self.simulator.run(qc).result()
        noisy_result = AerSimulator(noise_model=noise_model, method="density_matrix").run(qc).result()
        
        ideal_dm = DensityMatrix(ideal_result.data(0)['density_matrix'])
        noisy_dm = DensityMatrix(noisy_result.data(0)['density_matrix'])
        
        # Calculate fidelity
        fidelity = state_fidelity(ideal_dm, noisy_dm)
        
        # Calculate throughput metrics
        throughput = self.calculate_quantum_throughput(fidelity, circuit_depth=1)
        
        # Analyze results
        print(f"Expected ideal state: Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2")
        print(f"State fidelity: {fidelity:.4f}")
        print(f"Effective ops/sec: {throughput['effective_ops_per_sec']:.0f}")
        print(f"Quantum volume: {throughput['quantum_volume']:.2f}")
        print(f"Info throughput: {throughput['info_throughput_bits_per_sec']:.0f} bits/sec")
        
        # Show probability distribution
        basis_states = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
        print(f"\nProbability distribution:")
        for i, state in enumerate(basis_states):
            ideal_prob = np.diag(ideal_dm.data)[i].real
            noisy_prob = np.diag(noisy_dm.data)[i].real
            print(f"  {state}: ideal={ideal_prob:.4f}, noisy={noisy_prob:.4f}, change={noisy_prob-ideal_prob:+.4f}")
        
        if pauli_type == 'XX':
            print(f"\n💡 Analysis: XX crosstalk causes correlated bit-flips.")
            print(f"   Throughput impact: {(1-fidelity)*100:.1f}% reduction in effective operations")
        
        return ideal_dm, noisy_dm, throughput
    
    def create_zz_coupling_demo(self, coupling_strength=0.1, evolution_time=1.0):
        """Demonstrate ZZ coupling with throughput analysis."""
        print(f"\n🔬 ZZ COUPLING DEMO (strength={coupling_strength}, time={evolution_time})")
        print("="*60)
        
        # Create superposition states on both qubits
        qc = QuantumCircuit(2)
        qc.h(0)  # |+⟩ = (|0⟩ + |1⟩)/√2
        qc.h(1)  # |+⟩ = (|0⟩ + |1⟩)/√2
        qc.save_density_matrix()
        
        # ZZ coupling Hamiltonian
        zz_unitary = np.array([
            [np.exp(-1j * coupling_strength * evolution_time), 0, 0, 0],
            [0, np.exp(1j * coupling_strength * evolution_time), 0, 0],
            [0, 0, np.exp(1j * coupling_strength * evolution_time), 0],
            [0, 0, 0, np.exp(-1j * coupling_strength * evolution_time)]
        ])
        
        # Create noise model
        zz_kraus = Kraus([zz_unitary])
        noise_model = NoiseModel()
        noise_model.add_quantum_error(zz_kraus, ['h'], [0, 1])
        
        # Simulate
        ideal_result = self.simulator.run(qc).result()
        noisy_result = AerSimulator(noise_model=noise_model, method="density_matrix").run(qc).result()
        
        ideal_dm = DensityMatrix(ideal_result.data(0)['density_matrix'])
        noisy_dm = DensityMatrix(noisy_result.data(0)['density_matrix'])
        
        fidelity = state_fidelity(ideal_dm, noisy_dm)
        throughput = self.calculate_quantum_throughput(fidelity, circuit_depth=2)
        
        print(f"State fidelity: {fidelity:.4f}")
        print(f"Effective ops/sec: {throughput['effective_ops_per_sec']:.0f}")
        print(f"Quantum volume: {throughput['quantum_volume']:.2f}")
        print(f"Error-corrected throughput: {throughput['error_corrected_throughput']:.0f} ops/sec")
        
        print(f"\n💡 Analysis: ZZ coupling creates conditional phase shifts.")
        print(f"   Always-on interaction limits parallel operation throughput")
        
        return ideal_dm, noisy_dm, throughput
    
    def create_amplitude_phase_demo(self, amp_strength=0.02, phase_strength=0.03):
        """Demonstrate amplitude-phase crosstalk with throughput analysis."""
        print(f"\n🔬 AMPLITUDE-PHASE CROSSTALK DEMO (amp={amp_strength}, phase={phase_strength})")
        print("="*60)
        
        # Create Bell state
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.save_density_matrix()
        
        # Combined Kraus operators
        sqrt_gamma = np.sqrt(amp_strength)
        sqrt_phi = np.sqrt(phase_strength)
        
        # 2-qubit Kraus operators
        K0 = np.sqrt(1 - amp_strength - phase_strength) * np.eye(4)
        A1 = np.array([[0, sqrt_gamma], [0, 0]])
        K1 = np.sqrt(amp_strength) * np.kron(A1, np.eye(2))
        K2 = np.sqrt(amp_strength) * np.kron(np.eye(2), A1)
        P1 = np.array([[0, 0], [0, sqrt_phi]])
        K3 = np.sqrt(phase_strength) * np.kron(P1, P1)
        
        amp_phase_kraus = Kraus([K0, K1, K2, K3])
        
        # Create noise model
        noise_model = NoiseModel()
        noise_model.add_quantum_error(amp_phase_kraus, ['cx'], [0, 1])
        
        # Simulate
        ideal_result = self.simulator.run(qc).result()
        noisy_result = AerSimulator(noise_model=noise_model, method="density_matrix").run(qc).result()
        
        ideal_dm = DensityMatrix(ideal_result.data(0)['density_matrix'])
        noisy_dm = DensityMatrix(noisy_result.data(0)['density_matrix'])
        
        fidelity = state_fidelity(ideal_dm, noisy_dm)
        throughput = self.calculate_quantum_throughput(fidelity, circuit_depth=2)
        
        # Calculate purity
        ideal_purity = np.trace(ideal_dm.data @ ideal_dm.data).real
        noisy_purity = np.trace(noisy_dm.data @ noisy_dm.data).real
        
        print(f"State fidelity: {fidelity:.4f}")
        print(f"Purity loss: {ideal_purity - noisy_purity:.4f}")
        print(f"Effective ops/sec: {throughput['effective_ops_per_sec']:.0f}")
        print(f"Info throughput: {throughput['info_throughput_bits_per_sec']:.0f} bits/sec")
        
        print(f"\n💡 Analysis: Combined errors severely impact quantum advantage")
        print(f"   Entanglement degradation reduces computational throughput")
        
        return ideal_dm, noisy_dm, throughput
    
    def throughput_scaling_analysis(self, max_qubits=8):
        """Analyze how crosstalk affects throughput scaling with system size."""
        print(f"\n🔬 THROUGHPUT SCALING ANALYSIS (up to {max_qubits} qubits)")
        print("="*60)
        
        results = []
        crosstalk_strengths = [0.0, 0.01, 0.02, 0.05]
        
        for n_qubits in range(2, max_qubits + 1):
            for strength in crosstalk_strengths:
                # Create random quantum circuit
                qc = QuantumCircuit(n_qubits)
                circuit_depth = n_qubits  # Scale depth with system size
                
                # Add random gates
                for _ in range(circuit_depth):
                    if np.random.random() < 0.5:
                        # Single-qubit gate
                        qubit = np.random.randint(n_qubits)
                        qc.h(qubit)
                    else:
                        # Two-qubit gate
                        q1, q2 = np.random.choice(n_qubits, 2, replace=False)
                        qc.cx(q1, q2)
                
                qc.save_density_matrix()
                
                # Create noise model if crosstalk present
                if strength > 0:
                    noise_model = NoiseModel()
                    # Add ZZ crosstalk to all adjacent pairs
                    for i in range(n_qubits - 1):
                        zz_angle = strength * np.pi/4
                        zz_unitary = np.array([
                            [np.exp(-1j * zz_angle), 0, 0, 0],
                            [0, np.exp(1j * zz_angle), 0, 0],
                            [0, 0, np.exp(1j * zz_angle), 0],
                            [0, 0, 0, np.exp(-1j * zz_angle)]
                        ])
                        zz_kraus = Kraus([zz_unitary])
                        noise_model.add_quantum_error(zz_kraus, ['cx'], [i, i+1])
                    
                    noisy_sim = AerSimulator(noise_model=noise_model, method="density_matrix")
                    noisy_result = noisy_sim.run(qc).result()
                    noisy_dm = DensityMatrix(noisy_result.data(0)['density_matrix'])
                else:
                    noisy_result = self.simulator.run(qc).result()
                    noisy_dm = DensityMatrix(noisy_result.data(0)['density_matrix'])
                
                # Calculate ideal state for comparison
                ideal_result = self.simulator.run(qc).result()
                ideal_dm = DensityMatrix(ideal_result.data(0)['density_matrix'])
                
                fidelity = state_fidelity(ideal_dm, noisy_dm)
                throughput = self.calculate_quantum_throughput(
                    fidelity, circuit_depth=circuit_depth
                )
                
                results.append({
                    'n_qubits': n_qubits,
                    'crosstalk_strength': strength,
                    'fidelity': fidelity,
                    'effective_ops_per_sec': throughput['effective_ops_per_sec'],
                    'quantum_volume': throughput['quantum_volume'],
                    'info_throughput': throughput['info_throughput_bits_per_sec'],
                    'circuit_depth': circuit_depth
                })
        
        df = pd.DataFrame(results)
        
        # Print summary for each crosstalk strength
        for strength in crosstalk_strengths:
            subset = df[df['crosstalk_strength'] == strength]
            print(f"\nCrosstalk strength = {strength}:")
            print(f"  Max effective ops/sec: {subset['effective_ops_per_sec'].max():.0f}")
            print(f"  Max quantum volume: {subset['quantum_volume'].max():.1f}")
            print(f"  Max info throughput: {subset['info_throughput'].max():.0f} bits/sec")
            
            # Show scaling trend
            if len(subset) > 1:
                throughput_trend = subset['effective_ops_per_sec'].iloc[-1] / subset['effective_ops_per_sec'].iloc[0]
                print(f"  Throughput scaling factor: {throughput_trend:.2f}x")
        
        return df
    
    def compare_crosstalk_effects(self):
        """Compare all crosstalk types with comprehensive throughput analysis."""
        print(f"\n🔬 COMPREHENSIVE CROSSTALK & THROUGHPUT COMPARISON")
        print("="*80)
        
        # Create 3-qubit GHZ state
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.save_density_matrix()
        
        # Ideal case
        ideal_result = self.simulator.run(qc).result()
        ideal_dm = DensityMatrix(ideal_result.data(0)['density_matrix'])
        
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
                noisy_dm = ideal_dm
                fidelity = 1.0
            else:
                # Create noise model
                noise_model = NoiseModel()
                strength = 0.02
                
                if config_type in ['XX', 'ZZ', 'YY']:
                    # Pauli crosstalk
                    pauli_dict = {
                        'X': np.array([[0, 1], [1, 0]]),
                        'Y': np.array([[0, -1j], [1j, 0]]),
                        'Z': np.array([[1, 0], [0, -1]])
                    }
                    
                    pauli_op = np.kron(pauli_dict[config_type[0]], pauli_dict[config_type[1]])
                    K0 = np.sqrt(1 - strength) * np.eye(4)
                    K1 = np.sqrt(strength) * pauli_op
                    kraus = Kraus([K0, K1])
                    
                    noise_model.add_quantum_error(kraus, ['cx'], [0, 1])
                    noise_model.add_quantum_error(kraus, ['cx'], [1, 2])
                    
                elif config_type == 'ZZ_coupling':
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
                    amp_str, phase_str = strength/2, strength/2
                    K0 = np.sqrt(1 - amp_str - phase_str) * np.eye(4)
                    A1 = np.array([[0, np.sqrt(amp_str)], [0, 0]])
                    K1 = np.kron(A1, np.eye(2))
                    K2 = np.kron(np.eye(2), A1)
                    P1 = np.array([[0, 0], [0, np.sqrt(phase_str)]])
                    K3 = np.kron(P1, P1)
                    kraus = Kraus([K0, K1, K2, K3])
                    noise_model.add_quantum_error(kraus, ['cx'], [0, 1])
                    noise_model.add_quantum_error(kraus, ['cx'], [1, 2])
                
                # Run simulation
                noisy_sim = AerSimulator(noise_model=noise_model, method="density_matrix")
                noisy_result = noisy_sim.run(qc).result()
                noisy_dm = DensityMatrix(noisy_result.data(0)['density_matrix'])
                fidelity = state_fidelity(ideal_dm, noisy_dm)
            
            # Calculate comprehensive metrics
            throughput = self.calculate_quantum_throughput(fidelity, circuit_depth=3)
            purity = np.trace(noisy_dm.data @ noisy_dm.data).real
            
            results.append({
                'Crosstalk Type': name,
                'Fidelity': fidelity,
                'Purity': purity,
                'Effective Ops/sec': throughput['effective_ops_per_sec'],
                'Quantum Volume': throughput['quantum_volume'],
                'Info Throughput (bits/sec)': throughput['info_throughput_bits_per_sec'],
                'EC Throughput': throughput['error_corrected_throughput'],
                'Success Rate': throughput['success_rate']
            })
        
        df = pd.DataFrame(results)
        print(df.to_string(index=False, float_format='%.2f'))
        
        # Throughput impact analysis
        print(f"\n📊 THROUGHPUT IMPACT ANALYSIS:")
        baseline_throughput = df.iloc[0]['Effective Ops/sec']  # No crosstalk case
        for _, row in df.iterrows():
            if row['Crosstalk Type'] != 'No Crosstalk':
                impact = (1 - row['Effective Ops/sec'] / baseline_throughput) * 100
                print(f"  {row['Crosstalk Type']:15}: {impact:+5.1f}% throughput reduction")
        
        return df

def main():
    """Run all crosstalk demonstrations with throughput analysis."""
    demo = CrosstalkDemonstration()
    
    print("🚀 QUANTUM CROSSTALK & THROUGHPUT ANALYSIS")
    print("="*80)
    
    # Individual crosstalk demos with throughput
    demo.create_pauli_crosstalk_demo('XX', 0.05)
    demo.create_zz_coupling_demo(0.1, 1.0)
    demo.create_amplitude_phase_demo(0.02, 0.03)
    
    # Scaling analysis
    scaling_df = demo.throughput_scaling_analysis(max_qubits=6)
    
    # Comprehensive comparison
    comparison_df = demo.compare_crosstalk_effects()
    
    print(f"\n✅ Analysis completed!")
    print(f"\n🎯 Key Throughput Insights:")
    print(f"   • Crosstalk reduces effective operations per second")
    print(f"   • Error correction overhead scales with noise level")
    print(f"   • Quantum volume decreases with crosstalk strength")
    print(f"   • Information throughput is limited by fidelity")
    print(f"   • System scaling is severely impacted by crosstalk")

if __name__ == "__main__":
    main()