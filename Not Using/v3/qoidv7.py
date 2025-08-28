import numpy as np
import pandas as pd
import time
import itertools
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import os

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import (
    DensityMatrix, 
    state_fidelity, 
    partial_trace, 
    Kraus,
    random_unitary,
    Statevector
)
from qiskit_aer.noise import (
    NoiseModel, 
    depolarizing_error, 
    amplitude_damping_error, 
    thermal_relaxation_error,
    phase_damping_error,
    pauli_error
)


class QuantumCrosstalkSimulator:
    """
    Comprehensive quantum crosstalk simulator using density matrix method.
    Supports various crosstalk models and configurable qubit pair interactions.
    """
    
    def __init__(self, num_qubits: int = 5):
        self.num_qubits = num_qubits
        self.simulator = AerSimulator(method="density_matrix")
        self.results = []
        
    def create_pauli_crosstalk_kraus(self, strength: float, pauli_type: str = 'XX') -> Kraus:
        """Create Pauli-based crosstalk Kraus operators."""
        I = np.eye(2)
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])
        
        pauli_dict = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
        
        if len(pauli_type) != 2:
            raise ValueError("pauli_type must be 2-character string like 'XX', 'XY', etc.")
            
        pauli1, pauli2 = pauli_type[0], pauli_type[1]
        pauli_op = np.kron(pauli_dict[pauli1], pauli_dict[pauli2])
        
        K0 = np.sqrt(1 - strength) * np.eye(4)
        K1 = np.sqrt(strength) * pauli_op
        
        # Ensure CPTP condition
        norm_check = K0.conj().T @ K0 + K1.conj().T @ K1
        if not np.allclose(norm_check, np.eye(4), atol=1e-10):
            # Renormalize
            total_norm = np.trace(norm_check)
            K0 = K0 / np.sqrt(total_norm)
            K1 = K1 / np.sqrt(total_norm)
            
        return Kraus([K0, K1])
    
    def create_zz_crosstalk_kraus(self, strength: float, coupling_angle: float = 0.0) -> Kraus:
        """Create ZZ-coupling crosstalk (common in superconducting qubits)."""
        # ZZ interaction: exp(-i * strength * Z⊗Z * coupling_angle)
        ZZ = np.kron(np.array([[1, 0], [0, -1]]), np.array([[1, 0], [0, -1]]))
        
        # Unitary evolution under ZZ coupling
        zz_unitary = np.array([
            [np.exp(-1j * strength * coupling_angle), 0, 0, 0],
            [0, np.exp(1j * strength * coupling_angle), 0, 0],
            [0, 0, np.exp(1j * strength * coupling_angle), 0],
            [0, 0, 0, np.exp(-1j * strength * coupling_angle)]
        ])
        
        # Convert to Kraus (unitary channel has single Kraus operator)
        return Kraus([zz_unitary])
    
    def create_amplitude_phase_crosstalk_error(self, amp_strength: float, phase_strength: float):
        """Create combined amplitude and phase crosstalk using Qiskit's built-in error models."""
        # Use Qiskit's built-in amplitude and phase damping errors
        amp_error = amplitude_damping_error(amp_strength)
        phase_error = phase_damping_error(phase_strength)
        
        # For 2-qubit crosstalk, we'll create correlated errors
        # First, create single-qubit errors for each qubit
        amp_error_2q = amp_error.tensor(amp_error)  # Independent amplitude damping on both qubits
        phase_error_2q = phase_error.tensor(phase_error)  # Independent phase damping on both qubits
        
        # Combine them (this creates a mixed error model)
        combined_error = amp_error_2q.compose(phase_error_2q)
        
        return combined_error
    
    def create_depolarizing_crosstalk_error(self, strength: float):
        """Create depolarizing crosstalk error."""
        return depolarizing_error(strength, 2)
    
    def create_random_crosstalk_kraus(self, strength: float, num_operators: int = 4) -> Kraus:
        """Create random crosstalk using random unitary matrices."""
        kraus_ops = []
        remaining_strength = 1.0
        
        for i in range(num_operators - 1):
            # Random strength for this operator
            op_strength = np.random.uniform(0, remaining_strength * strength)
            remaining_strength -= op_strength / strength
            
            # Random 4x4 unitary
            random_u = random_unitary(4).data
            kraus_ops.append(np.sqrt(op_strength) * random_u)
        
        # Last operator gets remaining strength
        final_strength = remaining_strength * strength
        if final_strength > 0:
            random_u = random_unitary(4).data
            kraus_ops.append(np.sqrt(final_strength) * random_u)
        
        # Identity component
        kraus_ops.insert(0, np.sqrt(1 - strength) * np.eye(4))
        
        return Kraus(kraus_ops)
    
    def build_crosstalk_noise_model(self, 
                                   crosstalk_config: Dict[str, Any],
                                   intersecting_pairs: List[Tuple[int, int]]) -> NoiseModel:
        """
        Build noise model with configurable crosstalk on specified qubit pairs.
        
        Args:
            crosstalk_config: Dictionary with crosstalk parameters
            intersecting_pairs: List of qubit pairs that have crosstalk
        """
        noise_model = NoiseModel()
        
        # Base noise parameters
        base_1q_error = crosstalk_config.get('base_1q_error', 0.001)
        base_2q_error = crosstalk_config.get('base_2q_error', 0.01)
        thermal_t1 = crosstalk_config.get('t1_time', 50e-6)
        thermal_t2 = crosstalk_config.get('t2_time', 30e-6)
        gate_time = crosstalk_config.get('gate_time', 100e-9)
        
        # Single-qubit errors
        if base_1q_error > 0:
            depol_1q = depolarizing_error(base_1q_error, 1)
            thermal_1q = thermal_relaxation_error(thermal_t1, thermal_t2, gate_time)
            combined_1q = depol_1q.compose(thermal_1q)
            noise_model.add_all_qubit_quantum_error(combined_1q, ['x', 'y', 'z', 'h', 'rx', 'ry', 'rz'])
        
        # Base two-qubit errors
        if base_2q_error > 0:
            depol_2q = depolarizing_error(base_2q_error, 2)
            noise_model.add_all_qubit_quantum_error(depol_2q, ['cx', 'cz'])
        
        # Add crosstalk to specified pairs
        crosstalk_type = crosstalk_config.get('type', 'pauli')
        crosstalk_strength = crosstalk_config.get('strength', 0.01)
        
        for q1, q2 in intersecting_pairs:
            if crosstalk_type == 'pauli':
                pauli_type = crosstalk_config.get('pauli_type', 'XX')
                crosstalk_kraus = self.create_pauli_crosstalk_kraus(crosstalk_strength, pauli_type)
            elif crosstalk_type == 'zz':
                coupling_angle = crosstalk_config.get('coupling_angle', np.pi/4)
                crosstalk_kraus = self.create_zz_crosstalk_kraus(crosstalk_strength, coupling_angle)
            elif crosstalk_type == 'amp_phase':
                # Use the new method that returns a proper Qiskit error
                amp_strength = crosstalk_config.get('amp_strength', crosstalk_strength/2)
                phase_strength = crosstalk_config.get('phase_strength', crosstalk_strength/2)
                crosstalk_error = self.create_amplitude_phase_crosstalk_error(amp_strength, phase_strength)
                # Apply the error directly
                noise_model.add_quantum_error(crosstalk_error, ['cx', 'cz'], [q1, q2])
                print(f"  → Applied {crosstalk_type} crosstalk (amp={amp_strength:.4f}, phase={phase_strength:.4f}) to qubits ({q1}, {q2})")
                continue  # Skip the Kraus application below
            elif crosstalk_type == 'depolarizing':
                crosstalk_error = self.create_depolarizing_crosstalk_error(crosstalk_strength)
                noise_model.add_quantum_error(crosstalk_error, ['cx', 'cz'], [q1, q2])
                print(f"  → Applied {crosstalk_type} crosstalk (strength={crosstalk_strength:.4f}) to qubits ({q1}, {q2})")
                continue  # Skip the Kraus application below
            elif crosstalk_type == 'random':
                num_ops = crosstalk_config.get('num_operators', 4)
                crosstalk_kraus = self.create_random_crosstalk_kraus(crosstalk_strength, num_ops)
            else:
                raise ValueError(f"Unknown crosstalk type: {crosstalk_type}")
            
            # Apply crosstalk to 2-qubit gates on this pair (only for Kraus-based methods)
            if crosstalk_type in ['pauli', 'zz', 'random']:
                noise_model.add_quantum_error(crosstalk_kraus, ['cx', 'cz'], [q1, q2])
                print(f"  → Applied {crosstalk_type} crosstalk (strength={crosstalk_strength:.4f}) to qubits ({q1}, {q2})")
        
        return noise_model
    
    def create_test_circuit(self, circuit_type: str = 'ghz') -> QuantumCircuit:
        """Create test circuits for crosstalk evaluation."""
        qc = QuantumCircuit(self.num_qubits)
        
        if circuit_type == 'ghz':
            # GHZ state preparation
            qc.h(0)
            for i in range(1, self.num_qubits):
                qc.cx(0, i)
                
        elif circuit_type == 'random_circuit':
            # Random circuit with mix of single and two-qubit gates
            np.random.seed(42)  # For reproducibility
            depth = 5
            for d in range(depth):
                # Random single-qubit gates
                for q in range(self.num_qubits):
                    if np.random.random() < 0.3:
                        gate_choice = np.random.choice(['x', 'y', 'z', 'h'])
                        getattr(qc, gate_choice)(q)
                
                # Random two-qubit gates
                for q in range(self.num_qubits - 1):
                    if np.random.random() < 0.4:
                        qc.cx(q, q + 1)
                        
        elif circuit_type == 'teleportation':
            # Quantum teleportation circuit (requires at least 3 qubits)
            if self.num_qubits < 3:
                raise ValueError("Teleportation requires at least 3 qubits")
            
            # Prepare state to teleport (|+> state)
            qc.h(0)
            
            # Create Bell pair
            qc.h(1)
            qc.cx(1, 2)
            
            # Bell measurement
            qc.cx(0, 1)
            qc.h(0)
            
        elif circuit_type == 'all_pairs':
            # Circuit that involves all possible qubit pairs
            for i in range(self.num_qubits):
                qc.h(i)  # Superposition
            
            for i in range(self.num_qubits):
                for j in range(i + 1, self.num_qubits):
                    qc.cx(i, j)
                    
        else:
            raise ValueError(f"Unknown circuit type: {circuit_type}")
        
        return qc
    
    def generate_intersecting_pairs(self, 
                                  pair_generation_method: str = 'nearest_neighbor',
                                  num_pairs: int = None,
                                  custom_pairs: List[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
        """Generate qubit pairs that will have crosstalk."""
        
        if custom_pairs is not None:
            return custom_pairs
        
        all_pairs = list(itertools.combinations(range(self.num_qubits), 2))
        
        if pair_generation_method == 'nearest_neighbor':
            # Adjacent qubits in linear topology
            pairs = [(i, i + 1) for i in range(self.num_qubits - 1)]
            
        elif pair_generation_method == 'all_pairs':
            pairs = all_pairs
            
        elif pair_generation_method == 'random':
            np.random.seed(42)
            if num_pairs is None:
                num_pairs = min(3, len(all_pairs))
            pairs = list(np.random.choice(len(all_pairs), size=num_pairs, replace=False))
            pairs = [all_pairs[i] for i in pairs]
            
        elif pair_generation_method == 'grid_2d':
            # 2D grid topology (assuming square grid)
            grid_size = int(np.sqrt(self.num_qubits))
            if grid_size * grid_size != self.num_qubits:
                print(f"Warning: {self.num_qubits} qubits don't form perfect square grid. Using nearest neighbor.")
                return self.generate_intersecting_pairs('nearest_neighbor')
            
            pairs = []
            for i in range(grid_size):
                for j in range(grid_size):
                    qubit = i * grid_size + j
                    # Right neighbor
                    if j < grid_size - 1:
                        pairs.append((qubit, qubit + 1))
                    # Bottom neighbor
                    if i < grid_size - 1:
                        pairs.append((qubit, qubit + grid_size))
                        
        else:
            raise ValueError(f"Unknown pair generation method: {pair_generation_method}")
        
        # Limit number of pairs if specified
        if num_pairs is not None and len(pairs) > num_pairs:
            pairs = pairs[:num_pairs]
            
        return pairs

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
    
    def evaluate_crosstalk_impact(self, 
                                circuit: QuantumCircuit,
                                crosstalk_config: Dict[str, Any],
                                intersecting_pairs: List[Tuple[int, int]],
                                shots: int = 1000) -> Dict[str, float]:
        """Evaluate the impact of crosstalk on circuit fidelity."""
        
        # Create ideal and noisy circuits
        ideal_circuit = circuit.copy()
        ideal_circuit.save_density_matrix()
        
        noisy_circuit = circuit.copy()
        noisy_circuit.save_density_matrix()
        
        # Build noise model
        noise_model = self.build_crosstalk_noise_model(crosstalk_config, intersecting_pairs)
        
        # Simulate ideal case
        ideal_sim = AerSimulator(method="density_matrix")
        ideal_result = ideal_sim.run(ideal_circuit).result()
        ideal_dm = DensityMatrix(ideal_result.data(0)['density_matrix'])
        
        # Simulate noisy case
        noisy_sim = AerSimulator(noise_model=noise_model, method="density_matrix")
        noisy_circuit_transpiled = transpile(noisy_circuit, noisy_sim)
        
        start_time = time.time()
        noisy_result = noisy_sim.run(noisy_circuit_transpiled).result()
        execution_time = time.time() - start_time
        
        noisy_dm = DensityMatrix(noisy_result.data(0)['density_matrix'])
        
        # Calculate fidelities
        full_fidelity = state_fidelity(ideal_dm, noisy_dm)
        
        # Purity measures
        ideal_purity = np.trace(ideal_dm.data @ ideal_dm.data).real
        noisy_purity = np.trace(noisy_dm.data @ noisy_dm.data).real
        
        # Trace distance
        trace_distance = 0.5 * np.trace(np.abs((ideal_dm.data - noisy_dm.data))).real

        # Calculate throughput metrics
        throughput = self.calculate_quantum_throughput(full_fidelity, circuit_depth=1)
        
        # Analyze results
        print(f"Expected ideal state: Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2")
        print(f"State fidelity: {full_fidelity:.4f}")
        print(f"Effective ops/sec: {throughput['effective_ops_per_sec']:.0f}")
        print(f"Quantum volume: {throughput['quantum_volume']:.2f}")
        print(f"Info throughput: {throughput['info_throughput_bits_per_sec']:.0f} bits/sec")

        # Data to store
        result_row = {
            'full_state_fidelity': full_fidelity,
            'ideal_purity': ideal_purity,
            'noisy_purity': noisy_purity,
            'purity_loss': ideal_purity - noisy_purity,
            'trace_distance': trace_distance,
            'execution_time': execution_time,
            'num_intersecting_pairs': len(intersecting_pairs),
            'total_time_us': throughput['total_time_us'],
            'raw_ops_per_sec': throughput['raw_ops_per_sec'],
            'effective_ops_per_sec': throughput['effective_ops_per_sec'],
            'quantum_volume': throughput['quantum_volume'],
            'info_throughput_bits_per_sec': throughput['info_throughput_bits_per_sec'],
            'error_corrected_throughput': throughput['error_corrected_throughput'],
            'success_rate': throughput['success_rate']
        }

        self.results.append(result_row)
        
        return {
            'full_state_fidelity': full_fidelity,
            'ideal_purity': ideal_purity,
            'noisy_purity': noisy_purity,
            'purity_loss': ideal_purity - noisy_purity,
            'trace_distance': trace_distance,
            'execution_time': execution_time,
            'num_intersecting_pairs': len(intersecting_pairs)
        }
    
    def save_csv(self):
        # Save to central CSV
        csv_file = "quantum_throughput_results.csv"
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df = pd.concat([df, pd.DataFrame(self.results)], ignore_index=True)
        else:
            df = pd.DataFrame(self.results)

        df.to_csv(csv_file, index=False)
        
        print("self.results.length", len(self.results))
        if len(self.results) > 0:
            print("self.results[0]", self.results[0])

        # Return result for further use
        return df

    
    def run_parameter_sweep(self, 
                           circuit_types: List[str] = ['ghz', 'random_circuit'],
                           crosstalk_strengths: List[float] = [0.001, 0.005, 0.01, 0.02, 0.05],
                           crosstalk_types: List[str] = ['pauli', 'zz', 'amp_phase'],
                           pair_counts: List[int] = [1, 2, 3, 5],
                           shots: int = 1000) -> pd.DataFrame:
        """
        Run comprehensive parameter sweep across different crosstalk configurations.
        """
        results = []
        
        for circuit_type in circuit_types:
            print(f"\n=== Evaluating {circuit_type} circuit ===")
            circuit = self.create_test_circuit(circuit_type)
            
            for crosstalk_type in crosstalk_types:
                print(f"\n--- Crosstalk type: {crosstalk_type} ---")
                
                for strength in crosstalk_strengths:
                    for num_pairs in pair_counts:
                        if num_pairs > (self.num_qubits * (self.num_qubits - 1)) // 2:
                            continue  # Skip if more pairs than possible
                        
                        # Generate intersecting pairs
                        intersecting_pairs = self.generate_intersecting_pairs('random', num_pairs)
                        
                        # Configure crosstalk
                        crosstalk_config = {
                            'type': crosstalk_type,
                            'strength': strength,
                            'base_1q_error': 0.001,
                            'base_2q_error': 0.005,
                            'pauli_type': 'XX',  # for Pauli crosstalk
                            'coupling_angle': np.pi/4,  # for ZZ crosstalk
                            'amp_strength': strength/2,  # for amp_phase crosstalk
                            'phase_strength': strength/2
                        }
                        
                        print(f"  Strength: {strength:.4f}, Pairs: {num_pairs}, Intersecting: {intersecting_pairs}")
                        
                        # Evaluate
                        try:
                            metrics = self.evaluate_crosstalk_impact(
                                circuit, crosstalk_config, intersecting_pairs, shots
                            )
                            
                            result = {
                                'circuit_type': circuit_type,
                                'crosstalk_type': crosstalk_type,
                                'crosstalk_strength': strength,
                                'num_intersecting_pairs': num_pairs,
                                'intersecting_pairs': str(intersecting_pairs),
                                **metrics
                            }
                            results.append(result)
                            
                            print(f"    → Fidelity: {metrics['full_state_fidelity']:.4f}, "
                                  f"Purity loss: {metrics['purity_loss']:.4f}")
                            
                        except Exception as e:
                            print(f"    → Error: {e}")
                            continue
        self.save_csv()
        return pd.DataFrame(results)


def main_simulation():
    """Main simulation function demonstrating crosstalk analysis."""
    
    # Initialize simulator
    simulator = QuantumCrosstalkSimulator(num_qubits=5)
    
    print("🔬 Quantum Crosstalk Simulation with Density Matrix Method")
    print("=" * 60)
    
    # Run parameter sweep
    results_df = simulator.run_parameter_sweep(
        circuit_types=['teleportation'],
        crosstalk_strengths=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
        crosstalk_types=['amp_phase', 'pauli', 'depolarizing', 'zz'],
        pair_counts=[1, 2, 3, 4],
        shots=1000
    )
    
    # Save results
    results_df.to_csv('quantum_crosstalk_analysis.csv', index=False)
    results_df.to_excel('quantum_crosstalk_analysis.xlsx', index=False)
    
    print(f"\n✅ Analysis complete! Results saved to CSV and Excel files.")
    print(f"📊 Total configurations analyzed: {len(results_df)}")
    
    # Display summary statistics
    print("\n📈 Summary Statistics:")
    print("-" * 40)
    
    if len(results_df) > 0:
        summary_stats = results_df.groupby(['crosstalk_type', 'circuit_type']).agg({
            'full_state_fidelity': ['mean', 'std', 'min', 'max'],
            'purity_loss': ['mean', 'std'],
            'trace_distance': ['mean', 'std']
        }).round(4)
        
        print(summary_stats)
        
        # Best and worst performing configurations
        print("\n🏆 Best Performing Configurations (Highest Fidelity):")
        best_configs = results_df.nlargest(5, 'full_state_fidelity')[
            ['circuit_type', 'crosstalk_type', 'crosstalk_strength', 'num_intersecting_pairs', 'full_state_fidelity']
        ]
        print(best_configs.to_string(index=False))
        
        print("\n💥 Worst Performing Configurations (Lowest Fidelity):")
        worst_configs = results_df.nsmallest(5, 'full_state_fidelity')[
            ['circuit_type', 'crosstalk_type', 'crosstalk_strength', 'num_intersecting_pairs', 'full_state_fidelity']
        ]
        print(worst_configs.to_string(index=False))
    
    return results_df


if __name__ == "__main__":
    # Run the main simulation
    results = main_simulation()
    
    # Optional: Create some basic visualizations if matplotlib/seaborn available
    try:
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Fidelity vs Crosstalk Strength
        plt.subplot(2, 2, 1)
        for ct_type in results['crosstalk_type'].unique():
            data = results[results['crosstalk_type'] == ct_type]
            grouped = data.groupby('crosstalk_strength')['full_state_fidelity'].mean()
            plt.plot(grouped.index, grouped.values, marker='o', label=ct_type)
        plt.xlabel('Crosstalk Strength')
        plt.ylabel('Average Fidelity')
        plt.title('Fidelity vs Crosstalk Strength')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Purity Loss vs Number of Intersecting Pairs
        plt.subplot(2, 2, 2)
        sns.boxplot(data=results, x='num_intersecting_pairs', y='purity_loss')
        plt.xlabel('Number of Intersecting Pairs')
        plt.ylabel('Purity Loss')
        plt.title('Purity Loss vs Intersecting Pairs')
        
        # Plot 3: Heatmap of Fidelity by Circuit Type and Crosstalk Type
        plt.subplot(2, 2, 3)
        pivot_data = results.groupby(['circuit_type', 'crosstalk_type'])['full_state_fidelity'].mean().unstack()
        sns.heatmap(pivot_data, annot=True, cmap='viridis', fmt='.3f')
        plt.title('Average Fidelity by Circuit and Crosstalk Type')
        
        # Plot 4: Execution Time vs Configuration Complexity
        plt.subplot(2, 2, 4)
        results['complexity'] = results['crosstalk_strength'] * results['num_intersecting_pairs']
        plt.scatter(results['complexity'], results['execution_time'], 
                   c=results['full_state_fidelity'], cmap='coolwarm', alpha=0.7)
        plt.xlabel('Configuration Complexity')
        plt.ylabel('Execution Time (s)')
        plt.title('Execution Time vs Complexity')
        plt.colorbar(label='Fidelity')
        
        plt.tight_layout()
        plt.savefig('quantum_crosstalk_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n📊 Visualization saved as 'quantum_crosstalk_analysis.png'")
        
    except ImportError:
        print("\n📊 Matplotlib/Seaborn not available for visualization")
    except Exception as e:
        print(f"\n📊 Visualization error: {e}")