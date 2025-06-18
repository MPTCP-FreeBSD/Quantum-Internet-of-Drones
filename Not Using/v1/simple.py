# %%
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Qubit, Clbit


def create_bell_pair(qr: QuantumRegister, cr: ClassicalRegister) -> QuantumCircuit:
    """Creates a bell pair between qubits a and b."""
    qc = QuantumCircuit(qr, cr)
    # unpack qubits
    # the first qubit is s but we won't be using it in this exercise
    s, a, b = qr

    # ENTER YOUR CODE BELOW EACH COMMENT
    # Put qubit a into state |+>
    qc.h(a)
    # CNOT with a as control and b as target
    qc.cx(a, b)
    # ENTER YOUR CODE ABOVE

    return qc  # For the grader

# %%
qr = QuantumRegister(3, name="q")
cr = ClassicalRegister(3, name="c")
qc = create_bell_pair(qr, cr)

qc.draw("mpl")


def alice_gates(qr: QuantumRegister, cr: ClassicalRegister):
    """Creates Alices's gates"""
    qc = create_bell_pair(qr, cr)
    qc.barrier()  # Use barrier to separate steps
    s, a, b = qr

    # ENTER YOUR CODE BELOW EACH COMMENT
    # CNOT with source as control and a as target
    qc.cx(s, a)
    # Apply Hadamard on qubit s
    qc.h(s)
    # ENTER YOUR CODE ABOVE

    return qc  # For the grader

# %%
qc = alice_gates(qr, cr)
qc.draw("mpl")

# %%
def measure_and_send(qr: QuantumRegister, cr: ClassicalRegister):
    """Measures qubits a & b and 'sends' the results to Bob"""
    qc = alice_gates(qr, cr)
    qc.barrier()  # Use barrier to separate steps
    s, a, b = qr
    c0, c1, c2 = cr

    # ENTER YOUR CODE BELOW EACH COMMENT
    # Measure qubit a into classical bit 0
    qc.measure(a, c0)
    # Measure qubit s into classical bit 1
    qc.measure(s, c1)
    # ENTER YOUR CODE ABOVE

    return qc  # For the grader

# %%
qc = measure_and_send(qr, cr)
qc.draw("mpl", cregbundle=False)

def bob_gates(qr: QuantumRegister, cr: ClassicalRegister):
    """Uses qc.if_test to control which gates are dynamically added"""
    qc = measure_and_send(qr, cr)
    qc.barrier()  # Use barrier to separate steps
    s, a, b = qr
    c0, c1, c2 = cr

    # ENTER YOUR CODE BELOW EACH COMMENT
    # Add an X gate to the qubit wire if c0 measures 1
    with qc.if_test((c0, 1)):
        qc.x(b)
    # Add a Z gate to the qubit wire if c1 measures 1
    with qc.if_test((c1, 1)):
        qc.z(b)
    # ENTER YOUR CODE ABOVE

    return qc  # For the grader

qc = bob_gates(qr, cr)
qc.draw("mpl", cregbundle=False)

teleportation_circuit = bob_gates(qr, cr)
s, a, b = qr
c0, c1, c2 = cr
teleportation_circuit.measure(b, c2)
teleportation_circuit.draw("mpl")


import math

teleport_superposition_circuit: QuantumCircuit


# Create a circuit that has the same structure as our teleportation circuit
state_prep = QuantumCircuit(qr, cr)

# Prepare the qubit
state_prep.rx(math.pi / 4, s)

# Put a barrier across all of the wires
state_prep.barrier()
# Add the teleportation circuit to the superposition circuit
teleport_superposition_circuit = state_prep.compose(teleportation_circuit)

teleport_superposition_circuit.draw("mpl", cregbundle=False)

# %%
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

sim = AerSimulator()
transpiled_circuit = transpile(teleport_superposition_circuit, sim)

# run job
shots = 1000
job = sim.run(transpiled_circuit, shots=shots, dynamic=True)

# Get the results and display them
exp_result = job.result()
exp_counts = exp_result.get_counts()
plot_histogram(exp_counts)

# %% [markdown]
# Let's compute the distribution of just Bob's measurement by marginalizing over the other measured bits.

# %%
# trace out Bob's results on qubit 2
from qiskit.result import marginal_counts

bobs_counts = marginal_counts(exp_counts, [qr.index(b)])
plot_histogram(bobs_counts)

