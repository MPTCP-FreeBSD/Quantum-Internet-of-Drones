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

def build_qc(qr: QuantumRegister, cr: ClassicalRegister):
    """Builds the quantum circuit for the teleportation protocol."""
    return bob_gates(qr, cr)