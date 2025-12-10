import numpy as np
import math
import cmath
import time
from timeit import timeit
import scipy.sparse as sparse

class Dirac:
    """
    Functions for the Dirac notation to describe (quantum) states and (quantum) operators.
    """
    @staticmethod
    def ket(N, a):
        """
        `|a>` is called 'ket' and represents a column vector with `1` in entry `a` and `0` everywhere else.
        """
        ket = np.zeros((N, 1))
        ket[a, 0] = 1
        return ket

    @staticmethod
    def bra(N, a):
        """
        `<a|` is called 'bra' and represents a row vector with `1` in entry `a` and `0` everywhere else.
        """
        bra = np.zeros((1, N))
        bra[0, a] = 1
        return bra

    @staticmethod
    def bra_ket(N, a, b):
        """
        `<a||b>` is the inner product of `<a|` and `|b>`, which is `1` if `a == b` and `0` `if a != b`.
        """
        bra = Dirac.bra(N, a)
        ket = Dirac.ket(N, b)
        return np.inner(bra, ket.T)

    @staticmethod
    def ket_bra(N, a, b):
        """
        `|a><b|` is the outer product of `|a>` and `<b|`, which is a matrix with `1` in entry (a,b) and `0` everywhere else.
        """
        ket = Dirac.ket(N, a)
        bra = Dirac.bra(N, b)
        return np.outer(ket, bra)
    
class QubitUnitaryOperation:
    """
    Functions to obtain 2 x 2 unitary matrices for unitary qubit operations.
    """    
    @staticmethod
    def get_identity():
        return np.array([[1, 0], [0, 1]], dtype=complex)
    
    @staticmethod
    def get_pauli_x():
        return np.array([[0, 1], [1, 0]], dtype=complex)
    
    @staticmethod
    def get_pauli_y():
        return np.array([[0, complex(0,-1)], [complex(0,1), 0]], dtype=complex)
    
    @staticmethod
    def get_pauli_z():
        return np.array([[1, 0], [0, -1]], dtype=complex)
    
    @staticmethod
    def get_hadamard():
        c = complex(1/np.sqrt(2), 0)
        return np.array([[c, c], [c, -c]], dtype=complex)
    
    @staticmethod
    def get_phase(theta):
        c = complex(np.cos(theta),np.sin(theta))
        return np.array([[1, 0], [0, c]], dtype=complex)
    
    @staticmethod
    def get_rotate_x(theta):
        sin = math.sin(theta/2)
        cos = math.cos(theta/2)
        return np.array([[cos, -1j * sin],[-1j * sin, cos]], dtype=complex)
    
    @staticmethod
    def get_rotate_y(theta):
        sin = math.sin(theta/2)
        cos = math.cos(theta/2)
        return np.array([[cos, -sin], [sin, cos]], dtype=complex)
    
    @staticmethod
    def get_rotate_z(theta):
        a = 0.5j * theta
        return np.array([[cmath.exp(-a), 0], [0, cmath.exp(a)]], dtype=complex)

class StateVector:
    """
    Class representing a quantum circuit of N qubits.
    """
    def __init__(self, N):
        self.N = N
        self.index = 0
        self.state_vector = np.zeros((2**self.N, 1), dtype=complex)
        self.state_vector[self.index] = 1

    def apply_unitary_operation(self, operation):
        # Check if operation is a unitary matrix
        if not np.allclose(np.eye(2**self.N), np.conj(operation.T) @ operation):
            raise ValueError("Input matrix is not unitary")
        
        t1 = time.perf_counter()
        _ = operation @ self.state_vector
        t2 = time.perf_counter()
        print(f"apply_unitary_operation: {round(t2-t1, 6)*1000}ms")

        self.state_vector = operation @ self.state_vector

    def measure(self):
        probalities = np.square(np.abs(self.state_vector)).flatten()
        self.index = np.random.choice(len(probalities), p=probalities)

    def get_quantum_state(self):
        return self.state_vector
    
    def get_classical_state_as_string(self):
        return self.__state_as_string(self.index, self.N)
    
    def print(self):
        for i, val in enumerate(self.state_vector):
            print(f"{self.__state_as_string(i, self.N)} : {val[0]}")

    def __state_as_string(self, i,N):
        """
        Function to convert integer i, 0 <= i < N, to a quantum state in Dirac notation.
        """
        # Check if 0 <= i < 2^N
        if i < 0 or i >= 2**N:
            raise ValueError("Input i and N must satisfy 0 <= i < 2^N")
        
        binary_string = bin(i)
        state_as_string = binary_string[2:]
        state_as_string = state_as_string.zfill(N)
        return "|" + state_as_string + ">"


class CircuitUnitaryOperation:
    """
    Functions to obtain 2^N x 2^N unitary matrices for unitary operations on quantum circuits of N qubits.
    """
    @staticmethod
    def get_combined_operation_for_qubit(operation, q, N):
        identity = QubitUnitaryOperation.get_identity()
        combined_operation = np.eye(1,1)

        t1 = time.perf_counter()

        for i in range(0, N):
            if i == q:
                combined_operation = np.kron(combined_operation, operation)
            else:
                combined_operation = np.kron(combined_operation, identity)

        # nnz = np.count_nonzero(combined_operation)
        # print(nnz)
        # print(f"{nnz}/{combined_operation.size} = {nnz/combined_operation.size*100}% sparseness")

        t2 = time.perf_counter()
        bytes = combined_operation.size * 16
        print(f"Generating combined operation: {round(t2-t1, 6)*1000}ms ({bytes:,} bytes)")

        return combined_operation

    
    @staticmethod
    def get_combined_operation_for_identity(q, N):
        return np.array(np.eye(2**N), dtype=complex)
    
    @staticmethod
    def get_combined_operation_for_pauli_x(q, N):
        pauli_x = QubitUnitaryOperation.get_pauli_x()
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(pauli_x, q, N)
    
    @staticmethod
    def get_combined_operation_for_pauli_y(q, N):
        pauli_y = QubitUnitaryOperation.get_pauli_y()
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(pauli_y, q, N)
    
    @staticmethod
    def get_combined_operation_for_pauli_z(q, N):
        pauli_z = QubitUnitaryOperation.get_pauli_z()
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(pauli_z, q, N)
    
    @staticmethod
    def get_combined_operation_for_hadamard(q, N):
        hadamard = QubitUnitaryOperation.get_hadamard()
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(hadamard, q, N)
    
    @staticmethod
    def get_combined_operation_for_phase(theta, q, N):
        phase = QubitUnitaryOperation.get_phase(theta)
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(phase, q, N)
    
    @staticmethod
    def get_combined_operation_for_rotate_x(theta, q, N):
        rotate = QubitUnitaryOperation.get_rotate_x(theta)
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(rotate, q, N)
    
    @staticmethod
    def get_combined_operation_for_rotate_y(theta, q, N):
        rotate = QubitUnitaryOperation.get_rotate_y(theta)
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(rotate, q, N)
    
    @staticmethod
    def get_combined_operation_for_rotate_z(theta, q, N):
        rotate = QubitUnitaryOperation.get_rotate_z(theta)
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(rotate, q, N)
    
    @staticmethod
    def get_combined_operation_for_cnot(control, target, N):
        identity = QubitUnitaryOperation.get_identity()
        pauli_x = QubitUnitaryOperation.get_pauli_x()
        ket_bra_00 = Dirac.ket_bra(2,0,0)
        ket_bra_11 = Dirac.ket_bra(2,1,1)
        combined_operation_zero = np.eye(1,1)
        combined_operation_one = np.eye(1,1)

        t1 = time.perf_counter()

        for i in range(0, N):
            if control == i:
                combined_operation_zero = np.kron(combined_operation_zero, ket_bra_00)
                combined_operation_one  = np.kron(combined_operation_one, ket_bra_11)
            elif target == i:
                combined_operation_zero = np.kron(combined_operation_zero, identity)
                combined_operation_one  = np.kron(combined_operation_one, pauli_x)
            else:
                combined_operation_zero = np.kron(combined_operation_zero, identity)
                combined_operation_one  = np.kron(combined_operation_one, identity)

        operation = combined_operation_zero + combined_operation_one

        t2 = time.perf_counter()
        print(f"Generating combined operation CNOT: {round(t2-t1, 6)*1000}ms")

        return operation
    
'''
Symbol for pi
'''
pi_symbol = '\u03c0'

class Circuit:
    """
    Class representing a quantum circuit of N qubits.
    """
    def __init__(self, N):
        self.qubits = N
        self.state_vector = StateVector(self.qubits)
        self.quantum_states = [self.state_vector.get_quantum_state()]
        self.descriptions = []
        self.operations = []

    def identity(self, q):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_identity(q, self.qubits)
        self.descriptions.append(f"Identity on qubit {q}")
        self.operations.append(combined_operation)

    def pauli_x(self, q):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_pauli_x(q, self.qubits)
        self.descriptions.append(f"Pauli X on qubit {q}")
        self.operations.append(combined_operation)

    def pauli_y(self, q):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_pauli_y(q, self.qubits)
        self.descriptions.append(f"Pauli Y on qubit {q}")
        self.operations.append(combined_operation)

    def pauli_z(self, q):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_pauli_z(q, self.qubits)
        self.descriptions.append(f"Pauli Z on qubit {q}")
        self.operations.append(combined_operation)

    def hadamard(self, q):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_hadamard(q, self.qubits)
        self.descriptions.append(f"Hadamard on qubit {q}")
        self.operations.append(combined_operation)

    def phase(self, theta, q):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_phase(theta, q, self.qubits)
        self.descriptions.append(f"Phase with theta = {theta/np.pi:.3f} {pi_symbol} on qubit {q}")
        self.operations.append(combined_operation)

    def rotate_x(self, theta, q):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_rotate_x(theta, q, self.qubits)
        self.descriptions.append(f"Rotate X with theta = {theta/np.pi:.3f} {pi_symbol} on qubit {q}")
        self.operations.append(combined_operation)
    
    def rotate_y(self, theta, q):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_rotate_y(theta, q, self.qubits)
        self.descriptions.append(f"Rotate Y with theta = {theta/np.pi:.3f} {pi_symbol} on qubit {q}")
        self.operations.append(combined_operation)
    
    def rotate_z(self, theta, q):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_rotate_z(theta, q, self.qubits)
        self.descriptions.append(f"Rotate Z with theta = {theta/np.pi:.3f} {pi_symbol} on qubit {q}")
        self.operations.append(combined_operation)

    def cnot(self, control, target):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_cnot(control, target, self.qubits)
        self.descriptions.append(f"CNOT with control qubit {control} and target qubit {target}")
        self.operations.append(combined_operation)

    def execute(self, print_state=False):
        self.state_vector = StateVector(self.qubits)
        if print_state:
            print("Initial quantum state")
            self.state_vector.print()
        print(type(self.operations))
        print(type(self.operations[0]))
        for operation, description in zip(self.operations, self.descriptions):
            self.state_vector.apply_unitary_operation(operation)
            self.quantum_states.append(self.state_vector.get_quantum_state())
            if print_state:
                print(description)
                print(operation)
                print("Current quantum state")
                self.state_vector.print()

    def measure(self, print_state=False):
        self.state_vector.measure()
        if print_state:
            print("Measured state:")
            print(self.state_vector.get_classical_state_as_string())

    def get_classical_state_as_string(self):
        return self.state_vector.get_classical_state_as_string()
    