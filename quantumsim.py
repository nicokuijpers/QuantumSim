import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
import random
from collections import Counter

"""
Function to convert integer i, 0 <= i < N, to a quantum state in Dirac notation.
"""
def state_as_string(i,N):
    if i < 0 or i >= 2**N:
        raise ValueError("Input i and N must satisfy 0 <= i < 2^N")

    binary_string = bin(i)
    state_as_string = binary_string[2:]
    state_as_string = state_as_string.zfill(N)
    return "|" + state_as_string + ">"

class Dirac:
    """
    Functions for the Dirac notation to describe (quantum) states and (quantum) operators.
    |a> is called 'ket' and represents a column vector with 1 in entry a and 0 everywhere else.
    <a| is called 'bra' and represents a row vector with 1 in entry a and 0 everywhere else.
    <a||b> is the inner product of <a| and |b>, which is 1 if a = b and 0 if a != b.
    |a><b| is the outer product of |a> and <b|, which is a matrix with 1 in entry (a,b) and 0 everywhere else.
    """
    @staticmethod
    def ket(N, a):
        ket = np.zeros((N, 1))
        ket[a, 0] = 1
        return ket

    @staticmethod
    def bra(N, a):
        bra = np.zeros((1, N))
        bra[0, a] = 1
        return bra

    @staticmethod
    def bra_ket(N, a, b):
        bra = Dirac.bra(N, a)
        ket = Dirac.ket(N, b)
        return np.inner(bra, ket.T)
    
    @staticmethod
    def ket_bra(N, a, b):
        ket = Dirac.ket(N, a)
        bra = Dirac.bra(N, b)
        return np.outer(ket, bra)


class QubitUnitaryOperation:
    """
    Functions to obtain 2 x 2 unitary matrices for unitary qubit operations.
    """
    @staticmethod
    def get_identity():
        return np.array([[1,0],[0,1]],dtype=complex)
    
    @staticmethod
    def get_pauli_x():
        return np.array([[0,1],[1,0]],dtype=complex)
    
    @staticmethod
    def get_pauli_y():
        return np.array([[0,complex(0,-1)],[complex(0,1),0]])
    
    @staticmethod
    def get_pauli_z():
        return np.array([[1,0],[0,-1]],dtype=complex)
    
    @staticmethod
    def get_hadamard():
        c = complex(1/np.sqrt(2),0)
        return np.array([[c,c],[c,-c]])
    
    @staticmethod
    def get_phase(theta):
        c = complex(np.cos(theta),np.sin(theta))
        return np.array([[1,0],[0,c]])


class CircuitUnitaryOperation:
    """
    Functions to obtain N x N unitary matrices for unitary operations on quantum circuits of N qubits.
    """
    @staticmethod
    def get_combined_operation_for_qubit(operation, q, N):
        identity = QubitUnitaryOperation.get_identity()
        combined_operation = np.eye(1,1)
        for i in range(0, N):
            if i == q:
                combined_operation = np.kron(combined_operation, operation)
            else:
                combined_operation = np.kron(combined_operation, identity)
        return combined_operation

    @staticmethod
    def get_combined_operation_for_identity(N):
        return np.array(np.eye(2**N),dtype=complex)
    
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
    def get_combined_operation_for_controlled_qubit_operation(operation, control, target, N):
        identity = QubitUnitaryOperation.get_identity()
        ket_bra_00 = Dirac.ket_bra(2,0,0)
        ket_bra_11 = Dirac.ket_bra(2,1,1)
        combined_operation_zero = np.eye(1,1)
        combined_operation_one = np.eye(1,1)
        for i in range (0, N):
            if control == i:
                combined_operation_zero = np.kron(combined_operation_zero, ket_bra_00)
                combined_operation_one  = np.kron(combined_operation_one, ket_bra_11)
            elif target == i:
                combined_operation_zero = np.kron(combined_operation_zero, identity)
                combined_operation_one  = np.kron(combined_operation_one, operation)
            else:
                combined_operation_zero = np.kron(combined_operation_zero, identity)
                combined_operation_one  = np.kron(combined_operation_one, identity)
            
        return combined_operation_zero + combined_operation_one
    
    @staticmethod
    def get_combined_operation_for_cnot(control, target, N):
        pauli_x = QubitUnitaryOperation.get_pauli_x()     
        return CircuitUnitaryOperation.get_combined_operation_for_controlled_qubit_operation(pauli_x, control, target, N)
    
    @staticmethod
    def get_combined_operation_for_controlled_pauli_y(control, target, N):
        pauli_y = QubitUnitaryOperation.get_pauli_y()
        return CircuitUnitaryOperation.get_combined_operation_for_controlled_qubit_operation(pauli_y, control, target, N)
    
    @staticmethod
    def get_combined_operation_for_controlled_pauli_z(control, target, N):
        pauli_z = QubitUnitaryOperation.get_pauli_z()
        return CircuitUnitaryOperation.get_combined_operation_for_controlled_qubit_operation(pauli_z, control, target, N)
    
    @staticmethod
    def get_combined_operation_for_controlled_hadamard(control, target, N):
        hadamard = QubitUnitaryOperation.get_hadamard()
        return CircuitUnitaryOperation.get_combined_operation_for_controlled_qubit_operation(hadamard, control, target, N)
    
    @staticmethod
    def get_combined_operation_for_controlled_phase(theta, control, target, N):
        phase_theta = QubitUnitaryOperation.get_phase(theta)
        return CircuitUnitaryOperation.get_combined_operation_for_controlled_qubit_operation(phase_theta, control, target, N)
    
    @staticmethod
    def get_combined_operation_for_swap(a, b, N):
        combined_operation_cnot_a_b = CircuitUnitaryOperation.get_combined_operation_for_cnot(a, b, N)
        combined_operation_cnot_b_a = CircuitUnitaryOperation.get_combined_operation_for_cnot(b, a, N)
        return np.dot(np.dot(combined_operation_cnot_a_b,combined_operation_cnot_b_a),combined_operation_cnot_a_b)
    
    @staticmethod
    def get_combined_operation_for_controlled_unitary_operation(operation):
        # Qubit 0 is the control
        identity = np.eye(*operation.shape)
        ket_bra_00 = Dirac.ket_bra(2,0,0)
        ket_bra_11 = Dirac.ket_bra(2,1,1)
        combined_operation_zero = np.kron(ket_bra_00,identity)
        combined_operation_one = np.kron(ket_bra_11,operation)
        return combined_operation_zero + combined_operation_one
    
    @staticmethod
    def get_combined_operation_for_controlled_unitary_operation_general(operation, control, target, N):
        # Qubit control is the control
        # Qubit target is the first qubit on which the unitary operation will be applied
        # N is total number of qubits (should be at least size of operation plus one)
        # control < target and target + size(operation) <= N
        identity = QubitUnitaryOperation.get_identity()
        ket_bra_00 = Dirac.ket_bra(2,0,0)
        ket_bra_11 = Dirac.ket_bra(2,1,1)
        identity_operation = np.eye(*operation.shape)
        combined_operation_zero = np.eye(1,1)
        combined_operation_one = np.eye(1,1)
        i = 0
        while i < N:
            if control == i:
                #combined_operation_zero = np.kron(combined_operation_zero, ket_bra_00)
                #combined_operation_one  = np.kron(combined_operation_one, ket_bra_11)
                combined_operation_zero = np.kron(ket_bra_00, combined_operation_zero)
                combined_operation_one  = np.kron(ket_bra_11, combined_operation_one)
                i = i + 1
            elif target == i:
                #combined_operation_one  = np.kron(combined_operation_one, operation)
                #combined_operation_zero = np.kron(combined_operation_zero, identity_operation)
                combined_operation_zero = np.kron(identity_operation, combined_operation_zero)
                combined_operation_one  = np.kron(operation, combined_operation_one)
                i = i + math.log(operation.shape[0],2)
            else:
                #combined_operation_zero = np.kron(combined_operation_zero, identity)
                #combined_operation_one  = np.kron(combined_operation_one, identity)
                combined_operation_zero = np.kron(identity, combined_operation_zero)
                combined_operation_one  = np.kron(identity, combined_operation_one)
                i = i + 1
        return combined_operation_zero + combined_operation_one
    
    @staticmethod
    def get_combined_operation_for_unitary_operation_general(operation, target, N):
        # Qubit target is the first qubit on which the unitary operation will be applied
        # N is total number of qubits (should be at least size of operation)
        # 0 <= target < N
        identity = QubitUnitaryOperation.get_identity()
        combined_operation = np.eye(1,1)
        i = 0
        while i < N:
            if target == i:
                combined_operation = np.kron(combined_operation, operation)
                i = i + math.log(operation.shape[0],2)
            else:
                combined_operation = np.kron(combined_operation, identity)
                i = i + 1
        return combined_operation

    @staticmethod
    def get_combined_operation_for_fredkin(control, a, b, N):
        combined_operation_swap_control_0 = CircuitUnitaryOperation.get_combined_operation_for_swap(control, 0, N)
        combined_operation_swap_a_b = CircuitUnitaryOperation.get_combined_operation_for_swap(a-1, b-1, N-1)
        combined_operation_fredkin = CircuitUnitaryOperation.get_combined_operation_for_controlled_unitary_operation(combined_operation_swap_a_b)
        return np.dot(np.dot(combined_operation_swap_control_0, combined_operation_fredkin), combined_operation_swap_control_0)
    
    @staticmethod
    def get_combined_operation_for_toffoli(control, a, b, N):
        combined_operation_swap_control_0 = CircuitUnitaryOperation.get_combined_operation_for_swap(control, 0, N)
        combined_operation_cnot_a_b = CircuitUnitaryOperation.get_combined_operation_for_cnot(a-1, b-1, N-1)
        combined_operation_toffoli = CircuitUnitaryOperation.get_combined_operation_for_controlled_unitary_operation(combined_operation_cnot_a_b)
        return np.dot(np.dot(combined_operation_swap_control_0, combined_operation_toffoli), combined_operation_swap_control_0)
    
    @staticmethod
    def get_combined_operation_for_multi_controlled_pauli_z_operation(N):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_identity(N)
        combined_operation[2**N-1,2**N-1] = -combined_operation[2**N-1,2**N-1]
        return combined_operation
    
    @staticmethod
    def get_combined_operation_for_multi_controlled_pauli_x_operation(N):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_identity(N)
        combined_operation[2**N-2,2**N-2] = 1 - combined_operation[2**N-2,2**N-2]
        combined_operation[2**N-2,2**N-1] = 1 - combined_operation[2**N-2,2**N-1]
        combined_operation[2**N-1,2**N-2] = 1 - combined_operation[2**N-1,2**N-2]
        combined_operation[2**N-1,2**N-1] = 1 - combined_operation[2**N-1,2**N-1]
        return combined_operation


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
        if not np.allclose(np.eye(2**self.N), np.dot(np.conj(operation.T), operation)):
            raise ValueError("Input matrix is not unitary")
        self.state_vector = np.dot(operation, self.state_vector)

    def measure(self):
        probalities = np.square(np.abs(self.state_vector)).flatten()
        self.index = np.random.choice(len(probalities), p=probalities)

    def get_quantum_state(self):
        return self.state_vector

    def get_classical_state_as_string(self):
        return state_as_string(self.index, self.N)
    
    def print(self):
        for i, val in enumerate(self.state_vector):
            print(f"{state_as_string(i,self.N)} : {val[0]}")


class Circuit:
    """
    Class representing a quantum circuit of N qubits.
    """
    def __init__(self,N):
        self.N = N
        self.state_vector = StateVector(self.N)
        self.quantum_states = [self.state_vector.get_quantum_state()]
        self.descriptions = []
        self.operations = []

    def identity(self, q):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_identity(q, self.N)
        self.descriptions.append(f"Identity on qubit {q}")
        self.operations.append(combined_operation)

    def pauli_x(self, q):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_pauli_x(q, self.N)
        self.descriptions.append(f"Pauli X on qubit {q}")
        self.operations.append(combined_operation)

    def pauli_y(self, q):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_pauli_y(q, self.N)
        self.descriptions.append(f"Pauli Y on qubit {q}")
        self.operations.append(combined_operation)

    def pauli_z(self, q):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_pauli_z(q, self.N)
        self.descriptions.append(f"Pauli Z on qubit {q}")
        self.operations.append(combined_operation)

    def hadamard(self, q):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_hadamard(q, self.N)
        self.descriptions.append(f"Hadamard on qubit {q}")
        self.operations.append(combined_operation)

    def phase(self, theta, q):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_phase(theta, q, self.N)
        pi_symbol = '\u03c0'
        self.descriptions.append(f"Phase with theta = {theta/np.pi:.3f} {pi_symbol} on qubit {q}")
        self.operations.append(combined_operation)

    def cnot(self, control, target):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_cnot(control, target, self.N)
        self.descriptions.append(f"CNOT with control qubit {control} and target qubit {target}")
        self.operations.append(combined_operation)

    def controlled_pauli_y(self, control, target):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_controlled_pauli_y(control, target, self.N)
        self.descriptions.append(f"Controlled Pauli Y with control qubit {control} and target qubit {target}")
        self.operations.append(combined_operation)

    def controlled_pauli_z(self, control, target):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_controlled_pauli_z(control, target, self.N)
        self.descriptions.append(f"Controlled Pauli Z with control qubit {control} and target qubit {target}")
        self.operations.append(combined_operation)
    
    def controlled_hadamard(self, control, target):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_controlled_hadamard(control, target, self.N)
        self.descriptions.append(f"Controlled Hadamard with control qubit {control} and target qubit {target}")
        self.operations.append(combined_operation)

    def controlled_phase(self, theta, control, target):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_controlled_phase(theta, control, target, self.N)
        pi_symbol = '\u03c0'
        self.descriptions.append(f"Controlled phase with theta = {theta/np.pi:.3f} {pi_symbol}, control qubit {control}, and target qubit {target}")
        self.operations.append(combined_operation)

    def swap(self, a, b):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_swap(a, b, self.N)
        self.descriptions.append(f"SWAP on qubit {a} and qubit {b}")
        self.operations.append(combined_operation)
        
    def fredkin(self, control, a, b):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_fredkin(control, a, b, self.N)
        self.descriptions.append(f"Fredkin with control qubit {control} and SWAP on qubit {a} and qubit {b}")
        self.operations.append(combined_operation)
    
    def toffoli(self, control_a, control_b, target):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_toffoli(control_a, control_b, target, self.N)
        self.descriptions.append(f"Toffoli with control qubit {control_a} and CNOT with control qubit {control_b} and target qubit {target}")
        self.operations.append(combined_operation)
    
    def multi_controlled_pauli_z(self):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_multi_controlled_pauli_z_operation(self.N)
        self.descriptions.append(f"Multi-controlled Pauli_Z")
        self.operations.append(combined_operation)

    def multi_controlled_pauli_x(self):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_multi_controlled_pauli_x_operation(self.N)
        self.descriptions.append(f"Multi-controlled Pauli_X")
        self.operations.append(combined_operation)

    def create_inverse_circuit(self):
        inverse_circuit = Circuit(self.N)
        for operation, description in zip(reversed(self.operations), reversed(self.descriptions)):
            inverse_circuit.operations.append(operation)
            inverse_circuit.descriptions.append(description)
        return inverse_circuit
    
    def append_circuit(self, circuit):
        if circuit.N != self.N:
            raise ValueError("Function append_circuit: circuit to be appended must have same number of qubits")
        for operation, description in zip(circuit.operations, circuit.descriptions):
            self.operations.append(operation)
            self.descriptions.append(description)
    
    def print_circuit(self):
        for description in self.descriptions:
            print(description)
        
    def execute(self, print_state=False):
        self.state_vector = StateVector(self.N)
        self.quantum_states = [self.state_vector.get_quantum_state()]
        if print_state:
            print("Initial quantum state")
            self.state_vector.print()
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


"""
Function to run a quantum circuit and measure the classical state.
"""
def run_circuit(circuit:Circuit, nr_runs=1000):
    result = []
    for i in range(nr_runs):
        circuit.execute()
        circuit.measure()
        result.append(circuit.get_classical_state_as_string())
    return result


"""
Function to run a quantum circuit once and measure the classical state many times.
"""
def measure_circuit(circuit:Circuit, nr_measurements=1000):
    circuit.execute()
    result = []
    for i in range(nr_measurements):
        circuit.measure()
        result.append(circuit.get_classical_state_as_string())
    return result


"""
Function to plot a histogram of all classical states after executing the circuit multiple times.
"""
def histogram_of_classical_states(string_array):
    histogram = Counter(string_array)
    unique_strings = sorted(list(histogram.keys()))
    counts = [histogram[string] for string in unique_strings]
    plt.bar(unique_strings, counts)
    if len(histogram) > 8:
        plt.xticks(rotation='vertical')
    plt.xlabel('Classical states')
    plt.ylabel('Nr occurrences')
    plt.title('Number of occurrences of classical states')
    plt.show()


"""
Function to plot a all intermediate (quantum) states of the last execution of a circuit.
"""
def show_all_intermediate_states(circuit:Circuit):
    matrix_of_all_states = np.zeros((2**circuit.N, len(circuit.quantum_states)), dtype=complex)
    i = 0
    for state_vector in circuit.quantum_states:
        matrix_of_all_states[:,i] = state_vector.flatten()
        i = i + 1

    fig_width  = 4 + circuit.N
    fig_height = 4 + 0.5*len(circuit.operations)
    fig, ax = plt.subplots()
    fig.set_size_inches(fig_width, fig_height)
    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    radius_circle = 0.45
    length_arrow = 0.4
    color_map = plt.get_cmap('jet')

    for (x, y), c in np.ndenumerate(matrix_of_all_states):
        r = abs(c)
        phase = cmath.phase(c)
        color = color_map(int(r*256))
        circle = plt.Circle([x + 0.5, y + 0.5], radius_circle, facecolor=color, edgecolor='black')
        dx = length_arrow * np.cos(phase)
        dy = length_arrow * np.sin(phase)
        arrow = plt.Arrow(x + 0.5 - dx, y + 0.5 - dy, 2*dx, 2*dy, facecolor='lightgray', edgecolor='black')
        ax.add_patch(circle)
        ax.add_patch(arrow)

    ax.autoscale_view()
    ax.invert_yaxis()

    positions_x = []
    all_states_as_string = []
    for i in range(0,2**circuit.N):
        positions_x.append(i + 0.5)
        all_states_as_string.append(state_as_string(i,circuit.N))
    plt.xticks(positions_x, all_states_as_string, rotation='vertical')

    j = 0.5
    positions_y = [j]
    all_operations_as_string = ['Initial state']
    j = j + 1
    for description in circuit.descriptions:
        positions_y.append(j)
        all_operations_as_string.append(description)
        j = j + 1
    plt.yticks(positions_y, all_operations_as_string)


"""
Function to plot a all intermediate probabilities of the last execution of a circuit.
"""
def show_all_pobrabilities(circuit:Circuit):
    matrix_of_probabilities = np.zeros((2**circuit.N,len(circuit.quantum_states)))
    i = 0
    for state_vector in circuit.quantum_states:
        probalities = np.square(np.abs(state_vector)).flatten()
        matrix_of_probabilities[:,i] = probalities
        i = i + 1

    fig_width  = 4 + circuit.N
    fig_height = 4 + 0.5*len(circuit.operations)
    fig, ax = plt.subplots()
    fig.set_size_inches(fig_width, fig_height)
    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    size = 0.9
    color_map = plt.get_cmap('jet')

    for (x, y), w in np.ndenumerate(matrix_of_probabilities):
        color = color_map(int(w*256))
        rect = plt.Rectangle([x - size/2, y - size/2], size, size,
                            facecolor=color, edgecolor='black')
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()
        
    positions_x = []
    all_states_as_string = []
    for i in range(0,2**circuit.N):
        positions_x.append(i)
        all_states_as_string.append(state_as_string(i, circuit.N))
    plt.xticks(positions_x, all_states_as_string, rotation='vertical')

    positions_y = [0]
    all_operations_as_string = ['Initial state']
    j = 1
    for description in circuit.descriptions:
        positions_y.append(j)
        all_operations_as_string.append(description)
        j = j + 1
    plt.yticks(positions_y, all_operations_as_string)