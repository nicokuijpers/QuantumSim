# TODO
# > If matrices are too big, return to CPU processing, no GPU acceleration

try:
    import cupy
    import cupyx.scipy.sparse as cupysparse
    GPU_AVAILABLE = True
except:
    print("[ERROR] Cupy could not be imported, make sure that your have installed Cupy")
    print("\tIf you do not have a NVIDIA GPU, you cannot install Cupy")
    print("\tQuantumsim will still work accordingly, just less performant")
    print("\t > Installation guide: https://docs.cupy.dev/en/stable/install.html")
    GPU_AVAILABLE = False
finally:
    from numba import njit 
    from typing import Union
    import scipy.sparse as sparse
    import numpy as np
    import matplotlib.pyplot as plt
    import cmath
    import math
    from collections import Counter
    import matplotlib.colors as mcol

'''
Symbol for pi
'''
pi_symbol = '\u03c0'

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
    
    @staticmethod
    def state_as_string(i, N) -> str:
        if i < 0 or i >= 2**N:
            raise ValueError("Input i and N must satisfy 0 <= i < 2^N")

        binary_string = bin(i)
        state_as_string = binary_string[2:].zfill(N)
        return "|" + state_as_string + ">"

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

        # NOTE: Statevector normally is column-based, I made a row-based.
        # np.zeros((2**self.N,1 ), dtype=complex)
        self.state_vector = np.zeros(2**self.N, dtype=complex)

        self.state_vector[self.index] = 1

    def apply_unitary_operation(self, operation: sparse.coo_matrix):
        # Check if operation is a unitary matrix
        # if not np.allclose(np.eye(2**self.N), np.conj(operation.T) @ operation):
        #     raise ValueError("Input matrix is not unitary")

        # NOTE: A row based statevector is roughly 15% faster than matrix-vector multiplication than a column based statevector
        # print(timeit(lambda: coo_spmv_row(operation.row, operation.col, operation.data, self.state_vector.flatten()), number=100))
        # print(timeit(lambda: coo_spmv_column(operation.row, operation.col, operation.data, self.state_vector), number=100))

        self.state_vector = coo_spmv_row(operation.row, operation.col, operation.data, self.state_vector)
        
    def measure_x(self, q):
        # Compute the real part of <psi|X|psi>
        X = CircuitUnitaryOperation.get_combined_operation_for_pauli_x(q, self.N)
        return np.vdot(self.state_vector, X.dot(self.state_vector)).real
    
    def measure_y(self, q):
        # Compute the real part of <psi|Y|psi>
        Y = CircuitUnitaryOperation.get_combined_operation_for_pauli_y(q, self.N)
        return np.vdot(self.state_vector, Y.dot(self.state_vector)).real

    def measure_z(self, q):
        # Compute the real part of <psi|Z|psi>
        Z = CircuitUnitaryOperation.get_combined_operation_for_pauli_z(q, self.N)
        return np.vdot(self.state_vector, Z.dot(self.state_vector)).real
    
    def measure(self):
        probalities = np.square(np.abs(self.state_vector))
        self.index = np.random.choice(len(probalities), p=probalities)

    def get_quantum_state(self):
        return self.state_vector
    
    def get_classical_state_as_string(self):
        return self.__state_as_string(self.index, self.N)
    
    def print(self):
        for i, val in enumerate(self.state_vector):
            print(f"{self.__state_as_string(i, self.N)} : {val}")

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
    def get_combined_operation_for_qubit(operation, q, N, gpu=False):
        # FIXME: If matrices are too big of GPU memory, it will turn of GPU computation
        # if gpu:
        #     mempool = cupy.get_default_memory_pool()
        #     bytes = mempool.total_bytes()

        #     # Checks whether matrix fits on GPU memory, if not, calculations will be done on GPU
        #     # Formula to calculate memory usage for haramard (most memory intensive) for N qubits 
        #     # bytes = 96*2^(N-1)
        #     gpu = (bytes/2) > 96*2**(N-2)

        # Converting dense numpy matrixes to sparse COO scipy matrixes
        operation =  sparse.coo_matrix(operation)
        identity = sparse.coo_matrix(QubitUnitaryOperation.get_identity())
        combined_operation = sparse.coo_matrix(np.eye(1,1))

        # "Selecting" regular scipy sparse matrix kronecker product
        kron = coo_kron

        if gpu:
            # Copy data to device (GPU) memory from host (CPU)
            operation = cupysparse.coo_matrix(operation)
            identity = cupysparse.coo_matrix(identity)
            combined_operation = cupysparse.coo_matrix(combined_operation)

            # "Selecting" sparse matrix GPU-accelerated matrix kronecker product
            kron = coo_kron_gpu
    	
        # Actual computation of kronecker product, this is sort of a iterative problem.
        # Size of "combined_operation" grows exponentially
        # Every qubit makes the kronecker product twice as sparse
        # Computation is done on GPU based on whether parater "GPU" is "True"
        for i in range(0, N):
            if i == q:
                combined_operation = kron(combined_operation, operation)
            else:
                combined_operation = kron(combined_operation, identity)
        
        # Copy data back from device (GPU) to host (CPU)
        if gpu: combined_operation = combined_operation.get()

        return combined_operation

    @staticmethod
    def get_combined_operation_for_identity(q, N, gpu=False):
        return np.array(np.eye(2**N), dtype=complex)
    
    @staticmethod
    def get_combined_operation_for_pauli_x(q, N, gpu=False):
        pauli_x = QubitUnitaryOperation.get_pauli_x()
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(pauli_x, q, N, gpu=gpu)
    
    @staticmethod
    def get_combined_operation_for_pauli_y(q, N, gpu=False):
        pauli_y = QubitUnitaryOperation.get_pauli_y()
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(pauli_y, q, N, gpu=gpu)
    
    @staticmethod
    def get_combined_operation_for_pauli_z(q, N, gpu=False):
        pauli_z = QubitUnitaryOperation.get_pauli_z()
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(pauli_z, q, N, gpu=gpu)
    
    @staticmethod
    def get_combined_operation_for_hadamard(q, N, gpu=False):
        hadamard = QubitUnitaryOperation.get_hadamard()
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(hadamard, q, N, gpu=gpu)
    
    @staticmethod
    def get_combined_operation_for_phase(theta, q, N, gpu=False):
        phase = QubitUnitaryOperation.get_phase(theta)
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(phase, q, N, gpu=gpu)
    
    @staticmethod
    def get_combined_operation_for_rotate_x(theta, q, N, gpu=False):
        rotate = QubitUnitaryOperation.get_rotate_x(theta)
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(rotate, q, N, gpu=gpu)
    
    @staticmethod
    def get_combined_operation_for_rotate_y(theta, q, N, gpu=False):
        rotate = QubitUnitaryOperation.get_rotate_y(theta)
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(rotate, q, N, gpu=gpu)
    
    @staticmethod
    def get_combined_operation_for_rotate_z(theta, q, N, gpu=False):
        rotate = QubitUnitaryOperation.get_rotate_z(theta)
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(rotate, q, N, gpu=gpu)
    
    @staticmethod
    def get_combined_operation_for_swap(a, b, N):
        combined_operation_cnot_a_b = CircuitUnitaryOperation.get_combined_operation_for_cnot(a, b, N)
        combined_operation_cnot_b_a = CircuitUnitaryOperation.get_combined_operation_for_cnot(b, a, N)

        return (combined_operation_cnot_a_b * combined_operation_cnot_b_a) * combined_operation_cnot_a_b
        # return np.dot(np.dot(combined_operation_cnot_a_b,combined_operation_cnot_b_a),combined_operation_cnot_a_b)

    @staticmethod
    def get_combined_operation_for_cnot(control, target, N, gpu=False):
        # Converting dense numpy matrixes to sparse COO scipy matrixes
        identity = sparse.coo_matrix(QubitUnitaryOperation.get_identity())
        pauli_x = sparse.coo_matrix(QubitUnitaryOperation.get_pauli_x())
        ket_bra_00 = sparse.coo_matrix(Dirac.ket_bra(2,0,0))
        ket_bra_11 = sparse.coo_matrix(Dirac.ket_bra(2,1,1))
        combined_operation_zero = sparse.coo_matrix(np.eye(1,1))
        combined_operation_one = sparse.coo_matrix(np.eye(1,1))
    
        # "Selecting" regular scipy sparse matrix kronecker product
        kron = coo_kron

        if gpu:
            # Copy data to device (GPU) memory from host (CPU)
            identity = cupysparse.coo_matrix(identity)
            pauli_x = cupysparse.coo_matrix(pauli_x)
            ket_bra_00 = cupysparse.coo_matrix(ket_bra_00)
            ket_bra_11 = cupysparse.coo_matrix(ket_bra_11)
            combined_operation_zero = cupysparse.coo_matrix(combined_operation_zero)
            combined_operation_one = cupysparse.coo_matrix(combined_operation_one)

            # "Selecting" sparse matrix GPU-accelerated matrix kronecker product
            kron = coo_kron_gpu

        # Actual computation of kronecker product, this is sort of a iterative problem.
        # Size of "combined_operation" grows exponentially
        # Every qubit makes the kronecker product twice as sparse
        # Computation is done on GPU based on whether parater "GPU" is "True"
        for i in range(0, N):
            if control == i:
                combined_operation_zero = kron(combined_operation_zero, ket_bra_00)
                combined_operation_one  = kron(combined_operation_one, ket_bra_11)
            elif target == i:
                combined_operation_zero = kron(combined_operation_zero, identity)
                combined_operation_one  = kron(combined_operation_one, pauli_x)
            else:
                combined_operation_zero = kron(combined_operation_zero, identity)
                combined_operation_one  = kron(combined_operation_one, identity)

        operation = combined_operation_zero + combined_operation_one
        # Copy data back from device (GPU) to host (CPU)
        if gpu: operation = operation.get()
        operation = sparse.coo_matrix(operation)
        
        return operation

class Circuit:
    """
    Class representing a quantum circuit of N qubits.
    """
    def __init__(self, N, use_cache=False, use_GPU=False, use_lazy=False, disk=False):
        self.N = N
        self.state_vector = StateVector(self.N)
        self.quantum_states = [self.state_vector.get_quantum_state()]
        self.descriptions = []
        self.gates = []
        self.operations: Union[list[function], list[sparse.coo_matrix]] = []
        
        # Optimization flags
        self.use_gpu = use_GPU and GPU_AVAILABLE # Only use GPU if available and enabled for use by user.
        self.lazy_evaluation = use_lazy
        self.use_cache = use_cache
        self.operations_cache = {}
        
        if use_cache:
            if use_lazy: print("[Warning] Lazy evaluation and caching cannot be both switched on. Caching is off, lazy evaluation is on")
            self.use_cache = not use_lazy
        else:
            self.use_cache = False

        if not GPU_AVAILABLE and use_GPU:
            print("[Warning] GPU will not be used. 'use_GPU' is set to 'True', but GPU is not available.")

        # "Warming up" the function, calling it compiles the function using Numba
        coo_spmv_row(np.array([0], dtype=np.int32), 
                     np.array([0], dtype=np.int32), 
                     np.array([0], dtype=np.complex128), 
                     np.array([0], dtype=np.complex128))

    def identity(self, q):
        key = (False, "identity", q)
        description = f"Hadamard on qubit {q}"
        self.descriptions.append(description)
        gate_as_string = '.' * self.N
        self.gates.append(gate_as_string)

        if self.lazy_evaluation:
            l = lambda: CircuitUnitaryOperation.get_combined_operation_for_identity(q, self.N, gpu=self.use_gpu)
            self.operations.append(l)
            return

        if self.use_cache and key in self.operations_cache: 
            self.descriptions.append(description)
            self.operations.append(self.operations_cache[key])
            return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_identity(q, self.N, gpu=self.use_gpu)
        self.operations.append(combined_operation)

        if self.use_cache:
            self.operations_cache[key] = combined_operation

    def pauli_x(self, q):
        key = (False, "pauli_x", q)
        description = f"pauli_x on qubit {q}"
        self.descriptions.append(description)
        gate_as_string = '.' * q + 'X' + '.' * (self.N - q - 1)
        self.gates.append(gate_as_string)

        if self.lazy_evaluation:
            l = lambda: CircuitUnitaryOperation.get_combined_operation_for_pauli_x(q, self.N, gpu=self.use_gpu)
            self.operations.append(l)
            return

        if self.use_cache and key in self.operations_cache: 
            self.descriptions.append(description)
            self.operations.append(self.operations_cache[key])
            return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_pauli_x(q, self.N, gpu=self.use_gpu)
        self.operations.append(combined_operation)

        if self.use_cache:
            self.operations_cache[key] = combined_operation

    def pauli_y(self, q):
        key = (False, "pauli_y", q)
        description = f"pauli_y on qubit {q}"
        self.descriptions.append(description)
        gate_as_string = '.' * q + 'Y' + '.' * (self.N - q - 1)
        self.gates.append(gate_as_string)

        if self.lazy_evaluation:
            l = lambda: CircuitUnitaryOperation.get_combined_operation_for_pauli_y(q, self.N, gpu=self.use_gpu)
            self.operations.append(l)
            return
        
        if self.use_cache and key in self.operations_cache: 
            self.operations.append(self.operations_cache[key])
            return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_pauli_y(q, self.N, gpu=self.use_gpu)
        self.operations.append(combined_operation)

        if self.use_cache:
            self.operations_cache[key] = combined_operation

    def pauli_z(self, q):
        key = (False, "pauli_z", q)
        description = f"pauli_z on qubit {q}"
        self.descriptions.append(description)
        gate_as_string = '.' * q + 'Z' + '.' * (self.N - q - 1)
        self.gates.append(gate_as_string)

        if self.lazy_evaluation:
            l = lambda: CircuitUnitaryOperation.get_combined_operation_for_pauli_z(q, self.N, gpu=self.use_gpu)
            self.operations.append(l)
            return
 
        if self.use_cache and key in self.operations_cache: 
            self.operations.append(self.operations_cache[key])
            return
        
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_pauli_z(q, self.N, gpu=self.use_gpu)
        self.operations.append(combined_operation)

        if self.use_cache:
            self.operations_cache[key] = combined_operation

    def hadamard(self, q):
        key = (False, "hadamard", q)
        description = f"Hadamard on qubit {q}"
        self.descriptions.append(description)
        gate_as_string = '.' * q + 'H' + '.' * (self.N - q - 1)
        self.gates.append(gate_as_string)

        if self.lazy_evaluation:
            l = lambda: CircuitUnitaryOperation.get_combined_operation_for_hadamard(q, self.N, gpu=self.use_gpu)
            self.operations.append(l)
            return
 
        if self.use_cache and key in self.operations_cache: 
            self.operations.append(self.operations_cache[key])
            return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_hadamard(q, self.N, gpu=self.use_gpu)
        self.operations.append(combined_operation)

        if self.use_cache:
            self.operations_cache[key] = combined_operation

    def phase(self, theta, q):
        key = (False, "phase", theta, q)
        description = f"Phase with theta = {theta/np.pi:.3f} {pi_symbol} on qubit {q}"
        self.descriptions.append(description)
        gate_as_string = '.' * q + 'S' + '.' * (self.N - q - 1)
        self.gates.append(gate_as_string)
        
        if self.lazy_evaluation:
            l = lambda: CircuitUnitaryOperation.get_combined_operation_for_phase(theta, q, self.N, gpu=self.use_gpu)
            self.operations.append(l)
            return
 
        if self.use_cache and key in self.operations_cache: 
            self.operations.append(self.operations_cache[key])
            return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_phase(theta, q, self.N, gpu=self.use_gpu)
        self.operations.append(combined_operation)
 
        if self.use_cache:
            self.operations_cache[key] = combined_operation

    def rotate_x(self, theta, q):
        key = (False, "rotate_x", theta, q)
        description = f"Rotate X with theta = {theta/np.pi:.3f} {pi_symbol} on qubit {q}"
        self.descriptions.append(description)
        gate_as_string = '.' * q + 'R' + '.' * (self.N - q - 1)
        self.gates.append(gate_as_string)
        
        if self.lazy_evaluation:
            l = lambda: CircuitUnitaryOperation.get_combined_operation_for_rotate_x(theta, q, self.N, gpu=self.use_gpu)
            self.operations.append(l)
            return
 
        if self.use_cache and key in self.operations_cache: 
            self.operations.append(self.operations_cache[key])
            return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_rotate_x(theta, q, self.N, gpu=self.use_gpu)
        self.operations.append(combined_operation)
 
        if self.use_cache:
            self.operations_cache[key] = combined_operation
    
    def rotate_y(self, theta, q):
        key = (False, "rotate_y", theta, q)
        description = f"Rotate_y with theta = {theta/np.pi:.3f} {pi_symbol} on qubit {q}"
        self.descriptions.append(description)
        gate_as_string = '.' * q + 'R' + '.' * (self.N - q - 1)
        self.gates.append(gate_as_string)

        if self.lazy_evaluation:
            l = lambda: CircuitUnitaryOperation.get_combined_operation_for_rotate_y(theta, q, self.N, gpu=self.use_gpu)
            self.operations.append(l)
            return
 
        if self.use_cache and key in self.operations_cache: 
            self.operations.append(self.operations_cache[key])
            return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_rotate_y(theta, q, self.N, gpu=self.use_gpu)
        self.operations.append(combined_operation)
 
        if self.use_cache:
            self.operations_cache[key] = combined_operation
    
    def rotate_z(self, theta, q):
        key = (False, "rotate_z", theta, q)
        description = f"Rotate_z with theta = {theta/np.pi:.3f} {pi_symbol} on qubit {q}"
        self.descriptions.append(description)
        gate_as_string = '.' * q + 'R' + '.' * (self.N - q - 1)
        self.gates.append(gate_as_string)

        if self.lazy_evaluation:
            l = lambda: CircuitUnitaryOperation.get_combined_operation_for_rotate_z(theta, q, self.N, gpu=self.use_gpu)
            self.operations.append(l)
            return
 
        if self.use_cache and key in self.operations_cache: 
            self.operations.append(self.operations_cache[key])
            return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_rotate_z(theta, q, self.N, gpu=self.use_gpu)
        self.operations.append(combined_operation)
 
        if self.use_cache:
            self.operations_cache[key] = combined_operation

    def cnot(self, control, target):
        key = (False, "cnot", control, target)
        description = f"CNOT with control qubit {control} and target qubit {target}"
        self.descriptions.append(description)
        gate_as_string = ''.join('*' if i == control else 'X' if i == target else '.' for i in range(self.N))
        self.gates.append(gate_as_string)

        if self.lazy_evaluation:
            l = lambda: CircuitUnitaryOperation.get_combined_operation_for_cnot(control, target, self.N, gpu=self.use_gpu)
            self.operations.append(l)
            return
 
        if self.use_cache and key in self.operations_cache: 
            self.operations.append(self.operations_cache[key])
            return

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_cnot(control, target, self.N, gpu=self.use_gpu)
        self.operations.append(combined_operation)
 
        if self.use_cache:
            self.operations_cache[key] = combined_operation

    def execute(self, print_state=False):
        self.state_vector = StateVector(self.N)
        if print_state:
            print("Initial quantum state")
            self.state_vector.print()

        # Checking variable type (based on if lazy flag is True) is correct
        assert isinstance(self.operations, list), "Operations should be a list"
        if self.lazy_evaluation == True:
            assert all(isinstance(op, type(lambda: None)) for op in self.operations), "Operation matrices are lazely evaluated but the operations list is not a list of functions"
        else:
            assert all(isinstance(op, sparse.coo_matrix) for op in self.operations), "Operation matrices are evaluated but the operations list is not a list of coo_matrix"

        for operation, description in zip(self.operations, self.descriptions):
            if self.lazy_evaluation: operation = operation()

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
        
    def print_circuit(self):
        for description in self.descriptions:
            print(description)

@njit
def coo_spmv_column(rowIdx, colIdx, values, v):
    """
    Performs sparse matrix-vector (column based) multiplication using COO format.
    
    Parameters:
    - rowIdx (list[int]): Row indices of nonzero elements.
    - colIdx (list[int]): Column indices of nonzero elements.
    - values (list[float]): Nonzero values of the matrix.
    - v (numpy array): Dense vector for multiplication.
    
    Returns:
    - numpy array: Result vector y = A * v
    """
    out = np.zeros((len(v), 1), dtype=values.dtype)  # Initialize output vector
    nnz = len(values)  # Number of nonzero elements

    for i in range(nnz):  # Iterate over nonzero elements
        out[rowIdx[i], 0] += values[i] * v[colIdx[i], 0]

    return out

@njit
def coo_spmv_row(rowIdx, colIdx, values, v):
    """
    Performs sparse matrix-vector (row based) multiplication using COO format.
    
    Parameters:
    - rowIdx (list[int]): Row indices of nonzero elements.
    - colIdx (list[int]): Column indices of nonzero elements.
    - values (list[float]): Nonzero values of the matrix.
    - v (numpy array): Dense vector for multiplication.
    
    Returns:
    - numpy array: Result vector y = A * v
    """
    out = np.zeros(len(v), dtype=values.dtype)  # Initialize output vector
    nnz = len(values)  # Number of nonzero elements

    for i in range(nnz):  # Iterate over nonzero elements
        out[rowIdx[i]] += values[i] * v[colIdx[i]]

    return out


# Based on the scipy implementation
# Source: https://github.com/scipy/scipy/blob/v1.15.1/scipy/sparse/_construct.py#L458
# Docs: https://docs.scipy.org/doc/scipy-1.15.1/reference/generated/scipy.sparse.kron.html
def coo_kron(A:sparse.coo_matrix, B:sparse.coo_matrix, format='coo'):
    output_shape = (A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])

    if A.nnz == 0 or B.nnz == 0:
        # kronecker product is the zero matrix
        return sparse.coo_matrix(output_shape).asformat(format)

    # Expand entries of a into blocks
    # When using more then 32 qubits, increase to int64
    row = np.asarray(A.row, dtype=np.int32).repeat(B.nnz)
    col = np.asarray(A.col, dtype=np.int32).repeat(B.nnz)
    data = A.data.repeat(B.nnz)
    
    row *= B.shape[0]
    col *= B.shape[1]

    # increment block indices
    row = row.reshape(-1, B.nnz)
    row += B.row
    row = row.reshape(-1)

    col = col.reshape(-1, B.nnz)
    col += B.col
    col = col.reshape(-1)

    # compute block entries
    data = data.reshape(-1, B.nnz) * B.data
    data = data.reshape(-1)

    return sparse.coo_matrix((data, (row, col)), shape=output_shape).asformat(format)

try:
    # Based on the Cupy implementation
    # Source: https://github.com/cupy/cupy/blob/v13.4.1/cupyx/scipy/sparse/_construct.py#L496
    # Docs: https://docs.cupy.dev/en/v13.4.1/reference/generated/cupyx.scipy.sparse.kron.html
    def coo_kron_gpu(A:cupysparse.coo_matrix, B:cupysparse.coo_matrix, format='coo'):
        out_shape = (A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])

        if A.nnz == 0 or B.nnz == 0:
            # kronecker product is the zero matrix
            return cupysparse.coo_matrix(out_shape).asformat(format)

        # expand entries of A into blocks
        row = A.row.astype(cupy.int32, copy=True) * B.shape[0]
        row = row.repeat(B.nnz)
        col = A.col.astype(cupy.int32, copy=True) * B.shape[1]
        col = col.repeat(B.nnz)
        data = A.data.repeat(B.nnz) 

        # increment block indices
        row = row.reshape(-1, B.nnz)
        row += B.row
        row = row.ravel()

        col = col.reshape(-1, B.nnz)
        col += B.col
        col = col.ravel()

        # compute block entries
        data = data.reshape(-1, B.nnz) * B.data
        data = data.ravel()

        return cupysparse.coo_matrix(
            (data, (row, col)), shape=out_shape).asformat(format)
except NameError:
    pass
except Exception as e:
    # print(e)
    exit(1)

@staticmethod
def execute_circuit(circuit:Circuit, nr_executions=100):
    """
    Function to run a quantum circuit and measure the classical state.
    """
    result = []
    for i in range(nr_executions):
        circuit.execute()
        result.append(circuit.state_vector.get_quantum_state())
    return result


@staticmethod
def run_circuit(circuit:Circuit, nr_runs=1000):
    """
    Function to run a quantum circuit and measure the classical state.
    """
    # if(circuit.save_instructions):
    #     raise Exception("Direct Operation Execution is enabled, QuantumUtil not supported with this flag")
    result = []
    for i in range(nr_runs):
        circuit.execute()
        circuit.measure()
        result.append(circuit.get_classical_state_as_string())
    return result


@staticmethod
def measure_circuit(circuit:Circuit, nr_executes=1, nr_measurements=1000, little_endian_formatted: bool=False):
    """
    Function to run a quantum circuit once and measure the classical state many times.
    """
    result = []
    for _ in range(nr_executes):
        circuit.execute()      
        for i in range(int(nr_measurements/nr_executes)):
            circuit.measure()
            result.append(circuit.get_classical_state_as_string())
    return result

"""
Function to plot a histogram of all classical states after executing the circuit multiple times.
"""
@staticmethod
def histogram_of_classical_states(ideal_string_array, noisy_string_array=None):
    ideal_histogram = Counter(ideal_string_array)
    ideal_unique_strings = sorted(list(ideal_histogram.keys()))
    ideal_counts = [ideal_histogram[string] for string in ideal_unique_strings]

    if noisy_string_array is None:
        plt.bar(ideal_unique_strings, ideal_counts)
        if len(ideal_histogram) > 8:
            plt.xticks(rotation='vertical')
        plt.xlabel('Classical states')
        plt.ylabel('Nr occurrences')
        plt.title('Number of occurrences of classical states')
        plt.show()
    else:
        width = 0.4  # Width of the bars

        # Combine and sort all unique strings
        noisy_histogram = Counter(noisy_string_array)
        all_unique_strings = sorted(set(ideal_unique_strings + list(noisy_histogram.keys())))
        ideal_counts = [ideal_histogram.get(string, 0) for string in all_unique_strings]
        noisy_counts = [noisy_histogram.get(string, 0) for string in all_unique_strings]

        # Generate x positions for the bars
        x = np.arange(len(all_unique_strings))

        # Plot ideal and noisy bars side by side
        plt.bar(x - width / 2, ideal_counts, width, label='Ideal')
        plt.bar(x + width / 2, noisy_counts, width, label='Noisy', color='#eb4034')

        # Set x-tick labels to the classical state strings
        plt.xticks(x, all_unique_strings, rotation='vertical' if len(all_unique_strings) > 8 else 'horizontal')

        # Add labels and title
        plt.xlabel('Classical states')
        plt.ylabel('Nr occurrences')
        plt.title('Number of occurrences of classical states')
        plt.tight_layout()
        plt.legend(loc='upper right')
        plt.show()


"""
Function to plot a all intermediate (quantum) states of the last execution of a circuit.
"""
@staticmethod
def show_all_intermediate_states(circuit:Circuit, show_description=True, show_colorbar=True):
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
    color_map = mcol.LinearSegmentedColormap.from_list('CmapBlueRed',['b','r'])
    norm = plt.Normalize(vmin=0, vmax=1)

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
        all_states_as_string.append(Dirac.state_as_string(i,circuit.N))
    plt.xticks(positions_x, all_states_as_string, rotation='vertical')

    j = 0.5
    positions_y = [j]
    if show_description:
        all_operations_as_string = ['Initial state  ' + '.'*circuit.N]
    else:
        all_operations_as_string = ['.'*circuit.N]
    j = j + 1
    for description, gate in zip(circuit.descriptions, circuit.gates):
        positions_y.append(j)
        if show_description:
            all_operations_as_string.append(f"{description}  {gate}")
        else:
            all_operations_as_string.append(f"{gate}")
        j = j + 1
    plt.yticks(positions_y, all_operations_as_string)

    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
        sm.set_array([])
        ax = plt.gca()
        divider = ax.get_position()
        shrink = divider.height
        cbar = plt.colorbar(sm, ax=ax, shrink=shrink)
    
    plt.title('Intermediate quantum states')
    plt.show()


"""
Function to plot a all intermediate probabilities of the last execution of a circuit.
"""
@staticmethod
def show_all_probabilities(circuit:Circuit, show_description=True, show_colorbar=True):
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
    color_map = mcol.LinearSegmentedColormap.from_list('CmapBlueRed',['b','r'])
    norm = plt.Normalize(vmin=0, vmax=1)

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
        all_states_as_string.append(Dirac.state_as_string(i, circuit.N))
    plt.xticks(positions_x, all_states_as_string, rotation='vertical')

    positions_y = [0]
    if show_description:
        all_operations_as_string = ['Initial state  ' + '.'*circuit.N]
    else:
        all_operations_as_string = ['.'*circuit.N]
    j = 1
    for description, gate in zip(circuit.descriptions, circuit.gates):
        positions_y.append(j)
        if show_description:
            all_operations_as_string.append(f"{description}  {gate}")
        else:
            all_operations_as_string.append(f"{gate}")
        j = j + 1
    plt.yticks(positions_y, all_operations_as_string)

    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
        sm.set_array([])
        ax = plt.gca()
        divider = ax.get_position()
        shrink = divider.height
        cbar = plt.colorbar(sm, ax=ax, shrink=shrink)
    
    plt.title('Intermediate probabilities')
    plt.show()

"""
Function to plot x, y, and z-values for each qubit during the last execution of a circuit.
If parameter noisy_circuit is defined, the x, y, and z values for that circuit will be shown in red.
"""
@staticmethod
def plot_intermediate_states_per_qubit(ideal_circuit:Circuit, noisy_circuit:Circuit=None):
    for q in range(ideal_circuit.N):
        x_measures_ideal = ideal_circuit.get_x_measures(q)
        y_measures_ideal = ideal_circuit.get_y_measures(q)
        z_measures_ideal = ideal_circuit.get_z_measures(q)
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))
        ax1.plot(x_measures_ideal, 'b', label='x ideal')
        ax1.set_ylim(-1.0, 1.0)
        ax1.set_title(f'X for qubit {q}')
        ax1.set_ylabel('X')
        ax2.plot(y_measures_ideal, 'b', label='y ideal')
        ax2.set_ylim(-1.0, 1.0)
        ax2.set_title(f'Y for qubit {q}')
        ax2.set_ylabel('Y')
        ax3.plot(z_measures_ideal, 'b', label='z ideal')
        ax3.set_ylim(-1.0, 1.0)
        ax3.set_title(f'Z for qubit {q}')
        ax3.set_xlabel('Circuit depth')
        ax3.set_ylabel('Z')

        if not noisy_circuit is None:
            x_measures_noisy = noisy_circuit.get_x_measures(q)
            y_measures_noisy = noisy_circuit.get_y_measures(q)
            z_measures_noisy = noisy_circuit.get_z_measures(q)
            ax1.plot(x_measures_noisy, 'r', label='x noisy')
            ax2.plot(y_measures_noisy, 'r', label='y noisy')
            ax3.plot(z_measures_noisy, 'r', label='z noisy')
            ax1.legend(loc='upper right')
            ax2.legend(loc='upper right')
            ax3.legend(loc='upper right')
