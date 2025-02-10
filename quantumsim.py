
"""
Copyright (c) 2024 Nico Kuijpers

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS åPROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import cmath
import matplotlib.colors as mcol
import matplotlib.animation as animation
import random
import re
from abc import ABC, abstractmethod
from collections import Counter


'''
This code requires QuTiP for rendering Bloch spheres.
See: https://qutip.org/
QuTiP can be installed by
pip install qutip
'''
from qutip import Bloch

'''
Set the default font family to Courier to ensure a monospaced font for labels of axes in 
'''
matplotlib.rcParams['font.family'] = 'Arial'

'''
Symbol for pi
'''
pi_symbol = '\u03c0'


"""
Functions for the Dirac notation to describe (quantum) states and (quantum) operators.
|a> is called 'ket' and represents a column vector with 1 in entry a and 0 everywhere else.
<a| is called 'bra' and represents a row vector with 1 in entry a and 0 everywhere else.
<a||b> is the inner product of <a| and |b>, which is 1 if a = b and 0 if a != b.
|a><b| is the outer product of |a> and <b|, which is a matrix with 1 in entry (a,b) and 0 everywhere else.
Function state_as_string converts integer i, 0 <= i < N, to a quantum state in Dirac notation.
"""
class Dirac:
    
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
    
    @staticmethod
    def state_as_string(i,N) -> str:
        if i < 0 or i >= 2**N:
            raise ValueError("Input i and N must satisfy 0 <= i < 2^N")

        binary_string = bin(i)
        state_as_string = binary_string[2:].zfill(N)
        return "|" + state_as_string + ">"


"""
Functions to obtain 2 x 2 unitary matrices for unitary qubit operations.
"""
class QubitUnitaryOperation:
    
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

    @staticmethod
    def get_u_gate(theta, phi, lam):
        sin = math.sin(theta/2)
        cos = math.cos(theta/2)
        a = cos
        b = -cmath.exp(1j * lam) * sin
        c = cmath.exp(1j * phi) * sin
        d = cmath.exp(1j * (phi + lam)) * cos
        return np.array([[a, b], [c, d]])
"""
Functions to obtain N x N unitary matrices for unitary operations on quantum circuits of N qubits.
"""
class CircuitUnitaryOperation:
    
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
    def get_combined_operation_for_u_gate(theta, phi, lam, q, N):
        u_gate = QubitUnitaryOperation.get_u_gate(theta, phi, lam)
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(u_gate, q, N)
    
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
    def get_combined_operation_for_controlled_rotate_x(theta, control, target, N):
        operation = QubitUnitaryOperation.get_rotate_x(theta)
        return CircuitUnitaryOperation.get_combined_operation_for_controlled_qubit_operation(operation, control, target, N)

    @staticmethod
    def get_combined_operation_for_controlled_rotate_y(theta, control, target, N):
        operation = QubitUnitaryOperation.get_rotate_y(theta)
        return CircuitUnitaryOperation.get_combined_operation_for_controlled_qubit_operation(operation, control, target, N)

    @staticmethod
    def get_combined_operation_for_controlled_rotate_z(theta, control, target, N):
        operation = QubitUnitaryOperation.get_rotate_z(theta)
        return CircuitUnitaryOperation.get_combined_operation_for_controlled_qubit_operation(operation, control, target, N)
    
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
    def get_combined_operation_for_controlled_u_gate(theta, phi, lam, control, target, N):
        u_gate = QubitUnitaryOperation.get_u_gate(theta, phi, lam)
        return CircuitUnitaryOperation.get_combined_operation_for_controlled_qubit_operation(u_gate, control, target, N)

    @staticmethod
    def get_combined_operation_for_swap(a, b, N):
        combined_operation_cnot_a_b = CircuitUnitaryOperation.get_combined_operation_for_cnot(a, b, N)
        combined_operation_cnot_b_a = CircuitUnitaryOperation.get_combined_operation_for_cnot(b, a, N)
        return np.dot(np.dot(combined_operation_cnot_a_b,combined_operation_cnot_b_a),combined_operation_cnot_a_b)
    
    @staticmethod
    def get_combined_operation_for_fredkin(control, a, b, N):
        if control == a or control == b:
            raise ValueError(f'Fredkin operation not supported for control = {control}, a = {a}, and b = {b}')
        if a != 0 and b != 0:
            combined_operation_swap_control_0 = CircuitUnitaryOperation.get_combined_operation_for_swap(control, 0, N)
            combined_operation_swap_a_b = CircuitUnitaryOperation.get_combined_operation_for_swap(a-1, b-1, N-1)
            combined_operation_fredkin = CircuitUnitaryOperation.get_combined_operation_for_controlled_unitary_operation(combined_operation_swap_a_b)
            return np.dot(np.dot(combined_operation_swap_control_0, combined_operation_fredkin), combined_operation_swap_control_0)
        elif a == 0:
            combined_operation_swap_control_a = CircuitUnitaryOperation.get_combined_operation_for_swap(control, a, N)
            combined_operation_fredkin = CircuitUnitaryOperation.get_combined_operation_for_fredkin(a, control, b, N)
            return np.dot(np.dot(combined_operation_swap_control_a, combined_operation_fredkin), combined_operation_swap_control_a)
        else:
            combined_operation_swap_control_b = CircuitUnitaryOperation.get_combined_operation_for_swap(control, b, N)
            combined_operation_fredkin = CircuitUnitaryOperation.get_combined_operation_for_fredkin(b, a, control, N)
            return np.dot(np.dot(combined_operation_swap_control_b, combined_operation_fredkin), combined_operation_swap_control_b)
 
    @staticmethod
    def get_combined_operation_for_toffoli(control_a, control_b, target, N):
        if control_a == control_b or control_a == target or control_b == target:
            raise ValueError(f'Toffoli operation not supported for control_a = {control_a}, control_b = {control_b}, and target = {target}')
        if control_b != 0 and target != 0:
            combined_operation_swap_control_a_0 = CircuitUnitaryOperation.get_combined_operation_for_swap(control_a, 0, N)
            combined_operation_cnot_control_b_target = CircuitUnitaryOperation.get_combined_operation_for_cnot(control_b-1, target-1, N-1)
            combined_operation_toffoli = CircuitUnitaryOperation.get_combined_operation_for_controlled_unitary_operation(combined_operation_cnot_control_b_target)
            return np.dot(np.dot(combined_operation_swap_control_a_0, combined_operation_toffoli), combined_operation_swap_control_a_0)
        elif control_b == 0:
            return CircuitUnitaryOperation.get_combined_operation_for_toffoli(control_b, control_a, target, N)
        else:
            combined_operation_swap_control_a_target = CircuitUnitaryOperation.get_combined_operation_for_swap(control_a, target, N)
            combined_operation_toffoli = CircuitUnitaryOperation.get_combined_operation_for_toffoli(target, control_b, control_a, N)
            return np.dot(np.dot(combined_operation_swap_control_a_target, combined_operation_toffoli), combined_operation_swap_control_a_target)
    
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
                combined_operation_zero = np.kron(combined_operation_zero, ket_bra_00)
                combined_operation_one  = np.kron(combined_operation_one, ket_bra_11)
                i = i + 1
            elif target == i:
                combined_operation_zero = np.kron(combined_operation_zero, identity_operation)
                combined_operation_one  = np.kron(combined_operation_one, operation)
                i = i + math.log(operation.shape[0],2)
            else:
                combined_operation_zero = np.kron(combined_operation_zero, identity)
                combined_operation_one  = np.kron(combined_operation_one, identity)
                i = i + 1
        return combined_operation_zero + combined_operation_one
    
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
    
    @staticmethod
    def get_combined_operation_for_generic_toffoli(controls:list[int], target:int, N:int):
        nr_controls = len(controls)
        if target in controls or nr_controls >= N:
            raise ValueError(f'Generic toffoli gate not supported for controls {controls} and target = {target}')
        if nr_controls not in controls:
            combined_operation_swap = np.eye(2**N)
            controls_sorted = sorted(controls)
            for i in range(nr_controls):
                q = controls_sorted[i]
                combined_operation_swap_q_i = CircuitUnitaryOperation.get_combined_operation_for_swap(q, i, N)
                combined_operation_swap = np.dot(combined_operation_swap, combined_operation_swap_q_i)
            combined_operation_swap_target_nr_controls = CircuitUnitaryOperation.get_combined_operation_for_swap(target, nr_controls, N)
            combined_operation_swap = np.dot(combined_operation_swap, combined_operation_swap_target_nr_controls)
            combined_operation_toffoli = CircuitUnitaryOperation.get_combined_operation_for_multi_controlled_pauli_x_operation(nr_controls + 1)
            combined_operation_toffoli_circuit = CircuitUnitaryOperation.get_combined_operation_for_unitary_operation_general(combined_operation_toffoli, 0, N)
            return np.dot(np.dot(combined_operation_swap, combined_operation_toffoli_circuit), np.conjugate(combined_operation_swap).T)
        else:
            combined_operation_swap_control_target = CircuitUnitaryOperation.get_combined_operation_for_swap(nr_controls, target, N)
            controls_updated = [q for q in controls if q != nr_controls]
            controls_updated.append(target)
            combined_operation_generic_toffoli = CircuitUnitaryOperation.get_combined_operation_for_generic_toffoli(controls_updated, nr_controls, N)
            return np.dot(np.dot(combined_operation_swap_control_target, combined_operation_generic_toffoli), np.conjugate(combined_operation_swap_control_target).T)

"""
Class representing the quantum state of a quantum circuit of N qubits.
"""
class StateVector:
    
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

    def apply_noisy_operation(self, operation):
        # A noisy operation does not have to be a unitary matrix
        self.state_vector = np.dot(operation, self.state_vector)

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

    def measure(self) -> str:
        probalities = np.square(np.abs(self.state_vector)).flatten()
        self.index = np.random.choice(len(probalities), p=probalities)
        return self.get_classical_state_as_string()
    
    def measure_qubit(self, q) -> int:
        identity = QubitUnitaryOperation.get_identity()
        ket_bra_00 = Dirac.ket_bra(2,0,0)
        ket_bra_11 = Dirac.ket_bra(2,1,1)
        P0 = np.eye(1,1)
        P1 = np.eye(1,1)
        for i in range(self.N):
            if i == q:
                P0 = np.kron(P0, ket_bra_00)
                P1 = np.kron(P1, ket_bra_11)
            else:
                P0 = np.kron(P0, identity)
                P1 = np.kron(P1, identity)
        prob_0 = np.vdot(self.state_vector, P0.dot(self.state_vector)).real
        prob_1 = np.vdot(self.state_vector, P1.dot(self.state_vector)).real
        r = np.random.random()
        if r <= prob_0:
            self.state_vector = np.dot(P0,self.state_vector)/np.sqrt(prob_0)
            return 0
        else:
            self.state_vector = np.dot(P1,self.state_vector)/np.sqrt(prob_1)
            return 1
        
    def reset_qubit(self, q):
        measured_value = self.measure_qubit(q)
        if measured_value == 1:
            combined_operation_pauli_x = CircuitUnitaryOperation.get_combined_operation_for_pauli_x(q, self.N)
            self.apply_unitary_operation(combined_operation_pauli_x)

    def noisy_measure(self):
        # For a noisy circuit, the sum of probabilities may not be equal to one
        probalities = np.square(np.abs(self.state_vector)).flatten()
        probalities = probalities / np.sum(probalities)
        self.index = np.random.choice(len(probalities), p=probalities)

    def get_quantum_state(self):
        return self.state_vector

    def get_classical_state_as_string(self):
        return Dirac.state_as_string(self.index, self.N)
    
    def print(self):
        for i, val in enumerate(self.state_vector):
            print(f"{Dirac.state_as_string(i,self.N)} : {val[0]}")

class RegisterPartition:
    """
    This object is used splice up the classical bit register, 
    its purpose is pretifying ClassicalBitRegisters output when working with many classical bits
    """
    def __init__(self, begin: int, end: int, name: str):
        self.begin = begin
        self.end = end
        self.name = name

    def toString(self) -> str:
        return "Partition: " + self.name + ", starting from: " + str(self.begin) + " to: " + str(self.end) + ""

class ClassicalBitRegister:

    def __init__(self, numberClassicBits: int):
        self.numberClassicBits = numberClassicBits
        self.partitions = []
        self.register = []
        for i in range(numberClassicBits):
            self.register.insert(-1, 0)

    def create_partition(self, begin: int, end: int, name: str):
        if(begin > end):
            raise Exception("Begin must be smaller than end")
        if(end > self.numberClassicBits):
            raise Exception("Can not partition beyond the register limits")
        for existingPartition in self.partitions:
            # Check if the new partition is within boundaries of a different one
            if(existingPartition.begin >= begin and begin <= existingPartition.end):
                raise Exception("Begin parameter is within boundaries of a different partition")
            if(existingPartition.begin >= end and end <= existingPartition.end):
                raise Exception("End parameter is within boundaries of a different partition")
            # existing 0 3 new is 4 7
            if(begin < existingPartition.begin and end > existingPartition.end):
                raise Exception("Begin and End parameters are overlapping with a different partition")
        self.partitions.append(RegisterPartition(begin, end, name))

    def write(self, index: int, value: int):
        if(index < 0 or index > self.numberClassicBits):
           raise Exception("Index out of bounds") 
        if(value != 0 and value != 1):
            raise Exception("Value must be either 0 or 1")
        self.register.pop(index)
        self.register.insert(index, value)

    def read(self, index: int) -> int:
        return self.register[index]

    def clear(self):
        for i in range(self.numberClassicBits):
            self.register[i] = 0

    def getAmountOfBits(self):
        return self.numberClassicBits

    def toString(self, beginBit: int=0, endBit: int=0):
        
        output = ""
        if beginBit == 0 and endBit == 0:
            for i in range(self.numberClassicBits):
                # Check first if the index is a beginning of a partition
                for partition in self.partitions:
                    if(partition.begin == i):
                        output = output + " " + partition.name + ":"
                        break
                output = output + str(self.register[i])
            return output
        else:
            for i in range(beginBit, endBit, 1):
                output = output + str(self.register[i])
            return output


    def print(self):
        print(self.toString())

"""
Class representing a quantum circuit of N qubits.
"""
class Circuit:
    
    def __init__(self, qubits: int, bits: int=0,  save_instructions: bool=False, noise_factor: float = 1):
        self.N = qubits
        self.classicalBitRegister = ClassicalBitRegister(bits)
        self.noise_factor = noise_factor

        self.state_vector = StateVector(self.N)
        self.quantum_states = [self.state_vector.get_quantum_state()]
        self.descriptions = []
        self.operations = []
        self.gates = []

        # Options / Flags
        self.save_instructions = save_instructions
        self.instructions = []

        # Keeps track of logical errors, only usable when running surface codes with recovery gates
        self.logical_error_count = 0

        self.state_vector = StateVector(self.N)
        self.noisy_operations_state_prep = []
        self.noisy_operations_incoherent = []
        self.noisy_operations_readout = []
        self.x_measures = np.empty(self.N, dtype=object)
        self.y_measures = np.empty(self.N, dtype=object)
        self.z_measures = np.empty(self.N, dtype=object)

        # Noisy gates
        self.phi = [0 for _ in range(self.N)] # Keep a list of phi values for every qubit

        # Load in the device parameters json
        device_params = DeviceParameters()
        
        device_params.load_from_json("./assets/noise_parameters/Virtual_Quantum_Computer.json")
        qiskit_kyiv_parameter_dict = device_params.__dict__()

        # Define a list of parameter values based on the stored device parameters
        self.parameters = {
            "T1": [float(qiskit_kyiv_parameter_dict["T1"][i % len(qiskit_kyiv_parameter_dict["T1"])]) for i in range(self.N)], # Loop over the T1 values of the device parameters to assign to each qubit
            "T2": [float(qiskit_kyiv_parameter_dict["T2"][i % len(qiskit_kyiv_parameter_dict["T2"])]) for i in range(self.N)], # Loop over the T2 values of the device parameters to assign to each qubit
            "p": [float(qiskit_kyiv_parameter_dict["p"][i % len(qiskit_kyiv_parameter_dict["p"])]) for i in range(self.N)], # Loop over the p values of the device parameters to assign to each qubit
        }

    def identity(self, q):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_identity(self.N)
        self.descriptions.append(f"Identity on qubit {q}")
        gate_as_string = '.'*self.N
        self.gates.append(gate_as_string)
        self.instructions.append(Identity(self.N, q))  if self.save_instructions else self.operations.append(combined_operation)

    def pauli_x(self, q):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_pauli_x(q, self.N)
        self.descriptions.append(f"Pauli X on qubit {q}")
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[q] = 'X'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)
        self.instructions.append(Pauli_X(self.N, q))  if self.save_instructions else self.operations.append(combined_operation)

    def noisy_pauli_x(self, q: int, p: float = None, T1: float = None, T2: float = None):
        """Adds a noisy Pauli X gate to the circuit

        Args:
            q (int): Qubit to operate on.
            p (float): Single-qubit depolarizing error probability.
            T1 (float): Qubit's amplitude damping time in ns.
            T2 (float): Qubit's dephasing time in ns.
        """
        # If any noise parameter is None use the generated value
        if p is None:
            p = self.parameters["p"][q] * self.noise_factor
        if T1 is None:
            T1 = self.parameters["T1"][q] / self.noise_factor
        if T2 is None:
            T2 = self.parameters["T2"][q] / self.noise_factor

        self.instructions.append(NoisyPauliX(q, self.N, p, T1, T2))
        self.descriptions.append(f"Noisy Pauli X on qubit {q}")
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[q] = 'X'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)

    def pauli_y(self, q):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_pauli_y(q, self.N)
        self.descriptions.append(f"Pauli Y on qubit {q}")
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[q] = 'Y'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)
        self.instructions.append(Pauli_Y(self.N, q))  if self.save_instructions else self.operations.append(combined_operation)

    def noisy_pauli_y(self, q: int, p: float= None, T1: float= None, T2: float= None):
        """Adds a noisy Pauli Y gate to the circuit

        Args:
            q (int): Qubit to operate on.
            p (float): Single-qubit depolarizing error probability.
            T1 (float): Qubit's amplitude damping time in ns.
            T2 (float): Qubit's dephasing time in ns.
        """

        # If any noise parameter is None use the generated value
        if p is None:
            p = self.parameters["p"][q] * self.noise_factor
        if T1 is None:
            T1 = self.parameters["T1"][q] / self.noise_factor
        if T2 is None:
            T2 = self.parameters["T2"][q] / self.noise_factor

        self.instructions.append(NoisyPauliY(q, self.N, p, T1, T2))
        self.descriptions.append(f"Noisy Pauli Y on qubit {q}")
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[q] = 'Y'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)


    def pauli_z(self, q):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_pauli_z(q, self.N)
        self.descriptions.append(f"Pauli Z on qubit {q}")
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[q] = 'Z'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)
        self.instructions.append(Pauli_Z(self.N, q))  if self.save_instructions else self.operations.append(combined_operation)

    # Define the new "virtual" Pauli Z gate
    def noisy_pauli_z(self, q: int, p: float = None, T1: float = None, T2: float = None):
        """Adds a noisy Pauli Z gate to the circuit

        Args:
            q (int): Qubit to operate on.
            p (float): Single-qubit depolarizing error probability.
            T1 (float): Qubit's amplitude damping time in ns.
            T2 (float): Qubit's dephasing time in ns.
        """

        # If any noise parameter is None use the generated value
        if p is None:
            p = self.parameters["p"][q] * self.noise_factor
        if T1 is None:
            T1 = self.parameters["T1"][q] / self.noise_factor
        if T2 is None:
            T2 = self.parameters["T2"][q] / self.noise_factor

        self.instructions.append(NoisyPauliZ(q, self.N, p, T1, T2))
        self.descriptions.append(f"Noisy Pauli Z on qubit {q}")
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[q] = 'Z'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)

    def hadamard(self, q):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_hadamard(q, self.N)
        self.descriptions.append(f"Hadamard on qubit {q}")
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[q] = 'H'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)
        self.instructions.append(Hadamard(self.N, q))  if self.save_instructions else self.operations.append(combined_operation)

    def noisy_hadamard(self, q: int, p: float= None, T1: float= None, T2: float= None):
        """Adds a noisy hadamard gate to the circuit

        Args:
            q (int): Qubit to operate on.
            p (float): Single-qubit depolarizing error probability.
            T1 (float): Qubit's amplitude damping time in ns.
            T2 (float): Qubit's dephasing time in ns.
        """

        # If any noise parameter is None use the generated value
        if p is None:
            p = self.parameters["p"][q] * self.noise_factor
        if T1 is None:
            T1 = self.parameters["T1"][q] / self.noise_factor
        if T2 is None:
            T2 = self.parameters["T2"][q] / self.noise_factor

        self.instructions.append(NoisyHadamard(q, self.N, p, T1, T2))
        self.descriptions.append(f"Noisy Hadamard on qubit {q}")
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[q] = 'H'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)

    def phase(self, theta, q):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_phase(theta, q, self.N)
        self.descriptions.append(f"Phase with theta = {theta/np.pi:.3f} {pi_symbol} on qubit {q}")
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[q] = 'S'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)
        self.instructions.append(Phase(self.N, q, theta))  if self.save_instructions else self.operations.append(combined_operation)


    def noisy_phase(self, theta: float, q: int, p: float = None, T1: float = None, T2: float = None):
        """This gate is implemented making use of perfect hadamard in combination with a X gate!

        Args:
            theta (float): Angle of rotation on the Bloch sphere.
            q (int): Qubit to operate on.
            p (float): Single-qubit depolarizing error probability.
            T1 (float): Qubit's amplitude damping time in ns.
            T2 (float): Qubit's dephasing time in ns.
        """

         # If any noise parameter is None use the generated value
        if p is None:
            p = self.parameters["p"][q]
        if T1 is None:
            T1 = self.parameters["T1"][q]
        if T2 is None:
            T2 = self.parameters["T2"][q]
 
        self.instructions.append(NoisyPhase(theta, q, self.N, p, T1, T2))
        self.descriptions.append(f"Noisy X rotation of {theta} on qubit {q}")
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[q] = 'X'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)

    def rotate_x(self, theta, q):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_rotate_x(theta, q, self.N)
        self.descriptions.append(f"Rotate X with theta = {theta/np.pi:.3f} {pi_symbol} on qubit {q}")
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[q] = 'R'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)
        self.instructions.append(Rotate_X(self.N, q, theta))  if self.save_instructions else self.operations.append(combined_operation)

    
    def rotate_y(self, theta, q):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_rotate_y(theta, q, self.N)
        self.descriptions.append(f"Rotate Y with theta = {theta/np.pi:.3f} {pi_symbol} on qubit {q}")
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[q] = 'R'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)
        self.instructions.append(Rotate_Y(self.N, q, theta)) if self.save_instructions else self.operations.append(combined_operation)

    
    def rotate_z(self, theta, q):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_rotate_z(theta, q, self.N)
        self.descriptions.append(f"Rotate Z with theta = {theta/np.pi:.3f} {pi_symbol} on qubit {q}")
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[q] = 'R'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)
        self.instructions.append(Rotate_Z(self.N, q, theta)) if self.save_instructions else self.operations.append(combined_operation)


    def u_gate(self, theta, phi, lam, q):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_u_gate(theta, phi, lam, q, self.N)
        self.descriptions.append(f"U-gate with (theta,phi,lam) = ({theta/np.pi:.3f} {pi_symbol}, {phi/np.pi:.3f} {pi_symbol}, {lam/np.pi:.3f} {pi_symbol}) on qubit {q}")
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[q] = 'U'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)
        self.instructions.append(U_Gate(self.N, q, theta, phi, lam)) if self.save_instructions else self.operations.append(combined_operation)

    def cnot(self, control, target):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_cnot(control, target, self.N)
        self.descriptions.append(f"CNOT with control qubit {control} and target qubit {target}")
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[control] = '*'
        gate_as_list[target] = 'X'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)
        self.instructions.append(CNOT(self.N, target, control))  if self.save_instructions else self.operations.append(combined_operation)

    # Define the new cnot gate with integrated noise 
    def noisy_cnot(self, c_qubit: int, t_qubit: int, c_p: float= None, t_p: float= None, gate_error: float=None, c_T1: float= None, t_T1: float= None, c_T2: float= None, t_T2: float= None):
        """Adds a noisy cnot gate to the circuit with depolarizing and
        relaxation errors on both qubits during the unitary evolution.

        Args:
            c_qubit (int): Control qubit for the gate.
            t_qubit (int): Target qubit for the gate.
            c_p (float): Depolarizing error probability for the control qubit.
            t_p (float): Depolarizing error probability for the target qubit.
            c_T1 (float): Amplitude damping time in ns for the control qubit.
            t_T1 (float): Amplitude damping time in ns for the target qubit.
            c_T2 (float): Dephasing time in ns for the control qubit.
            t_T2 (float): Dephasing time in ns for the target qubit.
            gate_error (float): CNOT depolarizing error probability.
        """

        # If any noise parameter is None use the generated value
        if c_p is None:
            c_p = self.parameters["p"][c_qubit] * self.noise_factor
        if c_T1 is None:
            c_T1 = self.parameters["T1"][c_qubit] / self.noise_factor
        if c_T2 is None:
            c_T2 = self.parameters["T2"][c_qubit] / self.noise_factor
        if t_p is None:
            t_p = self.parameters["p"][t_qubit] * self.noise_factor
        if t_T1 is None:
            t_T1 = self.parameters["T1"][t_qubit] / self.noise_factor
        if t_T2 is None:
            t_T2 = self.parameters["T2"][t_qubit] / self.noise_factor
        if gate_error is None:
            gate_error = 0.015 # Used by Tycho's implementation

        self.instructions.append(NoisyCNOT(c_qubit, t_qubit, self.N, c_p, t_p, c_T1, t_T1, c_T2, t_T2, gate_error))

        self.descriptions.append(f"Noisy CNOT with target qubit {t_qubit} and control qubit {c_qubit}")
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[c_qubit] = '*'
        gate_as_list[t_qubit] = 'x'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string) 

    def controlled_pauli_y(self, control, target):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_controlled_pauli_y(control, target, self.N)
        self.descriptions.append(f"Controlled Pauli Y with control qubit {control} and target qubit {target}")
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[control] = '*'
        gate_as_list[target] = 'Y'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)
        self.instructions.append(Controlled_Pauli_Y(self.N, target, control))   if self.save_instructions else self.operations.append(combined_operation)

    def controlled_pauli_z(self, control, target):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_controlled_pauli_z(control, target, self.N)
        self.descriptions.append(f"Controlled Pauli Z with control qubit {control} and target qubit {target}")
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[control] = '*'
        gate_as_list[target] = 'Z'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)
        self.instructions.append(Controlled_Pauli_Z(self.N, target, control))   if self.save_instructions else self.operations.append(combined_operation)
    
    def controlled_hadamard(self, control, target):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_controlled_hadamard(control, target, self.N)
        self.descriptions.append(f"Controlled Hadamard with control qubit {control} and target qubit {target}")
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[control] = '*'
        gate_as_list[target] = 'H'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)
        self.instructions.append(Controlled_Hadamard(self.N, target, control))  if self.save_instructions else self.operations.append(combined_operation)

    def controlled_phase(self, theta, control, target):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_controlled_phase(theta, control, target, self.N)
        self.descriptions.append(f"Controlled phase with theta = {theta/np.pi:.3f} {pi_symbol}, control qubit {control}, and target qubit {target}")
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[control] = '*'
        gate_as_list[target] = 'S'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)
        self.instructions.append(Controlled_Phase(theta, self.N, target, control))  if self.save_instructions else self.operations.append(combined_operation)

    def controlled_rotate_x(self, theta, control, target):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_controlled_rotate_x(theta, control, target, self.N)
        self.descriptions.append(f"Controlled rotate X with theta = {theta/np.pi:.3f} {pi_symbol}, control qubit {control}, and target qubit {target}")
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[control] = '*'
        gate_as_list[target] = 'R'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)
        self.instructions.append(Controlled_Rotate_X(theta, self.N, target, control))  if self.save_instructions else self.operations.append(combined_operation)


    def controlled_rotate_y(self, theta, control, target):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_controlled_rotate_y(theta, control, target, self.N)
        self.descriptions.append(f"Controlled rotate Y with theta = {theta/np.pi:.3f} {pi_symbol}, control qubit {control}, and target qubit {target}")
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[control] = '*'
        gate_as_list[target] = 'R'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)
        self.instructions.append(Controlled_Rotate_Y(theta, self.N, target, control))   if self.save_instructions else self.operations.append(combined_operation)


    def controlled_rotate_z(self, theta, control, target):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_controlled_rotate_z(theta, control, target, self.N)
        self.descriptions.append(f"Controlled rotate Z with theta = {theta/np.pi:.3f} {pi_symbol}, control qubit {control}, and target qubit {target}")
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[control] = '*'
        gate_as_list[target] = 'R'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)
        self.instructions.append(Controlled_Rotate_Z(theta, self.N, target, control))  if self.save_instructions else self.operations.append(combined_operation)

    def controlled_u_gate(self, theta, phi, lam, control, target):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_controlled_u_gate(theta, phi, lam, control, target, self.N)
        self.descriptions.append(f"Controlled U-gate (theta,phi,lam) = ({theta/np.pi:.3f} {pi_symbol}, {phi/np.pi:.3f} {pi_symbol}, {lam/np.pi:.3f} {pi_symbol}), control {control}, and target {target}")
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[control] = '*'
        gate_as_list[target] = 'U'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)
        self.instructions.append(Controlled_U_Gate(theta, phi, lam, self.N, target, control))  if self.save_instructions else self.operations.append(combined_operation)

    def controlled_unitary_operation(self, operation, control, target):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_controlled_unitary_operation_general(operation, control, target, self.N)
        self.descriptions.append(f"Controlled unitary operation with control qubit {control} and target qubit {target}")
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[control] = '*'
        gate_as_list[target] = 'U'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)
        self.instructions.append(Controlled_Unitary_Operation(self.N, operation, target, control))  if self.save_instructions else self.operations.append(combined_operation)

    def swap(self, a, b):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_swap(a, b, self.N)
        self.descriptions.append(f"SWAP on qubit {a} and qubit {b}")
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[a] = 'x'
        gate_as_list[b] = 'x'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)
        self.instructions.append(Swap(self.N, a, b)) if self.save_instructions else self.operations.append(combined_operation)

    def fredkin(self, control, a, b):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_fredkin(control, a, b, self.N)
        self.descriptions.append(f"Fredkin with control qubit {control} and SWAP on qubit {a} and qubit {b}")
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[control] = '*'
        gate_as_list[a] = 'x'
        gate_as_list[b] = 'x'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)
        self.instructions.append(Fredkin(self.N, control, a, b))  if self.save_instructions else self.operations.append(combined_operation)
    
    def toffoli(self, control_a, control_b, target):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_toffoli(control_a, control_b, target, self.N)
        self.descriptions.append(f"Toffoli with control qubit {control_a} and CNOT with control qubit {control_b} and target qubit {target}")
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[control_a] = '*'
        gate_as_list[control_b] = '*'
        gate_as_list[target] = 'X'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)
        self.instructions.append(Toffoli(self.N, control_a, control_b, target))  if self.save_instructions else self.operations.append(combined_operation)
    
    def multi_controlled_pauli_z(self):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_multi_controlled_pauli_z_operation(self.N)
        self.descriptions.append(f"Multi-controlled Pauli_Z")
        gate_as_string = '*'*self.N
        self.gates.append(gate_as_string)
        self.instructions.append(Multi_Controlled_Pauli_Z(self.N))  if self.save_instructions else self.operations.append(combined_operation)

    def multi_controlled_pauli_x(self):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_multi_controlled_pauli_x_operation(self.N)
        self.descriptions.append(f"Multi-controlled Pauli_X")
        gate_as_string = '*'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[self.N-1] = 'X'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string) 
        self.instructions.append(Multi_Controlled_Pauli_X(self.N))   if self.save_instructions else self.operations.append(combined_operation)

    def generic_toffoli(self, controls:list[int], target:int):
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_generic_toffoli(controls, target, self.N)
        self.descriptions.append(f"Generic Toffoli with controls {controls} and target {target}")
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        for c in controls:
            gate_as_list[c] = '*'
        gate_as_list[target] = 'X'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)
        self.instructions.append(Generic_Toffoli(self.N, controls, target))   if self.save_instructions else self.operations.append(combined_operation)

    """
    Measurement of a single qubit
    """
    def measurement(self, measurementQubit: int, copyClassicBit: int) -> int:
        self.descriptions.append(f"Measurement on Qubit {measurementQubit} projected on bit: {copyClassicBit}")
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[measurementQubit] = 'M'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)
        self.instructions.append(Measurement(measurementQubit, copyClassicBit)) if self.save_instructions else self.operations.append(np.eye(2**self.N))

    def reset(self, targetQubit: int, readBit: int) -> int:
        self.descriptions.append(f"Reset on Qubit {targetQubit} read from bit: {readBit}")
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[targetQubit] = 'R'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)
        self.instructions.append(Reset(targetQubit, readBit)) if self.save_instructions else self.operations.append(np.eye(2**self.N))

    def noisy_reset(self, q: int, readBit: int, p: float = None, T1: float = None, T2: float = None):
        """Adds a noisy reset gate to the circuit

        Args:
            q (int): Qubit to operate on.
            p (float): Single-qubit depolarizing error probability.
            T1 (float): Qubit's amplitude damping time in ns.
            T2 (float): Qubit's dephasing time in ns.
        """
        # If any noise parameter is None use the generated value
        if p is None:
            p = self.parameters["p"][q]
        if T1 is None:
            T1 = self.parameters["T1"][q]
        if T2 is None:
            T2 = self.parameters["T2"][q]

        self.instructions.append(NoisyReset(q, readBit, self.N, p, T1, T2))
        self.descriptions.append(f"Noisy reset on Qubit {q} read from bit: {readBit}")
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[q] = 'R'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)

    def recovery_phase_flip(self, startIndexBit: int):
        """
        Special function used for doing recoveries when using surface codes
        """
        self.descriptions.append(f"Phase flip recovery, syndrome extracted from the first 4 bits starting from {startIndexBit}")
        self.instructions.append(Recovery_Phase_Flip(startIndexBit, self.N))

        gate_as_string = 'P'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[0] = 'P'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)

    def recovery_bit_flip(self, startIndexBit: int):
        """
        Special function used for doing recoveries when using surface codes
        """
        self.descriptions.append(f"Bit flip recovery, syndrome extracted from the first 4 bits starting from {startIndexBit}")
        self.instructions.append(Recovery_Bit_Flip(startIndexBit, self.N))

        gate_as_string = 'B'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[0] = 'B'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)
    """
    Swap the registers such that the most significant qubit becomes the least significant qubit and vice versa.
    """
    def swap_registers(self):
        for q in range(self.N//2):
            self.swap(q, self.N-q-1)

    """
    Create a controlled version of this circuit.
    """
    def create_controlled_circuit(self, control, target, nr_qubits):
        # control is control qubit
        # target is qubit in new circuit corresponding to first qubit of current circuit
        # nr_qubits is number of qubits in new circuit
        controlled_circuit = Circuit(nr_qubits)
        for operation, description, gate in zip(self.operations, self.descriptions, self.gates):
            combined_operation = CircuitUnitaryOperation.get_combined_operation_for_controlled_unitary_operation_general(operation, control, target, nr_qubits)
            controlled_circuit.operations.append(combined_operation)
            controlled_circuit.descriptions.append(f"Controlled unitary operation {description}")
            gate_as_string = '.'*controlled_circuit.N
            gate_as_list = list(gate_as_string)
            gate_as_list[control] = '*'
            gate_as_list[target:target + len(gate)] = list(gate)
            gate_as_string = ''.join(gate_as_list)
            controlled_circuit.gates.append(gate_as_string)
        return controlled_circuit

    def create_inverse_circuit(self):
        inverse_circuit = Circuit(self.N)
        for operation, description, gate in zip(reversed(self.operations), reversed(self.descriptions), reversed(self.gates)):
            inverse_circuit.operations.append(operation)
            inverse_circuit.descriptions.append(description)
            inverse_circuit.gates.append(gate)
        return inverse_circuit
    
    def append_circuit(self, circuit):
        if circuit.N != self.N:
            raise ValueError("Function append_circuit: circuit to be appended must have same number of qubits")
        if(self.save_instructions == True and circuit.save_instructions == True):
            # Does not work with operations, rather with instructions, copy all instructions
            for instruction, description, gate in zip(circuit.instructions, circuit.descriptions, circuit.gates):
                self.instructions.append(instruction)
                self.descriptions.append(description)
                self.gates.append(gate)
        elif(self.save_instructions == False and circuit.save_instructions == False):
            # Does not work with instructions, rather with operations, copy all instructions
            for operation, description, gate in zip(circuit.operations, circuit.descriptions, circuit.gates):
                self.operations.append(operation)
                self.descriptions.append(description)
                self.gates.append(gate)
        else:
            raise Exception("Save instruction flag must be equal for both circuits")

    """
    Removes gates on indexes mentioned between the startGateIndex and endGateIndex. 
    For example: start = 0, end = 1, removes first gate (index [0])
    start = 1, end = 2, removes second gate (index [1])
    """    
    def remove_circuit_part(self, startGateIndex: int, endGateIndex: int):
        if(startGateIndex < 0):
            raise Exception("startGateIndex out of bounds")
        if(endGateIndex > len(self.gates)):
            raise Exception("endGateIndex out of bounds")
        if(startGateIndex >= endGateIndex):
            raise Exception("Start gate index is greater or equal than end gate index")
        
        numberOfGatesToRemove = endGateIndex - startGateIndex
        for i in range(numberOfGatesToRemove):
            self.remove_circuit_gate(startGateIndex)

    def remove_circuit_gate(self, gateIndex: int):
        if(gateIndex < 0):
            raise Exception("gateIndex out of bounds, smaller than zero")
        if(gateIndex > len(self.gates)):
            raise Exception("gateIndex out of bounds, greater than total amount of gates")
        self.descriptions.pop(gateIndex)
        self.gates.pop(gateIndex)
        if(self.save_instructions):
            self.instructions.pop(gateIndex)
        else:
            self.operations.pop(gateIndex)

    def append_circuit_general(self, circuit, start):
        if circuit.N > self.N:
            raise ValueError("Function append_circuit_general: circuit to be appended must have less or same number of qubits")
        for operation, description, gate in zip(circuit.operations, circuit.descriptions, circuit.gates):
            combined_operation = CircuitUnitaryOperation.get_combined_operation_for_unitary_operation_general(operation, start, self.N)
            self.operations.append(combined_operation)
            self.descriptions.append(f"Append operation {description}")
            gate_as_string = '.'*self.N
            gate_as_list = list(gate_as_string)
            gate_as_list[start:start + len(gate)] = list(gate)
            gate_as_string = ''.join(gate_as_list)
            self.gates.append(gate_as_string)

    def create_noisy_circuit(self):
        noisy_circuit = NoisyCircuit(self.N)
        for operation, description, gate in zip(self.operations, self.descriptions, self.gates):
            noisy_circuit.operations.append(operation)
            noisy_circuit.descriptions.append(description)
            noisy_circuit.gates.append(gate)
        return noisy_circuit
    
    # Define a virtual Rz gate to mimic the "quantum-gates" package
    def virtual_rotate_z(self, q: int, theta: float):
        """ This gate is implemented virtualy and thus is not executed on the actual qubit!
        Update the phase to implement virtual Rz(theta) gate on qubit i

        Args:
            q: index of the qubit
            theta: angle of rotation on the Bloch sphere

        Returns:
             None
        """
        self.phi[q] = self.phi[q] + theta
    
    def bitflip_error(self, q):
        self.pauli_x(q)
        self.descriptions.pop()
        self.descriptions.append(f"Bit-flip error (Pauli X) on qubit {q}")

    def bitflip_error_random(self, start=0, end=0):
        """ 
        Adds a bitflip error (Pauili X) gate to the circuit effecting one random qubit
        """
        if(start == 0 and end == 0):
            self.bitflip_error(random.randint(0, (self.N-1)))
        elif(start > end):
            raise ValueError("start qubit can't be greater than end qubit")
        elif(end > self.N):
            raise ValueError("End qubit can't be greater than the amount of qubits")
        elif(start < 0):
            raise ValueError("Start qubit can't be negative")
        else:
            self.bitflip_error(random.randint(start, end))

    def phaseflip_error(self, q):
        self.pauli_z(q)
        self.descriptions.pop()
        self.descriptions.append(f"Phase-flip error (Pauli Z) on qubit {q}")

    def phaseflip_error_random(self, start=0, end=0):
        """
        Adds a phaseflip error (Pauli Z) gate to the circuit effecting one random qubit
        """
        if(start == 0 and end == 0):
            self.phaseflip_error(random.randint(0, (self.N-1)))
        elif(start > end):
            raise ValueError("start qubit can't be greater than end qubit")
        elif(end > self.N):
            raise ValueError("End qubit can't be greater than the amount of qubits")
        elif(start < 0):
            raise ValueError("Start qubit can't be negative")
        else:
            self.phaseflip_error(random.randint(start, end))
    
    def print_circuit(self):
        for description in self.descriptions:
            print(description)

    def print_gates(self):
        for gate in self.gates:
            print(gate)

    def print_gates_and_descriptions(self):
        if(len(self.descriptions) != len(self.gates)):
            raise Exception("Number of gates is not equal to the number of descriptions")
        for gate, description in zip(self.gates, self.descriptions):
            print(gate + "\t" + description)

    def toString_gates_and_descriptions(self):
        returnString = ""
        if(len(self.descriptions) != len(self.gates)):
            raise Exception("Number of gates is not equal to the number of descriptions")
        for gate, description in zip(self.gates, self.descriptions):
            returnString = returnString.__add__(str(gate + "\t" + description + "\n"))

        return returnString  
    
    def __noisy_instruction_handler(self, instruction):
        if isinstance(instruction, NoisyPauliX):
            # Theta and phi to construct Pauli X
            instruction.setTheta(np.pi)
            instruction.setPhi(-self.phi[instruction.q])
            self.state_vector.apply_noisy_operation(instruction.getNoisyOperation())
        elif isinstance(instruction, NoisyPauliY):
            # First execute a virtual Rz gate
            self.virtual_rotate_z(instruction.q, np.pi)
            instruction.setTheta(np.pi)
            instruction.setPhi(-self.phi[instruction.q])
            self.state_vector.apply_noisy_operation(instruction.getNoisyOperation())
        elif isinstance(instruction, NoisyPauliZ):
            # self.virtual_rotate_z(instruction.q, np.pi)
            instruction.setTheta(np.pi)
            instruction.setPhi(-self.phi[instruction.q])
            self.state_vector.apply_unitary_operation(CircuitUnitaryOperation.get_combined_operation_for_hadamard(instruction.q, instruction.N))
            self.state_vector.apply_noisy_operation(instruction.getNoisyOperation())
            self.state_vector.apply_unitary_operation(CircuitUnitaryOperation.get_combined_operation_for_hadamard(instruction.q, instruction.N))
        elif isinstance(instruction, NoisyPhase):
            instruction.setPhi(-self.phi[instruction.q])
            self.state_vector.apply_unitary_operation(CircuitUnitaryOperation.get_combined_operation_for_hadamard(instruction.q, instruction.N))
            self.state_vector.apply_noisy_operation(instruction.getNoisyOperation())
            self.state_vector.apply_unitary_operation(CircuitUnitaryOperation.get_combined_operation_for_hadamard(instruction.q, instruction.N))
        elif isinstance(instruction, NoisyHadamard):
            # First execute a virtual Rz gate
            self.virtual_rotate_z(instruction.q, np.pi / 2)

            instruction.setTheta(np.pi / 2)
            instruction.setPhi(-self.phi[instruction.q])

            self.state_vector.apply_noisy_operation(instruction.getNoisyOperation())

            # To complete the gate end with a virtual Rz gate
            self.virtual_rotate_z(instruction.q, np.pi / 2)
        elif isinstance(instruction, NoisyCNOT):
            instruction.setPhiControl(self.phi[instruction.c_qubit])
            instruction.setPhiTarget(self.phi[instruction.t_qubit])
            instruction.setTheta(np.pi)
            instruction
            self.state_vector.apply_noisy_operation(instruction.getNoisyOperation())
        elif(isinstance(instruction, NoisyReset)):
            if(self.classicalBitRegister.read(instruction.readBit == 1)):
                instruction.setTheta(np.pi)
                instruction.setPhi(-self.phi[instruction.q])
                self.state_vector.apply_noisy_operation(instruction.getNoisyOperation())
    def __direct_execute__(self, operation: CircuitUnitaryOperation):
        self.state_vector.apply_unitary_operation(operation)
        self.quantum_states.append(self.state_vector.get_quantum_state())
    
    def __measure_execute__(self, measureQubit: int, dataBit: int) -> int:
        # Collapse the state of the qubit to either |0> or |1>
        self.state_vector.measure_qubit(measureQubit)
        
        copyStateVector = self.state_vector
        measurement = copyStateVector.measure()
        # Find the qubit which value should be projected in the bit register
        # Char: '|' should be ignored, therefore '+1'
        qubitValue = int(measurement[measureQubit + 1])
        self.classicalBitRegister.write(dataBit, qubitValue)
        self.quantum_states.append(self.state_vector.get_quantum_state())
    
        return qubitValue
    
    def __reset_execute__(self, targetQubit: int, readBit: int):
        if(self.classicalBitRegister.read(readBit) == 1):
            self.state_vector.apply_unitary_operation(CircuitUnitaryOperation.get_combined_operation_for_pauli_x(targetQubit, self.N))
        self.quantum_states.append(self.state_vector.get_quantum_state())

    def __noisy_reset_execute__(self, targetQubit: int, readBit: int):
        if(self.classicalBitRegister.read(readBit) == 1):
            self.state_vector.apply_unitary_operation(CircuitUnitaryOperation.get_combined_operation_for_pauli_x(targetQubit, self.N))

                
    def execute(self, print_state=False, create_new_state_vector=True):
        if create_new_state_vector:
            self.state_vector = StateVector(self.N)
        self.quantum_states = [self.state_vector.get_quantum_state()]
        if print_state:
            print("Initial quantum state")
            self.state_vector.print()
        
        if(self.save_instructions):
            for instruction in self.instructions:
                if(isinstance(instruction, Measurement)):
                    self.__measure_execute__(instruction.measureQubit, instruction.dataBit)
                elif(isinstance(instruction, Reset)):
                    self.__reset_execute__(instruction.targetQubit, instruction.readBit)
                elif(isinstance(instruction, Recovery_Bit_Flip)):
                    targetQubit = instruction.getTargetQubit(self.classicalBitRegister)
                    if(targetQubit == -1):
                        # No bit flips found, no recovery applied
                        pass
                    elif(targetQubit == -2):
                        # Encountered logical error, unknown syndrome. No recovery applied
                        self.logical_error_count = self.logical_error_count + 1
                    else:
                        self.state_vector.apply_unitary_operation(CircuitUnitaryOperation.get_combined_operation_for_pauli_x(targetQubit, self.N))
                elif(isinstance(instruction, Recovery_Phase_Flip)):
                    targetQubit = instruction.getTargetQubit(self.classicalBitRegister)
                    if(targetQubit == -1):
                        # No phase flips found, no recovery applied
                        pass
                    elif(targetQubit == -2):
                        # Encountered logical error, unknown syndrome. No recovery applied
                        self.logical_error_count = self.logical_error_count + 1
                    else:
                        self.state_vector.apply_unitary_operation(CircuitUnitaryOperation.get_combined_operation_for_pauli_z(targetQubit, self.N))
                elif(isinstance(instruction, NoisyGateInstruction)):
                    self.__noisy_instruction_handler(instruction)
                else:
                    self.__direct_execute__(instruction.getOperation()) 
        else:
            for operation, description in zip(self.operations, self.descriptions):
                if "Measurement" not in description and "Reset" not in description:
                    self.state_vector.apply_unitary_operation(operation)
                else:
                    string = re.search(r"\d+", description)
                    q = int(string.group())
                    if "Measurement" in description:
                        value = self.state_vector.measure_qubit(q)
                        self.classicalBitRegister.write(q, value)
                    else:
                        self.state_vector.reset_qubit(q)
                        self.classicalBitRegister.write(q, 0)
                self.quantum_states.append(self.state_vector.get_quantum_state())
                if print_state:
                    print(description)
                    print(operation)
                    print("Current quantum state")
                    self.state_vector.print()
    
    def measure(self, print_state: bool=False) -> str:
        self.state_vector.measure()
        if print_state:
            print("Measured state:")
            print(self.state_vector.get_classical_state_as_string())
        return self.get_classical_state_as_string()

    def get_classical_state_as_string(self, little_endian_formatted: bool=False):
        string = self.state_vector.get_classical_state_as_string()
        return string if not little_endian_formatted else string[0] + string[1:-1][::-1] + string[-1]
    
    def get_classical_state_of_qubit_as_string(self, qubit):
        string = self.get_classical_state_as_string()
        return "|" + string[qubit+1] + ">" + "\t Measured value of qubit " + str(qubit) 

'''
Class representing a noisy quantum circuit of N qubits.
Inherits from Circuit.
'''
class NoisyCircuit(Circuit):
    def __init__(self, N):
        super().__init__(N)
        self.state_vector = StateVector(self.N)
        self.noisy_operations_state_prep = []
        self.noisy_operations_incoherent = []
        self.noisy_operations_readout = []
        self.x_measures = np.empty(self.N, dtype=object)
        self.y_measures = np.empty(self.N, dtype=object)
        self.z_measures = np.empty(self.N, dtype=object)

        
        # Noisy gates
        self.phi = [0 for _ in range(self.N)] # Keep a list of phi values for every qubit

        # Load in the device parameters json
        device_params = DeviceParameters()
        device_params.load_from_json("./assets/noise_parameters/QiskitKyiv_DeviceParameters.json")
        qiskit_kyiv_parameter_dict = device_params.__dict__()

        # Define a list of parameter values based on the stored device parameters
        self.parameters = {
            "T1": [float(qiskit_kyiv_parameter_dict["T1"][i % len(qiskit_kyiv_parameter_dict["T1"])]) for i in range(self.N)], # Loop over the T1 values of the device parameters to assign to each qubit
            "T2": [float(qiskit_kyiv_parameter_dict["T2"][i % len(qiskit_kyiv_parameter_dict["T2"])]) for i in range(self.N)], # Loop over the T2 values of the device parameters to assign to each qubit
            "p": [float(qiskit_kyiv_parameter_dict["p"][i % len(qiskit_kyiv_parameter_dict["p"])]) for i in range(self.N)], # Loop over the p values of the device parameters to assign to each qubit
        }


    def add_noisy_operation_state_prep(self, p, q):
        noisy_operation_state_prep = (1-p)*Dirac.ket_bra(2,0,0) + p*Dirac.ket_bra(2,1,1)
        combined_noisy_operation_state_prep = CircuitUnitaryOperation.get_combined_operation_for_qubit(noisy_operation_state_prep, q, self.N)
        self.noisy_operations_state_prep.append(combined_noisy_operation_state_prep)

    def add_noisy_operation_coherent_x(self, theta, q):
        theta_radians = (theta/180)*np.pi
        noisy_operation_coherent = QubitUnitaryOperation.get_rotate_x(theta_radians)
        combined_noisy_operation_coherent = CircuitUnitaryOperation.get_combined_operation_for_qubit(noisy_operation_coherent, q, self.N)
        self.descriptions.append(f"Coherent noise rot_X {theta} deg")
        self.operations.append(combined_noisy_operation_coherent)
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[q] = 'N'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)

    def add_noisy_operation_coherent_y(self, theta, q):
        theta_radians = (theta/180)*np.pi
        noisy_operation_coherent = QubitUnitaryOperation.get_rotate_y(theta_radians)
        combined_noisy_operation_coherent = CircuitUnitaryOperation.get_combined_operation_for_qubit(noisy_operation_coherent, q, self.N)
        self.descriptions.append(f"Coherent noise rot_Y {theta} deg")
        self.operations.append(combined_noisy_operation_coherent)
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[q] = 'N'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)
    
    def add_noisy_operation_coherent_z(self, theta, q):
        theta_radians = (theta/180)*np.pi
        noisy_operation_coherent = QubitUnitaryOperation.get_rotate_z(theta_radians)
        combined_noisy_operation_coherent = CircuitUnitaryOperation.get_combined_operation_for_qubit(noisy_operation_coherent, q, self.N)
        self.descriptions.append(f"Coherent noise rot_Z {theta} deg")
        self.operations.append(combined_noisy_operation_coherent)
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[q] = 'N'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)

    def add_noisy_operation_incoherent(self, px, py, pz, q):
        I = QubitUnitaryOperation.get_identity()
        X = QubitUnitaryOperation.get_pauli_x()
        Y = QubitUnitaryOperation.get_pauli_y()
        Z = QubitUnitaryOperation.get_pauli_z()
        noisy_operation_incoherent = (1-px-py-pz)*I + px*X + py*Y +pz*Z
        combined_noisy_operation_incoherent = CircuitUnitaryOperation.get_combined_operation_for_qubit(noisy_operation_incoherent, q, self.N)
        self.noisy_operations_incoherent.append(combined_noisy_operation_incoherent)

    def add_noisy_operation_readout(self, epsilon, nu, q):
        noisy_operation_readout = np.array([[1-epsilon,nu],[epsilon,1-nu]])
        combined_noisy_operation_readout = CircuitUnitaryOperation.get_combined_operation_for_qubit(noisy_operation_readout, q, self.N)
        self.noisy_operations_readout.append(combined_noisy_operation_readout)

    def create_ideal_circuit(self):
        ideal_circuit = NoisyCircuit(self.N)
        for operation, description, gate in zip(self.operations, self.descriptions, self.gates):
            if "Coherent noise" not in description:
                ideal_circuit.operations.append(operation)
                ideal_circuit.descriptions.append(description)
                ideal_circuit.gates.append(gate)
        return ideal_circuit

    # Override method execute() from class Circuit
    def execute(self, print_state=False):
        self.state_vector = StateVector(self.N)
        for noisy_operation in self.noisy_operations_state_prep:
            self.state_vector.apply_noisy_operation(noisy_operation)
        self.quantum_states = [self.state_vector.get_quantum_state()]
        for q in range(self.N):
            self.x_measures[q] = [self.state_vector.measure_x(q)]
            self.y_measures[q] = [self.state_vector.measure_y(q)]
            self.z_measures[q] = [self.state_vector.measure_z(q)]
        if print_state:
            print("Initial quantum state")
            self.state_vector.print()
        for operation, description in zip(self.operations, self.descriptions):
            self.state_vector.apply_unitary_operation(operation)
            self.quantum_states.append(self.state_vector.get_quantum_state())
            if "Coherent noise" not in description:
                for noisy_operation in self.noisy_operations_incoherent:
                    self.state_vector.apply_noisy_operation(noisy_operation)
                for q in range(self.N):
                    self.x_measures[q].append(self.state_vector.measure_x(q))
                    self.y_measures[q].append(self.state_vector.measure_y(q))
                    self.z_measures[q].append(self.state_vector.measure_z(q))
                if print_state:
                    print(description)
                    print(operation)
                    print("Current quantum state")
                    self.state_vector.print()

    # Override method measure() from class Circuit
    def measure(self, print_state=False):
        for noisy_operation in self.noisy_operations_readout:
            self.state_vector.apply_noisy_operation(noisy_operation)
        self.state_vector.noisy_measure()
        if print_state:
            print("Measured state:")
            print(self.state_vector.get_classical_state_as_string())

    def get_x_measures(self, q):
        return self.x_measures[q]
    
    def get_y_measures(self, q):
        return self.y_measures[q]
    
    def get_z_measures(self, q):
        return self.z_measures[q]

    # Define a virtual Rz gate to mimic the "quantum-gates" package
    def virtual_rotate_z(self, q: int, theta: float):
        """ This gate is implemented virtualy and thus is not executed on the actual qubit!
        Update the phase to implement virtual Rz(theta) gate on qubit i

        Args:
            q: index of the qubit
            theta: angle of rotation on the Bloch sphere

        Returns:
             None
        """
        self.phi[q] = self.phi[q] + theta

    # Define the new Pauli X gate with integrated noise 
    def noisy_pauli_x(self, q: int, p: float = None, T1: float = None, T2: float = None):
        """Adds a noisy Pauli X gate to the circuit

        Args:
            q (int): Qubit to operate on.
            p (float): Single-qubit depolarizing error probability.
            T1 (float): Qubit's amplitude damping time in ns.
            T2 (float): Qubit's dephasing time in ns.
        """
        # If any noise parameter is None use the generated value
        if p is None:
            p = self.parameters["p"][q]
        if T1 is None:
            T1 = self.parameters["T1"][q]
        if T2 is None:
            T2 = self.parameters["T2"][q]

        # Theta and phi to construct Pauli X
        theta = np.pi
        phi = -self.phi[q]

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_qubit(NoisyGate.construct(theta, phi, p, T1, T2), q, self.N)
        self.descriptions.append(f"Noisy Pauli X on qubit {q}")
        self.operations.append(combined_operation)
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[q] = 'X'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)

    # Define the new Pauli Y gate with integrated noise 
    def noisy_pauli_y(self, q: int, p: float= None, T1: float= None, T2: float= None):
        """Adds a noisy Pauli Y gate to the circuit

        Args:
            q (int): Qubit to operate on.
            p (float): Single-qubit depolarizing error probability.
            T1 (float): Qubit's amplitude damping time in ns.
            T2 (float): Qubit's dephasing time in ns.
        """

        # If any noise parameter is None use the generated value
        if p is None:
            p = self.parameters["p"][q]
        if T1 is None:
            T1 = self.parameters["T1"][q]
        if T2 is None:
            T2 = self.parameters["T2"][q]

        # First execute a virtual Rz gate
        self.virtual_rotate_z(q, np.pi)

        # Theta and phi to construct Pauli Y
        theta = np.pi
        phi = -self.phi[q]

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_qubit(NoisyGate.construct(theta, phi, p, T1, T2), q, self.N)
        self.descriptions.append(f"Noisy Pauli Y on qubit {q}")
        self.operations.append(combined_operation)
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[q] = 'Y'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)


    # Define the new "virtual" Pauli Z gate
    def noisy_pauli_z(self, q: int):
        """This gate is implemented virtualy and thus is not executed on the actual qubit!

        Args:
            q (int): Qubit to operate on.
        """

        # Execute a virtual Rz gate
        self.virtual_rotate_z(q, np.pi)
    
    # Define the new hadamard gate with integrated noise 
    def noisy_hadamard(self, q: int, p: float= None, T1: float= None, T2: float= None):
        """Adds a noisy hadamard gate to the circuit

        Args:
            q (int): Qubit to operate on.
            p (float): Single-qubit depolarizing error probability.
            T1 (float): Qubit's amplitude damping time in ns.
            T2 (float): Qubit's dephasing time in ns.
        """

        # If any noise parameter is None use the generated value
        if p is None:
            p = self.parameters["p"][q]
        if T1 is None:
            T1 = self.parameters["T1"][q]
        if T2 is None:
            T2 = self.parameters["T2"][q]

        # First execute a virtual Rz gate
        self.virtual_rotate_z(q, np.pi / 2)

        # Theta for a square root X gate and phi from phi list
        theta = np.pi / 2
        phi = -self.phi[q]

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_qubit(NoisyGate.construct(theta, phi, p, T1, T2), q, self.N)
        self.descriptions.append(f"Noisy Hadamard on qubit {q}")
        self.operations.append(combined_operation)
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[q] = 'H'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)

        # To complete the gate end with a virtual Rz gate
        self.virtual_rotate_z(q, np.pi / 2)
    
    # Define the new "virtual" Phase gate
    def noisy_phase_virtual(self, q: int):
        """This gate is implemented virtualy and thus is not executed on the actual qubit!

        Args:
            q (int): Qubit to operate on.
            p (float): Single-qubit depolarizing error probability.
            T1 (float): Qubit's amplitude damping time in ns.
            T2 (float): Qubit's dephasing time in ns.
        """

        # Execute a virtual Rz gate
        self.virtual_rotate_z(q, np.pi / 2)
    
    def noisy_phase(self, theta: float, q: int, p: float = None, T1: float = None, T2: float = None):
        """This gate is implemented making use of perfect hadamard in combination with a X gate!

        Args:
            theta (float): Angle of rotation on the Bloch sphere.
            q (int): Qubit to operate on.
            p (float): Single-qubit depolarizing error probability.
            T1 (float): Qubit's amplitude damping time in ns.
            T2 (float): Qubit's dephasing time in ns.
        """

         # If any noise parameter is None use the generated value
        if p is None:
            p = self.parameters["p"][q]
        if T1 is None:
            T1 = self.parameters["T1"][q]
        if T2 is None:
            T2 = self.parameters["T2"][q]

        phi = -self.phi[q]

        # Set in hadamard basis
        self.hadamard(q)

        # X gate is now 
        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_qubit(NoisyGate.construct(theta, phi, p, T1, T2), q, self.N)
        self.descriptions.append(f"Noisy X rotation of {theta} on qubit {q}")
        self.operations.append(combined_operation)
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[q] = 'X'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)

        # Get out of hadamard basis
        self.hadamard(q)

    def noisy_sqrt_x(self, q: int, p: float = None, T1: float = None, T2: float = None):
        """Adds a noisy Sqrt(X) gate to the circuit

        Args:
            q (int): Qubit to operate on.
            p (float): Single-qubit depolarizing error probability.
            T1 (float): Qubit's amplitude damping time in ns.
            T2 (float): Qubit's dephasing time in ns.
        """
        # If any noise parameter is None use the generated value
        if p is None:
            p = self.parameters["p"][q]
        if T1 is None:
            T1 = self.parameters["T1"][q]
        if T2 is None:
            T2 = self.parameters["T2"][q]

        # Theta and phi to construct Sqrt(X)
        theta = np.pi/2
        phi = -self.phi[q]

        combined_operation = CircuitUnitaryOperation.get_combined_operation_for_qubit(NoisyGate.construct(theta, phi, p, T1, T2), q, self.N)
        self.descriptions.append(f"Noisy Sqrt(X) on qubit {q}")
        self.operations.append(combined_operation)
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[q] = 'x'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)


    # Define the new cnot gate with integrated noise 
    def noisy_cnot(self, c_qubit: int, t_qubit: int, c_p: float= None, t_p: float= None, c_T1: float= None, t_T1: float= None, c_T2: float= None, t_T2: float= None, gate_error: float=None):
        """Adds a noisy cnot gate to the circuit with depolarizing and
        relaxation errors on both qubits during the unitary evolution.

        Args:
            c_qubit (int): Control qubit for the gate.
            t_qubit (int): Target qubit for the gate.
            c_p (float): Depolarizing error probability for the control qubit.
            t_p (float): Depolarizing error probability for the target qubit.
            c_T1 (float): Amplitude damping time in ns for the control qubit.
            t_T1 (float): Amplitude damping time in ns for the target qubit.
            c_T2 (float): Dephasing time in ns for the contorl qubit.
            t_T2 (float): Dephasing time in ns for the target qubit.
            gate_error (float): CNOT depolarizing error probability.
        """

        # If any noise parameter is None use the generated value
        if c_p is None:
            c_p = self.parameters["p"][c_qubit]
        if c_T1 is None:
            c_T1 = self.parameters["T1"][c_qubit]
        if c_T2 is None:
            c_T2 = self.parameters["T2"][c_qubit]
        if t_p is None:
            t_p = self.parameters["p"][t_qubit]
        if t_T1 is None:
            t_T1 = self.parameters["T1"][t_qubit]
        if t_T2 is None:
            t_T2 = self.parameters["T2"][t_qubit]
        if gate_error is None:
            gate_error = 0.015

        gate_length = 5.61777778e-07

        # Create an identity matrix for the remaining qubits
        identity_matrix = np.eye(2**(self.N - 2))

        # Create cnot matrix
        if c_qubit < t_qubit:
            cnot_operation = NoisyGate.construct_cnot(self.phi[c_qubit], self.phi[t_qubit], gate_length, gate_error, c_p, t_p, c_T1, c_T2, t_T1, t_T2)
            self.phi[c_qubit] = self.phi[c_qubit] - np.pi/2
        else:
            cnot_operation = NoisyGate.construct_cnot_inverse(self.phi[c_qubit], self.phi[t_qubit], gate_length, gate_error, c_p, t_p, c_T1, c_T2, t_T1, t_T2)
            self.phi[c_qubit] = self.phi[c_qubit] + np.pi/2 + np.pi
            self.phi[t_qubit] = self.phi[t_qubit] + np.pi/2
        
        # Perform the Kronecker product to expand the CNOT operation
        cnot_operation = np.kron(cnot_operation, identity_matrix)

        # Create swap matrices
        if c_qubit < t_qubit:
            # control qubit should swap with qubit 0
            swap_control = CircuitUnitaryOperation.get_combined_operation_for_swap(0, c_qubit, self.N) if c_qubit != 0 else np.eye(2**self.N)
            swap_target = CircuitUnitaryOperation.get_combined_operation_for_swap(1, t_qubit, self.N) if t_qubit != 1 else np.eye(2**self.N)
        else:
            # target qubit should swap with qubit 0
            swap_target = CircuitUnitaryOperation.get_combined_operation_for_swap(0, t_qubit, self.N) if t_qubit != 0 else np.eye(2**self.N)
            swap_control = CircuitUnitaryOperation.get_combined_operation_for_swap(1, c_qubit, self.N) if c_qubit != 1 else np.eye(2**self.N)

        # Construct the full CNOT operation with swaps
        operation = swap_control @ swap_target @ cnot_operation @ swap_target.T.conj() @ swap_control.T.conj() 

        self.descriptions.append(f"Noisy CNOT with target qubit {t_qubit} and control qubit {c_qubit}")
        self.operations.append(operation)
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[c_qubit] = '*'
        gate_as_list[t_qubit] = 'x'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)
    
    # Define the new ecr gate with integrated noise 
    def noisy_ecr(self, c_qubit: int, t_qubit: int, c_p: float= None, t_p: float= None, c_T1: float= None, t_T1: float= None, c_T2: float= None, t_T2: float= None, gate_error: float=None):
        """Adds a noisy ecr gate to the circuit with depolarizing and
        relaxation errors on both qubits during the unitary evolution.

        Args:
            c_qubit (int): Control qubit for the gate.
            t_qubit (int): Target qubit for the gate.
            c_p (float): Depolarizing error probability for the control qubit.
            t_p (float): Depolarizing error probability for the target qubit.
            c_T1 (float): Amplitude damping time in ns for the control qubit.
            t_T1 (float): Amplitude damping time in ns for the target qubit.
            c_T2 (float): Dephasing time in ns for the contorl qubit.
            t_T2 (float): Dephasing time in ns for the target qubit.
            gate_error (float): ecr depolarizing error probability.
        """

        # If any noise parameter is None use the generated value
        if c_p is None:
            c_p = self.parameters["p"][c_qubit]
        if c_T1 is None:
            c_T1 = self.parameters["T1"][c_qubit]
        if c_T2 is None:
            c_T2 = self.parameters["T2"][c_qubit]
        if t_p is None:
            t_p = self.parameters["p"][t_qubit]
        if t_T1 is None:
            t_T1 = self.parameters["T1"][t_qubit]
        if t_T2 is None:
            t_T2 = self.parameters["T2"][t_qubit]
        if gate_error is None:
            gate_error = 0.015

        gate_length = 5.61777778e-07

        # Create an identity matrix for the remaining qubits
        identity_matrix = np.eye(2**(self.N - 2))

        # Create ecr matrix
        if c_qubit < t_qubit:
            ecr_operation = NoisyGate.construct_ecr(self.phi[c_qubit], self.phi[t_qubit], gate_length, gate_error, c_p, t_p, c_T1, c_T2, t_T1, t_T2)
        else:
            ecr_operation = NoisyGate.construct_ecr_inverse(self.phi[c_qubit], self.phi[t_qubit], gate_length, gate_error, c_p, t_p, c_T1, c_T2, t_T1, t_T2)
        
        # Perform the Kronecker product to expand the ecr operation
        ecr_operation = np.kron(ecr_operation, identity_matrix)

        # Create swap matrices
        if c_qubit < t_qubit:
            # control qubit should swap with qubit 0
            swap_control = CircuitUnitaryOperation.get_combined_operation_for_swap(0, c_qubit, self.N) if c_qubit != 0 else np.eye(2**self.N)
            swap_target = CircuitUnitaryOperation.get_combined_operation_for_swap(1, t_qubit, self.N) if t_qubit != 1 else np.eye(2**self.N)
        else:
            # target qubit should swap with qubit 0
            swap_target = CircuitUnitaryOperation.get_combined_operation_for_swap(0, t_qubit, self.N) if t_qubit != 0 else np.eye(2**self.N)
            swap_control = CircuitUnitaryOperation.get_combined_operation_for_swap(1, c_qubit, self.N) if c_qubit != 1 else np.eye(2**self.N)

        # Construct the full ecr operation with swaps
        ecr_operation = swap_control @ swap_target @ ecr_operation @ swap_target.T.conj() @ swap_control.T.conj()

        self.descriptions.append(f"Noisy ecr with target qubit {t_qubit} and control qubit {c_qubit}")
        self.operations.append(ecr_operation)
        gate_as_string = '.'*self.N
        gate_as_list = list(gate_as_string)
        gate_as_list[c_qubit] = '*'
        gate_as_list[t_qubit] = 'E'
        gate_as_string = ''.join(gate_as_list)
        self.gates.append(gate_as_string)


"""
Circuit creation for quantum Fourier transform (QFT) and inverse quantum Fourier transform (iQFT).
"""
class QuantumFourier:
    """
    Private function to rotate the qubits of a circuit for quantum Fourier transform (QFT)
    """
    @staticmethod
    def __qft_rotations(circuit:Circuit, n):
        if n == 0:
            return circuit
        
        # Apply Hadamard operation to the most significant qubit
        circuit.hadamard(n-1) 
        for qubit in range(n-1):
            # For each less significant qubit, a controlled rotation
            # is applied with a smaller angle.
            circuit.controlled_phase(-np.pi/2**(n-1-qubit), qubit, n-1)
        
        # Recursive function call with n-1
        QuantumFourier.__qft_rotations(circuit, n-1)


    """
    Function to create a circuit for quantum Fourier transform (QFT)
    """
    @staticmethod
    def create_qft_circuit(N, swap_registers=False):
        circuit = Circuit(N)
        QuantumFourier.__qft_rotations(circuit, N)
        if swap_registers:
            circuit.swap_registers()
        return circuit

    """
    Function to create a circuit for inverse quantum Fourier transform (QFT)
    """
    @staticmethod
    def create_iqft_circuit(N, swap_registers=False):
        circuit = QuantumFourier.create_qft_circuit(N, swap_registers=swap_registers)
        return circuit.create_inverse_circuit()



"""
Supporting functions for execution, measurement, and visualisation of intermediate quantum states.
"""
class QuantumUtil:
    """
    Function to run a quantum circuit and measure the classical state.
    """
    @staticmethod
    def run_circuit(circuit:Circuit, nr_runs=1000):
        # if(circuit.save_instructions):
        #     raise Exception("Direct Operation Execution is enabled, QuantumUtil not supported with this flag")
        result = []
        for i in range(nr_runs):
            circuit.execute()
            circuit.measure()
            result.append(circuit.get_classical_state_as_string())
        return result

    """
    Function to run a quantum circuit once and measure the classical state many times.
    """
    @staticmethod
    def measure_circuit(circuit:Circuit, nr_measurements=1000, little_endian_formatted: bool=False):
        circuit.execute()
        result = []
        for i in range(nr_measurements):
            circuit.measure()
            result.append(circuit.get_classical_state_as_string(little_endian_formatted))
        return result

    """"
    Function to run a quantum circuit many times and measure its classical register state many times
    """
    @staticmethod
    def measure_circuit_bit_register(circuit:Circuit, nr_measurements=100, beginBit: int=0, endBit: int = 0):
        result = []
        for i in range(nr_measurements):
            circuit.execute()
            result.append(circuit.classicalBitRegister.toString(beginBit, endBit))
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


    """
    Function to create an animation of the execution of a noisy quantum circuit using Bloch spheres.
    Parameter ideal_circuit should have the same number of qubits and the same gate operations as noisy_circuit, 
    but without decoherence or quantum noise.
    """
    @staticmethod
    def create_animation(ideal_circuit:NoisyCircuit, noisy_circuit:NoisyCircuit=None):

        # Define the number of frames for the animation
        num_frames = len(ideal_circuit.get_x_measures(0))

        # Create a figure for the plot
        fig_width = 3 * ideal_circuit.N
        fig_height = 4
        fig = plt.figure()
        fig.set_size_inches(fig_width, fig_height)

        # Create a Bloch sphere object for each qubit
        b = []
        for q in range(ideal_circuit.N):
            ax = fig.add_subplot(1, ideal_circuit.N, q+1, projection='3d')
            b.append(Bloch(fig=fig, axes=ax))

        # Function to update the Bloch sphere for each frame
        def animate(i):
            for q in range(ideal_circuit.N):
                # Clear the previous vectors and points
                b[q].clear()  

                # Define the state vector for the ideal circuit
                x = ideal_circuit.get_x_measures(q)[i]
                y = ideal_circuit.get_y_measures(q)[i]
                z = ideal_circuit.get_z_measures(q)[i]
                ideal_state_vector = np.array([x, y, z])

                # Add the ideal state to the Bloch sphere
                b[q].add_vectors(ideal_state_vector)

                # Define the state vector for the noisy circuit
                if not noisy_circuit is None:
                    x = noisy_circuit.get_x_measures(q)[i]
                    y = noisy_circuit.get_y_measures(q)[i]
                    z = noisy_circuit.get_z_measures(q)[i]
                    noisy_state_vector = np.array([x, y, z])

                    # Add the noisy state to the Bloch sphere
                    b[q].add_vectors(noisy_state_vector)

                # Green is ideal state, red is noisy state
                b[q].vector_color = ['g', 'r']

                # Redraw the Bloch sphere
                b[q].make_sphere()  

        # Create an animation
        ani = animation.FuncAnimation(fig, animate, frames=num_frames, repeat=False)

        return ani

    
"""
Abstact Base Class
"""

"""
This class holds all quantum gate instruction objects.
These instructions are being used by the execute() function of the Circuit class if save_instruction flag is TRUE
"""

class GateInstruction(ABC):
    @abstractmethod
    def getOperation(self):
        pass

class Identity(GateInstruction):
    def __init__(self, totalQubits: int, targetQubit: int):
        self.totalQubits = totalQubits
        self.targetQubit = targetQubit

    def getOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_identity(self.targetQubit, self.totalQubits)


class Pauli_X(GateInstruction):
    def __init__(self, totalQubits: int, targetQubit: int):
        self.totalQubits = totalQubits
        self.targetQubit = targetQubit

    def getOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_pauli_x(self.targetQubit, self.totalQubits)
    
class Pauli_Y(GateInstruction):
    def __init__(self, totalQubits: int, targetQubit: int):
        self.totalQubits = totalQubits
        self.targetQubit = targetQubit

    def getOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_pauli_y(self.targetQubit, self.totalQubits)
    
class Pauli_Z(GateInstruction):
    def __init__(self, totalQubits: int, targetQubit: int):
        self.totalQubits = totalQubits
        self.targetQubit = targetQubit

    def getOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_pauli_z(self.targetQubit, self.totalQubits)
    
class Hadamard(GateInstruction):
    def __init__(self, totalQubits: int, targetQubit: int):
        self.totalQubits = totalQubits
        self.targetQubit = targetQubit

    def getOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_hadamard(self.targetQubit, self.totalQubits)
    
class Phase(GateInstruction):
    def __init__(self, totalQubits: int, targetQubit: int, theta: float):
        self.totalQubits = totalQubits
        self.targetQubit = targetQubit
        self.theta = theta

    def getOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_phase(self.theta, self.targetQubit, self.totalQubits)

class Rotate_X(GateInstruction):
    def __init__(self, totalQubits: int, targetQubit: int, theta: float):
        self.totalQubits = totalQubits
        self.targetQubit = targetQubit
        self.theta = theta

    def getOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_rotate_x(self.theta, self.targetQubit, self.totalQubits)

class Rotate_Y(GateInstruction):
    def __init__(self, totalQubits: int, targetQubit: int, theta: float):
        self.totalQubits = totalQubits
        self.targetQubit = targetQubit
        self.theta = theta

    def getOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_rotate_y(self.theta, self.targetQubit, self.totalQubits)


class Rotate_Z(GateInstruction):
    def __init__(self, totalQubits: int, targetQubit: int, theta: float):
        self.totalQubits = totalQubits
        self.targetQubit = targetQubit
        self.theta = theta

    def getOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_rotate_z(self.theta, self.targetQubit, self.totalQubits)
    
class U_Gate(GateInstruction):
    def __init__(self, totalQubits: int, targetQubit: int, theta: float, phi: float, lam: float):
        self.totalQubits = totalQubits
        self.targetQubit = targetQubit
        self.theta = theta
        self.phi = phi
        self.lam = lam

    def getOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_u_gate(self.theta, self.phi, self.lam, self.targetQubit, self.totalQubits)

class CNOT(GateInstruction):
    def __init__(self, totalQubits: int, targetQubit: int, controlQubit: int):
        self.totalQubits = totalQubits
        self.targetQubit = targetQubit
        self.controlQubit = controlQubit

    def getOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_cnot(self.controlQubit, self.targetQubit, self.totalQubits)
    
class Controlled_Pauli_Y(GateInstruction):
    def __init__(self, totalQubits: int, targetQubit: int, controlQubit: int):
        self.totalQubits = totalQubits
        self.targetQubit = targetQubit
        self.controlQubit = controlQubit

    def getOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_controlled_pauli_y(self.controlQubit, self.targetQubit, self.totalQubits)
       
class Controlled_Pauli_Z(GateInstruction):
    def __init__(self, totalQubits: int, targetQubit: int, controlQubit: int):
        self.totalQubits = totalQubits
        self.targetQubit = targetQubit
        self.controlQubit = controlQubit

    def getOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_controlled_pauli_z(self.controlQubit, self.targetQubit, self.totalQubits)
    
class Controlled_Hadamard(GateInstruction):
    def __init__(self, totalQubits: int, targetQubit: int, controlQubit: int):
        self.totalQubits = totalQubits
        self.targetQubit = targetQubit
        self.controlQubit = controlQubit

    def getOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_controlled_hadamard(self.controlQubit, self.targetQubit, self.totalQubits)
    
class Controlled_Phase(GateInstruction):
    def __init__(self, theta: float, totalQubits: int, targetQubit: int, controlQubit: int):
        self.theta = theta
        self.totalQubits = totalQubits
        self.targetQubit = targetQubit
        self.controlQubit = controlQubit

    def getOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_controlled_phase(self.theta, self.controlQubit, self.targetQubit, self.totalQubits)
    
class Controlled_Rotate_X(GateInstruction):
    def __init__(self, theta: float, totalQubits: int, targetQubit: int, controlQubit: int):
        self.theta = theta
        self.totalQubits = totalQubits
        self.targetQubit = targetQubit
        self.controlQubit = controlQubit

    def getOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_controlled_rotate_x(self.theta, self.controlQubit, self.targetQubit, self.totalQubits)
    
class Controlled_Rotate_Y(GateInstruction):
    def __init__(self, theta: float, totalQubits: int, targetQubit: int, controlQubit: int):
        self.theta = theta
        self.totalQubits = totalQubits
        self.targetQubit = targetQubit
        self.controlQubit = controlQubit

    def getOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_controlled_rotate_y(self.theta, self.controlQubit, self.targetQubit, self.totalQubits)
    
class Controlled_Rotate_Z(GateInstruction):
    def __init__(self, theta: float, totalQubits: int, targetQubit: int, controlQubit: int):
        self.theta = theta
        self.totalQubits = totalQubits
        self.targetQubit = targetQubit
        self.controlQubit = controlQubit

    def getOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_controlled_rotate_z(self.theta, self.controlQubit, self.targetQubit, self.totalQubits)

class Controlled_U_Gate(GateInstruction):
    def __init__(self, theta: float, phi: float, lam: float, totalQubits: int, targetQubit: int, controlQubit: int):
        self.theta = theta
        self.phi = phi
        self.lam = lam
        self.totalQubits = totalQubits
        self.targetQubit = targetQubit
        self.controlQubit = controlQubit

    def getOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_controlled_u_gate(self.theta, self.phi, self.lam, self.controlQubit, self.targetQubit, self.totalQubits)

class Controlled_Unitary_Operation(GateInstruction):
    def __init__(self, totalQubits: int, operation, targetQubit: int, controlQubit: int):
        self.totalQubits = totalQubits
        self.operation = operation
        self.targetQubit = targetQubit
        self.controlQubit = controlQubit

    def getOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_controlled_unitary_operation_general(self.operation, self.controlQubit, self.targetQubit, self.totalQubits)
    
 
class Swap(GateInstruction):
    def __init__(self, totalQubits: int, a: int, b: int):
        self.totalQubits = totalQubits
        self.a = a
        self.b = b

    def getOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_swap(self.a, self.b, self.totalQubits)
   
class Fredkin(GateInstruction):
    def __init__(self, totalQubits: int, controlQubit: int, a: int, b: int):
        self.totalQubits = totalQubits
        self.controlQubit = controlQubit
        self.a = a
        self.b = b

    def getOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_fredkin(self.controlQubit, self.a, self.b, self.totalQubits)
    
class Toffoli(GateInstruction):
    def __init__(self, totalQubits: int, control_a: int, control_b: int, targetQubit: int):
        self.totalQubits = totalQubits
        self.control_a = control_a
        self.control_b = control_b
        self.targetQubit = targetQubit

    def getOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_toffoli(self.control_a, self.control_b, self.targetQubit, self.totalQubits)
    
class Multi_Controlled_Pauli_Z(GateInstruction):
    def __init__(self, totalQubits: int):
        self.totalQubits = totalQubits

    def getOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_multi_controlled_pauli_z_operation(self.control_a, self.control_b, self.targetQubit, self.totalQubits)
    

class Multi_Controlled_Pauli_X(GateInstruction):
    def __init__(self, totalQubits: int):
        self.totalQubits = totalQubits

    def getOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_multi_controlled_pauli_x_operation(self.totalQubits)

class Generic_Toffoli(GateInstruction):
    def __init__(self, totalQubits: int, controls:list[int], target:int):
        self.totalQubits = totalQubits
        self.controls = controls
        self.target = target

    def getOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_generic_toffoli(self.controls, self.target, self.totalQubits)
    
class Measurement():
    def __init__(self, measureQubit: int, dataBit: int):
        self.measureQubit = measureQubit
        self.dataBit = dataBit
"""
Applies a Pauli x gate to the targetQubit IF the readBit is equal to 1
Results in the qubit always being |0>
"""
class Reset():
    def __init__(self, targetQubit: int, readBit: int):
        self.readBit = readBit
        self.targetQubit = targetQubit

"""
Special operation, used for executing a recovery from a phase flip when using surface codes
"""
class Recovery_Phase_Flip():
    """
    Creates a Recovery instruction for phase flips

    Args:
        syndromeStartBit: Beginning index bit of the syndrome, standard size of such register is 4
        totalQubits: Total number of qubits in the circuit
    """
    def __init__(self, syndromeStartBit: int, totalQubits: int):
        self.syndromeStartBit = syndromeStartBit
        self.totalQubits = totalQubits

    def __get_recovery(self, syndrome: str) -> int:
        recovery_actions = {
            '1000': '0', # Phase flip on D1
            '1010': '1', # Phase flip on D2
            '0010': '2', # Phase flip on D3 or D6
            '0100': '3', # Phase flip on D4 or D7
            '0110': '4', # Phase flip on D5
            '0101': '7', # Phase flip on D8
            '0001': '8', # Phase flip on D9
            '0000': '-1', # No Phase flips detected
            'Logical Error': '-2'
        }
        return recovery_actions.get(syndrome, recovery_actions['Logical Error'])

    def getTargetQubit(self, register: ClassicalBitRegister) -> int:
        # Deciding the best suitable recovery option based on a classical 4 bits register...
        syndrome = str(register.read(self.syndromeStartBit)) + str(register.read(self.syndromeStartBit + 1)) + str(register.read(self.syndromeStartBit + 2)) + str(register.read(self.syndromeStartBit + 3))
        targetQubit = int(self.__get_recovery(syndrome))
        if(targetQubit == -2):
            print("Logical error when deciding phase flip recovery option")
            return targetQubit
        elif(targetQubit == -1):
            print("No phase flips detected, no recovery applied")
            return targetQubit
        else:
            print(f"Phase flip recovery (Pauli Z) applied on qubit: {targetQubit}")
            return targetQubit
        return targetQubit    
"""
Special operation, used for executing a recovery from a bit flip when using surface codes
"""
class Recovery_Bit_Flip():
    """
    Creates a Recovery instruction for bit flips

    Args:
        syndromeStartBit: Beginning index bit of the syndrome, standard size of such register is 4
        totalQubits: Total number of qubits in the circuit
    """
    def __init__(self, syndromeStartBit: int, totalQubits: int):
        self.syndromeStartBit = syndromeStartBit
        self.totalQubits = totalQubits

    def __get_recovery(self, syndrome: str) -> int:
        recovery_actions = {
            '0100': '0', # Bit flip on D1 or D2
            '0001': '2', # Bit flip on D3
            '1100': '3', # Bit flip on D4
            '0110': '4', # Bit flip on D5
            '0011': '5', # Bit flip on D6
            '1000': '6', # Bit flip on D7
            '0010': '7', # Bit flip on D8 or D9
            '0000': '-1', # No Bit flips detected
            'Logical Error': '-2'
        }
        return recovery_actions.get(syndrome, recovery_actions['Logical Error'])

    def getTargetQubit(self, register: ClassicalBitRegister) -> int:
        # Deciding the best suitable recovery option based on a classical 4 bits register...
        syndrome = str(register.read(self.syndromeStartBit)) + str(register.read(self.syndromeStartBit + 1)) + str(register.read(self.syndromeStartBit + 2)) + str(register.read(self.syndromeStartBit + 3))
        targetQubit = int(self.__get_recovery(syndrome))
        if(targetQubit == -2):
            print("Logical error when deciding bit flip recovery option")
            return targetQubit
        elif(targetQubit == -1):
            print("No bit flips detected, no recovery applied")
            return targetQubit
        else:
            print(f"Bit flip recovery (Pauli X) applied on qubit: {targetQubit}")
            return targetQubit
        return targetQubit                                                                                  

class NoisyGateInstruction(ABC):
    @abstractmethod
    def getNoisyOperation(self):
        pass

class NoisyPauliX(NoisyGateInstruction):
    def __init__(self, q: int, totalQubits: int, p: float = None, T1: float = None, T2: float = None, ):
        self.q = q
        self.N = totalQubits
        self.p = p
        self.T1 = T1
        self.T2 = T2

    def setTheta(self, theta: float):
        self.theta = theta
        
    def setPhi(self, phi: float):
        self.phi = phi

    def getNoisyOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(NoisyGate.construct(self.theta, self.phi, self.p, self.T1, self.T2),self.q, self.N)
    
class NoisyPauliY(NoisyGateInstruction):
    def __init__(self, q: int, totalQubits: int, p: float = None, T1: float = None, T2: float = None, ):
        self.q = q
        self.N = totalQubits
        self.p = p
        self.T1 = T1
        self.T2 = T2

    def setTheta(self, theta: float):
        self.theta = theta
        
    def setPhi(self, phi: float):
        self.phi = phi
        
    def getNoisyOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(NoisyGate.construct(self.theta, self.phi, self.p, self.T1, self.T2),self.q, self.N)

class NoisyPauliZ(NoisyGateInstruction):
    def __init__(self, q: int, totalQubits: int, p: float = None, T1: float = None, T2: float = None, ):
        self.q = q
        self.N = totalQubits
        self.p = p
        self.T1 = T1
        self.T2 = T2

    def setTheta(self, theta: float):
        self.theta = theta
        
    def setPhi(self, phi: float):
        self.phi = phi

    def getNoisyOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(NoisyGate.construct(self.theta, self.phi, self.p, self.T1, self.T2),self.q, self.N)

class NoisyPhase(NoisyGateInstruction):
    def __init__(self, theta: float, q: int, totalQubits: int, p: float = None, T1: float = None, T2: float = None, ):
        self.q = q
        self.theta = theta
        self.N = totalQubits
        self.p = p
        self.T1 = T1
        self.T2 = T2

    def setPhi(self, phi: float):
        self.phi = phi

    def getNoisyOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(NoisyGate.construct(self.theta, self.phi, self.p, self.T1, self.T2),self.q, self.N)



class NoisyHadamard(NoisyGateInstruction):
    def __init__(self, q: int, totalQubits: int, p: float = None, T1: float = None, T2: float = None, ):
        self.q = q
        self.N = totalQubits
        self.p = p
        self.T1 = T1
        self.T2 = T2

    def setTheta(self, theta: float):
        self.theta = theta
        
    def setPhi(self, phi: float):
        self.phi = phi
        
    def getNoisyOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(NoisyGate.construct(self.theta, self.phi, self.p, self.T1, self.T2),self.q, self.N)
    
class NoisyCNOT(NoisyGateInstruction):
    def __init__(self, c_qubit: int, t_qubit: int, N: int, c_p: float= None, t_p: float= None, c_T1: float= None, t_T1: float= None, c_T2: float= None, t_T2: float= None, gate_error: float=None):
        self.c_qubit = c_qubit
        self.t_qubit = t_qubit
        self.N = N
        self.c_p = c_p
        self.t_p = t_p
        self.c_T1 = c_T1
        self.t_T1 = t_T1
        self.c_T2 = c_T2
        self.t_T2 = t_T2
        self.gate_error = gate_error

    def setTheta(self, theta: float):
        self.theta = theta
        
    def setPhiControl(self, phi: float):
        self.c_phi = phi

    def setPhiTarget(self, phi: float):
        self.t_phi = phi

    def getNoisyOperation(self) -> CircuitUnitaryOperation:
        # This is a solution for mimicking noise of two qubits by adding single qubit gates, returns a pauli x gate
        # return CircuitUnitaryOperation.get_combined_operation_for_qubit(NoisyGate.construct(self.theta, self.c_phi, self.p, self.T1, self.T2),self.q, self.N)

        gate_length = 5.61777778e-07

        # Create an identity matrix for the remaining qubits
        identity_matrix = np.eye(2**(self.N - 2))

        # Create cnot matrix
        if self.c_qubit < self.t_qubit:
            cnot_operation = NoisyGate.construct_cnot(self.c_phi, self.t_phi, gate_length, self.gate_error, self.c_p, self.t_p, self.c_T1, self.c_T2, self.t_T1, self.t_T2)
            self.c_phi = self.c_phi - np.pi/2
        else:
            cnot_operation = NoisyGate.construct_cnot_inverse(self.c_phi, self.t_phi, gate_length, self.gate_error, self.c_p, self.t_p, self.c_T1, self.c_T2, self.t_T1, self.t_T2)
            self.c_phi = self.c_phi + np.pi/2 + np.pi
            self.t_phi = self.t_phi + np.pi/2
        
        # Perform the Kronecker product to expand the CNOT operation
        cnot_operation = np.kron(cnot_operation, identity_matrix)

        # Create swap matrices
        if self.c_qubit < self.t_qubit:
            # control qubit should swap with qubit 0
            swap_control = CircuitUnitaryOperation.get_combined_operation_for_swap(0, self.c_qubit, self.N) if self.c_qubit != 0 else np.eye(2**self.N)
            swap_target = CircuitUnitaryOperation.get_combined_operation_for_swap(1, self.t_qubit, self.N) if self.t_qubit != 1 else np.eye(2**self.N)
        else:
            # target qubit should swap with qubit 0
            swap_target = CircuitUnitaryOperation.get_combined_operation_for_swap(0, self.t_qubit, self.N) if self.t_qubit != 0 else np.eye(2**self.N)
            swap_control = CircuitUnitaryOperation.get_combined_operation_for_swap(1, self.c_qubit, self.N) if self.c_qubit != 1 else np.eye(2**self.N)

        # Construct the full CNOT operation with swaps
        operation = swap_control @ swap_target @ cnot_operation @ swap_target.T.conj() @ swap_control.T.conj() 
        return operation
    
class NoisyReset(NoisyGateInstruction):
    def __init__(self, q: int, readBit: int, totalQubits: int, p: float = None, T1: float = None, T2: float = None):
        self.q = q
        self.readBit = readBit
        self.N = totalQubits
        self.p = p
        self.T1 = T1
        self.T2 = T2

    def setTheta(self, theta: float):
        self.theta = theta
        
    def setPhi(self, phi: float):
        self.phi = phi

    def getNoisyOperation(self) -> CircuitUnitaryOperation:
        return CircuitUnitaryOperation.get_combined_operation_for_qubit(NoisyGate.construct(self.theta, self.phi, self.p, self.T1, self.T2),self.q, self.N)

# The following classes are adapted from `quantum-gates`:
# Source: https://pypi.org/project/quantum-gates/
# License: MIT License
# Original Authors: M. Grossi, G. D. Bartolomeo, M. Vischi, P. Da Rold, R. Wixinger

"""
Class for loading, storing, validating and passing device parameters. These parameter represent the noise level of the
device.
"""

import os
from datetime import datetime
import json
import numpy as np

class DeviceParameters(object):
    """Snapshot of the noise of the IBM backend. Can load and save the properties.

    Args:
        qubits_layout (list[int]): Layout of the qubits.

    Attributes:
        qubits_layout (list[int]): Layout of the qubits.
        nr_of_qubits (int): Number of qubits to be used.
        T1 (np.array): T1 time.
        T2 (np.array): T2 time.
        p (np.array): To be added.
        rout (np.array): To be added.
        p_int (np.array): Error probabilites in the 2 qubit gate.
        p_int (np.array): Gate time to implement controlled not operations in the 2 qubit gate.
        tm (np.array): To be added.
        dt (np.array): To be added.
        
    """

    def __init__(self):
        self.qubits_layout = None
        self.nr_of_qubits = None
        self.T1 = None
        self.T2 = None
        self.p = None
        self.rout = None
        self.p_int = None
        self.t_int = None
        self.tm = None
        self.dt = None
        self.metadata = None
        self._names = ["T1", "T2", "p", "rout", "p_int", "t_int", "tm", "dt", "metadata"]
        self._f_txt = ["T1.txt", "T2.txt", "p.txt", "rout.txt", "p_int.txt", "t_int.txt", "tm.txt", "dt.txt",
                       "metadata.json"]

    def load_from_json(self, location: str):
        """ Load device parameters from single json file at the location.
        """
        # Verify that it exists
        self._json_exists_at_location(location)

        # Load
        f = open(location)
        data_dict = json.load(f)

        # Check json keys
        if any((name not in data_dict for name in self._names)):
            raise Exception("Loading of device parameters from json not successful: At least one quantity is missing.")

        # Add lists to instance as arrays
        self.qubits_layout = np.array(data_dict["metadata"]["qubits_layout"])
        self.nr_of_qubits = data_dict["metadata"]["config"]["n_qubits"]
        self.T1 = np.array(data_dict["T1"])
        self.T2 = np.array(data_dict["T2"])
        self.p = np.array(data_dict["p"])
        self.rout = np.array(data_dict["rout"])
        self.p_int = np.array(data_dict["p_int"])
        self.t_int = np.array(data_dict["t_int"])
        self.tm = np.array(data_dict["tm"])
        self.dt = np.array(data_dict["dt"])
        self.metadata = data_dict["metadata"]

        # Verify
        if not self.is_complete():
            raise Exception("Loading of device parameters from json was not successful: Did not pass verification.")

        return

    def load_from_texts(self, location: str):
        """ Load device parameters from many text files at the location.
        """

        # Verify that exists
        self._texts_exist_at_location(location)

        # Load -> If the text has only one line, we have to make it into an 1x1 array explicitely.
        if self.nr_of_qubits == 1:
            # Here we use 'array' because with only one qubit 'loadtxt' doesn't load an array
            self.T1 = np.array([np.loadtxt(location + self.f_T1)])
            self.T2 = np.array([np.loadtxt(location + self.f_T2)])
            self.p = np.array([np.loadtxt(location + self.f_p)])
            self.rout = np.array([np.loadtxt(location + self.f_rout)])
            self.p_int = np.array([np.loadtxt(location + self.f_p_int)])
            self.t_int = np.array([np.loadtxt(location + self.f_t_int)])
            self.tm = np.array([np.loadtxt(location + self.f_tm)])
        else:
            self.T1 = np.loadtxt(location + self.f_T1)
            self.T2 = np.loadtxt(location + self.f_T2)
            self.p = np.loadtxt(location + self.f_p)
            self.rout = np.loadtxt(location + self.f_rout)
            self.p_int = np.loadtxt(location + self.f_p_int)
            self.t_int = np.loadtxt(location + self.f_t_int)
            self.tm = np.loadtxt(location + self.f_tm)
        self.dt = np.array([np.loadtxt(location + self.f_dt)])
        with open(location + self.f_metadata, "r") as metadata_file:
            self.metadata = json.load(metadata_file)

        # Verify
        if not self.is_complete():
            raise Exception("Loading of device parameters from text files was not successful: Did not pass verification.")

        return

    def get_as_tuple(self) -> tuple:
        """ Get the parameters as a tuple. The parameters have to be already loaded.
        """
        if not self.is_complete():
            raise Exception("Exception in DeviceParameters.get_as_tuble(): At least one of the parameters is None.")
        return self.T1, self.T2, self.p, self.rout, self.p_int, self.t_int, self.tm, self.dt, self.metadata

    def is_complete(self) -> bool:
        """ Returns whether all device parameters have been successfully initialized.
        """
        # Check not None
        if any((
                self.T1 is None,
                self.T2 is None,
                self.p is None,
                self.rout is None,
                self.p_int is None,
                self.t_int is None,
                self.tm is None,
                self.dt is None,
                self.metadata is None)):
            return False

        return True

    def check_T1_and_T2_times(self, do_raise_exception: bool) -> bool:
        """ Checks the T1 and T2 times. Raises an exception in case of invalid T1, T2 times if the flag is set. Returns
            whether or not all qubits are flawless.
        """

        print("Verifying the T1 and T2 times of the device: ")
        nr_bad_qubits = 0
        for i, (T1, T2) in enumerate(zip(self.T1, self.T2)):
            if T1 >= 2*T2:
                nr_bad_qubits += 1
                print('The qubit n.', self.qubits_layout[i], 'is bad.')
                print('Delete the affected qubit from qubits_layout and change the layout.')

        if nr_bad_qubits:
            print(f'Attention, there are {nr_bad_qubits} bad qubits.')
            print('In case of side effects contact Jay Gambetta.')
        else:
            print('All right!')

        if nr_bad_qubits and do_raise_exception:
            raise Exception(f'Stop simulation: The DeviceParameters class found {nr_bad_qubits} bad qubits.')

        return nr_bad_qubits == 0

    def _texts_exist_at_location(self, location):
        """ Checks if the text files with the device parameters exist at the expected location. Raises an exception
            if more than one text is missing.
        """
        missing = [f for f in self._f_txt if not os.path.exists(location + f)]
        if len(missing) > 0:
            raise FileNotFoundError(
                f"DeviceParameter found that at {location} the files {missing} are missing."
            )
        return

    def _json_exists_at_location(self, location):
        """ Checks if the json files with the device parameters exist, otherwise raises an exception.
        """
        if not os.path.exists(location):
            raise FileNotFoundError(
                f"DeviceParameter found that at {location} the file is missing."
            )
        return

    def __dict__(self):
        """ Get dict representation. """
        return {
            "T1": self.T1,
            "T2": self.T2,
            "p": self.p,
            "rout": self.rout,
            "p_int": self.p_int,
            "t_int": self.t_int,
            "tm": self.tm,
            "dt": self.dt,
            "metadata": self.metadata
        }

    def __str__(self):
        """ Representation as str. """
        return json.dumps(self.__dict__(), indent=4, default=default_serializer)

    def __eq__(self, other):
        """ Allows us to compare instances. """
        return self.__str__() == other.__str__()


def default_serializer(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

# The following classes are adapted from `quantum-gates`:
# Source: https://pypi.org/project/quantum-gates/
# License: MIT License
# Original Authors: M. Grossi, G. D. Bartolomeo, M. Vischi, P. Da Rold, R. Wixinger

import numpy as np
import scipy.integrate
import scipy.stats

"""Define pulse shapes and their parametrizations.

Attributes:
    constant_pulse (ConstantPulse): Pulse of constant height which uses an analytical lookup in the integrator.
    constant_pulse_numerical (ConstantPulseNumerical): Pulse of constant height which uses numerical integration.
    gaussian_pulse (GaussianPulse): Gaussian pulse with location = 0.5 and scale = 0.25.

Todo:
    * Add parametrized pulses based on Power Series or Fourier Series.
"""
class Pulse(object):
    """ Parent class for pulses with basic utility.

    Args:
        pulse (callable): Function f: [0,1] -> R>=0: Waveform of the pulse, must integrate up to 1.
        parametrization (callable): Function F: [0,1] -> [0,1]: Parameter integral of the pulse. Monotone with
            F(0) = 0 and F(1) = 1, as well as x <= y implies F(x) <= F(y).
        perform_checks (bool): Tells whether the properties of the pulse and parametrization should be validated.
        use_lookup (bool): Bool whether the pulse is constant. Then one can lookup the integration result in the
            integrator.

    Example:
        .. code:: python

           from quantum_gates.pulses import Pulse

           pulse = lambda x: 1
           parametrization = lambda x: x

           constant_pulse = Pulse(
               pulse=pulse,
               parametrization=parametrization,
               perform_checks=False
               )

    Attributes:
        pulse:              Waveform of the pulse as function, f: [0,1] -> R, f >= 0
        parametrization:    Parameter integral of the waveform, F: [0,1] -> [0,1], F >= 0, monotonically increasing
        use_lookup:         In the Integrator, should a integration result lookup be used. True if pulse is constant
    """

    epsilon = 1e-6
    check_n_points = 10

    def __init__(self, pulse: callable, parametrization: callable, perform_checks: bool=False, use_lookup: bool=False):
        if perform_checks:
            assert self._pulse_is_valid(pulse), "Pulse was not valid"
            assert self._parametrization_is_valid(parametrization), "Parametrization was not valid"
            assert self._are_compatible(pulse, parametrization), "Pulse and parametrization are incompatible. "
        self.pulse = pulse
        self.parametrization = parametrization
        self.use_lookup = use_lookup

    def get_pulse(self):
        """Get the waveform f of the pulse as callable.
        """
        return self.pulse

    def get_parametrization(self):
        """Get the parametrization F of the pulse as callable.
        """
        return self.parametrization

    def _pulse_is_valid(self, pulse: callable) -> bool:
        """Returns whether the pulse is a probability distribution on [0,1].

        Args:
            pulse (callable): The waveform which is to be checked.

        Returns:
            Result of the check as boolean.
        """
        integrates_to_1 = abs(scipy.integrate.quad(pulse, 0, 1)[0] - 1) < self.epsilon
        is_non_negative = all((pulse(x) >= 0) for x in np.linspace(0, 1, self.check_n_points))
        return integrates_to_1 and is_non_negative

    def _parametrization_is_valid(self, parametrization: callable) -> bool:
        """ Returns whether the parametrization is monotone and has valid bounds.

        Args:
            parametrization (callable): The parametrization which is to be checked.

        Returns:
            Result of the check as boolean.
        """
        starts_at_0 = abs(parametrization(0) - 0) < self.epsilon
        stops_at_0 = abs(parametrization(1) - 1) < self.epsilon
        is_monotone = all((parametrization(x + self.epsilon) >= parametrization(x))
                          for x in np.linspace(0, 1-self.epsilon, self.check_n_points))
        return starts_at_0 and stops_at_0 and is_monotone

    def _are_compatible(self, pulse, parametrization) -> bool:
        """ Returns whether the integral of the pulse is the parametrization.

        Args:
            pulse (callable): The waveform which is to be checked.
            parametrization (callable): The parametrization which is to be checked.

        Returns:
            Result of the check as boolean.
        """
        for x in np.linspace(self.epsilon, 1-self.epsilon, self.check_n_points):
            difference = abs(scipy.integrate.quad(pulse, 0, x)[0] - parametrization(x))
            if difference > self.epsilon:
                return False
        return True


class ConstantPulse(Pulse):
    """Constant pulse which uses the lookup in the integrator.
    """

    def __init__(self):
        super(ConstantPulse, self).__init__(
            pulse=one,
            parametrization=identity,
            perform_checks=False,
            use_lookup=True
        )


class ConstantPulseNumerical(Pulse):
    """Constant pulse which uses numerical integration.

    Note:
        We can use this class for unit testing the ConstantPulse class.
    """
    def __init__(self):
        super(ConstantPulseNumerical, self).__init__(
            pulse=one,
            parametrization=identity,
            perform_checks=False,
            use_lookup=False
        )


class GaussianPulse(Pulse):
    """ Pulse based on a Gaussian located at loc with variance according to scale.

    Make sure that loc is near to the interval [0,1] or has a high variance. Otherwise, the overlap with the
    interval [0,1] is too small.

    Note:
        The integral over the interval [0,1] of the choosen Gaussian should be larger than 1e-6. This is because the
        shape of the pulse is the shape that the Gaussian has in this interval.

    Example:
        .. code:: python

            from quantum_gates.pulses import GaussianPulse

            loc = 0.5   # Location of the Gaussian
            scale = 0.5 # Standard deviation of the Gaussian

            constant_pulse = GaussianPulse(loc=loc, scale=scale)

    Args:
        loc (float): Location of the pulse on the real axis.
        scale (float): Standard deviation or size of the Gaussian pulse.
        perform_check (bool): Whether the pulse should be verified.
    """

    use_lookup = False  # We perform numerical integration in the Integrator

    def __init__(self, loc: float, scale: float, perform_checks: bool=False):
        self._validate_inputs(loc, scale)
        self._loc = loc
        self._scale = scale
        super(GaussianPulse, self).__init__(
            pulse=self._gaussian_pulse,
            parametrization=self._gaussian_parametrization,
            perform_checks=perform_checks,
            use_lookup=False
        )

    def _gaussian_pulse(self, x):
        return scipy.stats.norm.pdf(x, self._loc, self._scale) / (scipy.stats.norm.cdf(1, self._loc, self._scale) - scipy.stats.norm.cdf(0, self._loc, self._scale))

    def _gaussian_parametrization(self, x):
        return (scipy.stats.norm.cdf(x, self._loc, self._scale) - scipy.stats.norm.cdf(0, self._loc, self._scale)) \
               / (scipy.stats.norm.cdf(1, self._loc, self._scale) - scipy.stats.norm.cdf(0, self._loc, self._scale))

    @staticmethod
    def _validate_inputs(loc, scale):
        # Validate type
        valid_types = [int, float, np.float64]
        assert type(scale) in valid_types, f"InputError in GaussianPulse: loc must be float but found {type(loc)}."
        assert type(scale) in valid_types, f"InputError in GaussianPulse: scale must be float but found {type(scale)}."

        # Validate that the denominator used in the further calculation does not evaluate to 0
        denominator = scipy.stats.norm.cdf(1, loc, scale) - scipy.stats.norm.cdf(0, loc, scale)
        assert denominator != 0, \
            "InputError in GaussianPulse: Denominator is zero because of the choice of loc and scale."


def one(x):
    """ Always returns 1.0.
    """
    return 1.0


def identity(x: float):
    """ Always returns the input.
    """
    return x


# Create instances of the different pulse types
constant_pulse = ConstantPulse()
constant_pulse_numerical = ConstantPulseNumerical()
gaussian_pulse = GaussianPulse(loc=0.5, scale=0.25)


""" Evaluates the integrals coming up in the Noisy gates approach for different pulse waveforms.

Because many of the integrals are evaluated many times with the same parameters, we can apply caching to speed things
up.
"""
class Integrator(object):
    """Calculates the integrals for a specific pulse parametrization.

    Args:
        pulse (Pulse): Object specifying the pulse waveform and parametrization.

    Attributes:
        pulse_parametrization (callable): Function F: [0,1] -> [0,1] representing the parametrization of the pulse.
        use_lookup (bool): Tells whether or not the lookup table of the analytical solution should be used.
    """

    _INTEGRAL_LOOKUP = {
        "sin(theta/a)**2": lambda theta, a: np.sin(theta/a)**2,
        "sin(theta/(2*a))**4": lambda theta, a: np.sin(theta/(2*a))**4,
        "sin(theta/a)*sin(theta/(2*a))**2": lambda theta, a: np.sin(theta/a)*np.sin(theta/(2*a))**2,
        "sin(theta/(2*a))**2": lambda theta, a: np.sin(theta/(2*a))**2,
        "cos(theta/a)**2": lambda theta, a: np.cos(theta/a)**2,
        "sin(theta/a)*cos(theta/a)": lambda theta, a: np.sin(theta/a)*np.cos(theta/a),
        "sin(theta/a)": lambda theta, a: np.sin(theta/a),
        "cos(theta/(2*a))**2": lambda theta, a: np.cos(theta/(2*a))**2
    }
    # For each key (integrand), we calculated the result (parametric integral from 0 to theta) using the parametrization
    # theta(t,t0) = omega(t-t0)/a, corresponding to a square pulse, which is one that has constant magnitude.
    _RESULT_LOOKUP = {
        "sin(theta/a)**2": lambda theta, a: a*(2*theta - np.sin(2*theta))/(4*theta),
        "sin(theta/(2*a))**4": lambda theta, a: a*(6*theta-8*np.sin(theta)+np.sin(2*theta))/(16*theta),
        "sin(theta/a)*sin(theta/(2*a))**2": lambda theta, a: a*((np.sin(theta/2))**4)/theta,
        "sin(theta/(2*a))**2": lambda theta, a: a*(theta - np.sin(theta))/(2 * theta),
        "cos(theta/a)**2": lambda theta, a: a*(2*theta + np.sin(2*theta))/(4*theta),
        "sin(theta/a)*cos(theta/a)": lambda theta, a: a*(np.sin(theta))**2/(2*theta),
        "sin(theta/a)": lambda theta, a: a*(1-np.cos(theta))/theta,
        "cos(theta/(2*a))**2": lambda theta, a: a*(theta + np.sin(theta))/(2*theta)
    }

    def __init__(self, pulse: Pulse):
        self.pulse_parametrization = pulse.get_parametrization()
        self.use_lookup = pulse.use_lookup
        self._cache = dict()

    def integrate(self, integrand: str, theta: float, a: float) -> float:
        """ Evaluates the integrand provided as string from zero to a based on the implicit pulse shape scaled by theta.

        If the pulse (pulse_parametrization) is None, we assume that the pulse height is constant. In this case, we do
        not perform numerical calculation but just lookup the result.

        Args:
            integrand (str): Name of the integrand.
            theta (str): Upper limit of the integration. Total area of the pulse waveform.
            a (str): Scaling parameter.

        Returns:
            Integration result as float.
        """

        # Caching
        if (integrand, theta, a) in self._cache:
            return self._cache[(integrand, theta, a)]

        # Input validation
        assert integrand in self._INTEGRAL_LOOKUP.keys(), "Unknown integrand."
        assert a > 0, f"Require non-vanishing gate time but found a = {a}."

        # Pulse is constant -> We can lookup the analytical result
        if self.use_lookup:
            y = self._analytical_integration(integrand, theta, a)

        # Pulse is variable
        else:
            y = self._numerical_integration(integrand, theta, a)

        # Caching
        self._cache[(integrand, theta, a)] = y

        return y

    def _analytical_integration(self, integrand_str: str, theta: float, a: float) -> float:
        """Lookups up the result of the integration for the case that the parametrization is None.

        Note:
            This method can/should only be used when the pulse height is constant. Otherwise, the result would be wrong.

        Args:
            integrand_str (str): Name of the integrand.
            theta (float): Upper limit of the integration. Total area of the pulse waveform.
            a (float): Scaling parameter.
        """
        integral = self._RESULT_LOOKUP[integrand_str]
        return integral(theta, a)

    def _numerical_integration(self, integrand_name: str, theta: float, a: float) -> float:
        """Looks up the integrand as function and performs numerical integration from 0 to theta.

        Uses the the parametrization specified in the class instance.

        Args:
            integrand_name (str): Name of the integrand.
            theta (float): Upper limit of the integration. Total area of the pulse waveform.
            a (float): Scaling parameter.

        Returns:
            Result of the integration as float.
        """
        integrand = self._INTEGRAL_LOOKUP[integrand_name]

        # The parametrization is a monotone function with param(t=0) == 0 and param(t=1) == 1.
        param = self.pulse_parametrization

        # We scale this parametrization such that scaled_param(t=0) == 0 and scaled_param(t=1) == theta.
        scaled_param = lambda t: param(t) * theta

        # We parametrize the integrand and integrate it from 0 to a. Integral should go from 0 to a.
        integrand_p = lambda t: integrand(scaled_param(t), a)
        y, abserr = scipy.integrate.quad(integrand_p, 0, a)

        return y
    
# Create the integrator for the noisy gates to use
integrator = Integrator(constant_pulse)


# Functions in this class are adapted from `quantum-gates`:
# Source: https://pypi.org/project/quantum-gates/
# License: MIT License
# Original Authors: M. Grossi, G. D. Bartolomeo, M. Vischi, P. Da Rold, R. Wixinger
class NoisyGate:
    @staticmethod
    def __get_unitary_contribution(theta, phi):
        """Unitary contribution due to drive Hamiltonian.

        Args:
            theta (float): Angle of rotation on the Bloch sphere.
            phi (float): Phase of the drive defining axis of rotation on the Bloch sphere.

        Returns:
            Array representing the unitary contribution due to drive Hamiltonian.
        """
        return np.array(
            [[np.cos(theta/2), - 1J * np.sin(theta/2) * np.exp(-1J * phi)],
             [- 1J * np.sin(theta/2) * np.exp(1J * phi), np.cos(theta/2)]]
        )
    
    @staticmethod
    def __ito_integrals_for_X_Y_sigma_min(theta):
        """Ito integrals.

        Ito integrals for the following processes:
            * depolarization for X(t)
            * depolarization for Y(t)
            * relaxation for sigma_min(t).

        As illustration, we leave the variables names for X(t) in the calculation.

        Args:
            theta (float): Angle of rotation on the Bloch sphere.

        Returns:
            Tuple of floats representing sampled results of the Ito integrals.
        """
        # Integral of sin(theta)**2
        Vdx_1 = integrator.integrate("sin(theta/a)**2", theta, 1)

        # Integral of sin**4(theta/2)
        Vdx_2 = integrator.integrate("sin(theta/(2*a))**4", theta, 1)

        # Integral of sin(theta) sin**2(theta/2)
        Covdx_12 = integrator.integrate("sin(theta/a)*sin(theta/(2*a))**2", theta, 1)

        # Integral of sin(theta)
        Covdx_1Wdx = integrator.integrate("sin(theta/a)", theta, 1)

        # Integral of sin**2(theta/2)
        Covdx_2Wdx = integrator.integrate("sin(theta/(2*a))**2", theta, 1)

        # Mean and covariance
        meand_x = np.array([0, 0, 0])
        covd_x = np.array([[Vdx_1, Covdx_12, Covdx_1Wdx], [Covdx_12, Vdx_2, Covdx_2Wdx], [Covdx_1Wdx, Covdx_2Wdx, 1]])

        # Sampling
        sample_dx = np.random.multivariate_normal(meand_x, covd_x, 1) # The variance of Wr is 1
        Idx1 = sample_dx[0,0]
        Idx2 = sample_dx[0,1]
        Wdx = sample_dx[0,2]

        return Idx1, Idx2, Wdx

    @staticmethod
    def __ito_integrals_for_Z(theta):
        """Ito integrals.

        Ito integrals for the following processes:
            * depolarization for Z(t)
            * relaxation for Z(t).

        As illustration, we leave the variable names for the depolarization Itô processes depending on Z(t).

        Args:
            theta (float): angle of rotation on the Bloch sphere.

        Returns:
             Tuple of floats representing sampled results of the Ito integrals.
        """

        # Integral of cos(theta)**2
        Vdz_1 = integrator.integrate("cos(theta/a)**2", theta, 1)

        # Integral of sin(theta)**2
        Vdz_2 = integrator.integrate("sin(theta/a)**2", theta, 1)

        # Integral of sin(theta)*cos(theta)
        Covdz_12 = integrator.integrate("sin(theta/a)*cos(theta/a)", theta, 1)

        # Mean and covariance
        meand_z = np.array([0,0])
        covd_z = np.array(
            [[Vdz_1,Covdz_12],
             [Covdz_12, Vdz_2]]
        )

        # Sampling
        sample_dz = np.random.multivariate_normal(meand_z, covd_z, 1)
        Idz1 = sample_dz[0,0]
        Idz2 = sample_dz[0,1]

        return Idz1, Idz2

    @staticmethod
    def __get_depolarization_contribution(theta, phi, ed):
        # Variances and covariances for depolarization Itô processes depending on X(t)
        Idx1, Idx2, Wdx = NoisyGate.__ito_integrals_for_X_Y_sigma_min(theta)
        Idx = ed * np.array([[np.sin(phi)*Idx1,Wdx + (np.exp(-2*1J*phi)-1)*Idx2],[Wdx + (np.exp(+2*1J*phi)-1)*Idx2,-np.sin(phi)*Idx1]])

        #Variances and covariances for depolarization Itô processes depending on Y(t)
        Idy1, Idy2, Wdy = NoisyGate.__ito_integrals_for_X_Y_sigma_min(theta)
        Idy = ed * np.array([[-np.cos(phi)*Idy1, -1J*Wdy + 1J*(np.exp(-2*1J*phi)+1)*Idy2], [1J*Wdy - 1J*(np.exp(2*1J*phi)+1)*Idy2, np.cos(phi)*Idy1]])

        # Variances and covariances for depolarization Itô processes depending on Z(t)
        Idz1, Idz2 = NoisyGate.__ito_integrals_for_Z(theta)
        Idz = ed * np.array(
            [[Idz1, -1J * np.exp(-1J*phi) * Idz2],
             [1J * np.exp(1J*phi) * Idz2, -Idz1]]
        )

        return Idx, Idy, Idz

    @staticmethod
    def __deterministic_relaxation(theta):
        """Deterministic contribution given by relaxation

        Args:
            theta (float): angle of rotation on the Bloch sphere.

        Returns:
            Array representing the deterministic part of the relaxation process.
        """

        # Integral of sin(theta/(2*a))**2
        det1 = integrator.integrate("sin(theta/(2*a))**2", theta, 1)

        # Integral of sin(theta)
        det2 = integrator.integrate("sin(theta/a)", theta, 1)

        # Integral of cos(theta/2)**2
        det3 = integrator.integrate("cos(theta/(2*a))**2", theta, 1)

        return det1, det2, det3

    @staticmethod
    def __get_relaxation_contribution(theta, phi, ep, e1):
        # Variances and covariances for relaxation Itô processes depending on sigma_min(t)
        Ir1, Ir2, Wr = NoisyGate.__ito_integrals_for_X_Y_sigma_min(theta)
        Ir = e1 * np.array([[-1J/2 * np.exp(1J*phi) * Ir1, Wr - Ir2], [np.exp(2*1J*phi)*Ir2,1J/2* np.exp(1J*phi) * Ir1]])

        # Deterministic contribution given by relaxation
        det1, det2, det3 = NoisyGate.__deterministic_relaxation(theta)
        deterministic = -e1**2/2 * np.array([[det1, 1J/2*np.exp(-1J*phi)*det2], [-1J/2*np.exp(1J*phi)*det2, det3]])

        # Variances and covariances for relaxation Itô processes depending on Z(t)
        Ip1, Ip2 = NoisyGate.__ito_integrals_for_Z(theta)
        Ip = ep * np.array([[Ip1, -1J * np.exp(-1J*phi) * Ip2], [1J * np.exp(1J*phi) * Ip2, -Ip1]])

        return Ir, deterministic, Ip
    
    @staticmethod
    def construct(theta, phi, p, T1, T2):
        """Constructs a noisy single qubit gate. 

        Args:
            theta (float): Angle of rotation on the Bloch sphere.
            phi (float): Phase of the drive defining axis of rotation on the Bloch sphere.
            lam (float): Z rotation.
            p (float): Single-qubit depolarizing error probability.
            T1 (float): Qubit's amplitude damping time in ns.
            T2 (float): Qubit's dephasing time in ns.

        Returns:
              Array representing a general single-qubit noisy quantum gate.
        """

        """ 0) CONSTANTS """

        tg = 35 * 10**(-9)
        ed = np.sqrt(p/4)

        # Amplitude damping time is zero
        if T1 == 0:
            e1 = 0
        else:
            e1 = np.sqrt(tg/T1)

        # Dephasing time is zero
        if T2 == 0:
            ep = 0
        else:
            e2 = np.sqrt(tg/T2)
            ep = np.sqrt((1/2) * (e2**2 - e1**2/2))

        """ 2) DEPOLARIZATION CONTRIBUTION """
        Idx, Idy, Idz = NoisyGate.__get_depolarization_contribution(theta, phi, ed)

        """ 3) RELAXATION CONTRIBUTION """
        Ir, deterministic, Ip = NoisyGate.__get_relaxation_contribution(theta, phi, ep, e1)

        """ 4) COMBINE CONTRIBUTIONS """
        return NoisyGate.__get_unitary_contribution(theta, phi) @ scipy.linalg.expm(deterministic) @ scipy.linalg.expm(1J * Idx + 1J * Idy + 1J * Idz + 1J * Ir + 1J * Ip)
    
    @staticmethod
    def _ito_integrals_for_depolarization_process(omega, phi, a) -> tuple[float]:
        """ Ito integrals.

         Used for the depolarization Itô processes depending on one of
            * [tensor(ID,Z)](t)
            * [tensor(X,ID)](t)
            * [tensor(Y,ID)](t)
            * [tensor(sigma_min,ID)](t)

        As illustration, we leave the variable names from the version with [tensor(ID,Z)](t).

        Args:
            omega: integral of theta from t0 to t1.
            phi: phase of the drive defining axis of rotation on the Bloch sphere.
            a: fraction representing CR gate time / gate time.

        Returns:
            Tuple of floats representing sampled results of the Ito integrals.
        """

        # Integral of cos(omega/a)**2
        Vp_trg_1 = integrator.integrate("cos(theta/a)**2", omega, a)

        # Integral of sin(omega/a)**2
        Vp_trg_2 = integrator.integrate("sin(theta/a)**2", omega, a)

        # Integral of sin(omega/a)*cos(omega/a)
        Covp_trg_12 = integrator.integrate("sin(theta/a)*cos(theta/a)", omega, a)

        # Mean and covariance
        meanp_trg = [0, 0]
        covp_trg = [[Vp_trg_1, Covp_trg_12], [Covp_trg_12, Vp_trg_2]]

        # Sample
        sample_p_trg = np.random.multivariate_normal(meanp_trg, covp_trg, 1)
        Ip_trg_1 = sample_p_trg[0,0]
        Ip_trg_2 = sample_p_trg[0,1]

        return Ip_trg_1, Ip_trg_2

    @staticmethod
    def _ito_integrals_for_depolarization_process_reversed_tensor(omega, a) -> tuple[float]:
        """ Ito integrals.

        Used for the depolarization Itô processes depending on one of
            * [tensor(ID,X)](t)
            * [tensor(ID,Y)](t)

        As illustration, we leave the variable names from the version with [tensor(ID,Y)](t).

        Args:
            omega (float): Integral of theta from t0 to t1.
            a (float): Fraction representing CR gate time / gate time.

        Returns:
            Tuple of floats representing sampled results of the Ito integrals.
        """

        # Integral of sin**2(omega/a)
        Vdy_trg_1 = integrator.integrate("sin(theta/a)**2", omega, a)

        # Integral of sin(omega/(2*a))**4
        Vdy_trg_2 = integrator.integrate("sin(theta/(2*a))**4", omega, a)

        # Integral of sin(omega/a) sin**2(omega/(2*a))
        Covdy_trg_12 = integrator.integrate("sin(theta/a)*sin(theta/(2*a))**2", omega, a)

        # Integral of sin(omega/a)
        Covdy_trg_1Wdy = integrator.integrate("sin(theta/a)", omega, a)

        # Integral of sin(omega/(2*a))**2
        Covdy_trg_2Wdy = integrator.integrate("sin(theta/(2*a))**2", omega, a)

        meandy_trg = np.array([0, 0, 0])
        covdy_trg = np.array(
            [[Vdy_trg_1, Covdy_trg_12, Covdy_trg_1Wdy],
             [Covdy_trg_12, Vdy_trg_2, Covdy_trg_2Wdy],
             [Covdy_trg_1Wdy, Covdy_trg_2Wdy, a]]
        )

        # The variance of Wdy is a
        sample_dy_trg = np.random.multivariate_normal(meandy_trg, covdy_trg, 1)

        Idy_trg_1 = sample_dy_trg[0,0]
        Idy_trg_2 = sample_dy_trg[0,1]
        Wdy_trg = sample_dy_trg[0,2]

        return Idy_trg_1, Idy_trg_2,  Wdy_trg

    @staticmethod
    def __get_cr_gate_contribution(theta, phi, t_cr, p, c_T1, c_T2, t_T1, t_T2):
        """Generates a CR gate.

        This is the 2 order approximated solution, non-unitary matrix. It implements the CR two-qubit noisy quantum gate
        with depolarizing and relaxation errors on both qubits during the unitary evolution.

        Args:
            theta (float): Angle of rotation on the Bloch sphere.
            phi (float): Phase of the drive defining axis of rotation on the Bloch sphere.
            t_cr (float): CR gate time in ns.
            p (float): Depolarizing error probability.
            T1 (float): Qubit's amplitude damping time in ns.
            T2 (float): Qubit's dephasing time in ns.

        Returns:
              CR two-qubit noisy quantum gate (numpy array)
        """

        """ 0) CONSTANTS """

        tg = 35 * 10**(-9)
        omega = theta
        a = t_cr / tg
        assert t_cr > 0, f"Expected t_cr to be > 0 but found {t_cr}"
        assert tg > 0, f"Expected tg to be > 0 but found {tg}"
        ed_cr = np.sqrt(p/(4*a))

        if c_T1 == 0:
            e1_ctr = 0
        else:
            e1_ctr = np.sqrt(tg/c_T1)

        if c_T2 == 0:
            ep_ctr = 0
        else:
            e2_ctr = np.sqrt(tg/c_T2)
            ep_ctr = np.sqrt((1/2) * (e2_ctr**2 - e1_ctr**2/2))

        if t_T1 == 0:
            e1_trg = 0
        else:
            e1_trg = np.sqrt(tg/t_T1)

        if t_T2 == 0:
            ep_trg = 0
        else:
            e2_trg = np.sqrt(tg/t_T2)
            ep_trg = np.sqrt((1/2) * (e2_trg**2 - e1_trg**2/2))

        U = np.array(
            [[np.cos(theta/2), -1J*np.sin(theta/2) * np.exp(-1J * phi), 0, 0],
             [-1J*np.sin(theta/2) * np.exp(1J * phi), np.cos(theta/2), 0, 0],
             [0, 0, np.cos(theta/2), 1J*np.sin(theta/2) * np.exp(-1J * phi)],
             [0, 0, 1J*np.sin(theta/2) * np.exp(1J * phi), np.cos(theta/2)]]
        )

        """ 1) RELAXATION CONTRIBUTIONS """

        # Variances and covariances for amplitude damping Itô processes depending on [tensor(sigma_min,ID)](t)
        Ir_ctr_1, Ir_ctr_2 = NoisyGate._ito_integrals_for_depolarization_process(omega, phi, a)

        Ir_ctr = e1_ctr * np.array(
            [[0, 0, Ir_ctr_1, 1J*Ir_ctr_2 * np.exp(-1J * phi)],
             [0, 0, 1J*Ir_ctr_2 * np.exp(1J * phi), Ir_ctr_1],
             [0, 0, 0, 0],
             [0, 0, 0, 0]]
        )

        # Variances and covariances for amplitude damping Itô processes depending on [tensor(ID,sigma_min)](t)
        Ir_trg_1, Ir_trg_2, Wr_trg = NoisyGate._ito_integrals_for_depolarization_process_reversed_tensor(omega, a)
        Ir_trg = e1_trg * np.array(
            [[-1J*(1/2)*Ir_trg_1*np.exp(1J*phi), Wr_trg-Ir_trg_2, 0, 0],
             [Ir_trg_2*np.exp(2*1J*phi), 1J*(1/2)*Ir_trg_1*np.exp(1J*phi), 0, 0],
             [0, 0, 1J*(1/2)*Ir_trg_1*np.exp(1J*phi),Wr_trg-Ir_trg_2],
             [0, 0, Ir_trg_2*np.exp(2*1J*phi), -1J*(1/2)*Ir_trg_1*np.exp(1J*phi)]]
        )

        # Variances and covariances for phase damping Itô processes depending on [tensor(Z,ID)](t)
        Wp_ctr = np.random.normal(0, np.sqrt(a))
        Ip_ctr = ep_ctr * np.array(
            [[Wp_ctr, 0, 0, 0],
             [0, Wp_ctr, 0, 0],
             [0, 0, -Wp_ctr, 0],
             [0, 0, 0, -Wp_ctr]]
        )

        # Variances and covariances for phase damping Itô processes depending on [tensor(ID,Z)](t)
        Ip_trg_1, Ip_trg_2 = NoisyGate._ito_integrals_for_depolarization_process(omega, phi, a)
        Ip_trg = ep_trg * np.array(
            [[Ip_trg_1, -1J*Ip_trg_2*np.exp(-1J*phi), 0, 0],
             [1J*Ip_trg_2*np.exp(1J*phi), -Ip_trg_1, 0, 0],
             [0, 0, Ip_trg_1, 1J*Ip_trg_2*np.exp(-1J*phi)],
             [0, 0, -1J*Ip_trg_2*np.exp(1J*phi), -Ip_trg_1]]
        )

        #Deterministic contribution given by relaxation
        det1 = (a*omega-a*np.sin(omega))/(2*omega)
        det2 = (a/omega)*(1-np.cos(omega))
        det3 = a/(2*omega)*(omega+np.sin(omega))

        deterministic_r_ctr = -e1_ctr**2/2 * np.array([[0,0,0,0],[0,0,0,0],[0,0,a,0],[0,0,0,a]])
        deterministic_r_trg = -e1_trg**2/2 * np.array(
            [[det1,1J*(1/2)*det2*np.exp(-1J*phi),0,0],
             [-1J*(1/2)*det2*np.exp(1J*phi),det3,0,0],
             [0,0,det1,-1J*(1/2)*det2*np.exp(-1J*phi)],[0,0,1J*(1/2)*det2*np.exp(1J*phi),det3]]
        )

        """ 2) DEPOLARIZATION CONTRIBUTIONS """

        # Variances and covariances for depolarization Itô processes depending on [tensor(X,ID)](t)
        Idx_ctr_1, Idx_ctr_2 = NoisyGate._ito_integrals_for_depolarization_process(omega, phi, a)
        Idx_ctr = ed_cr * np.array(
            [[0, 0, Idx_ctr_1, 1J*Idx_ctr_2 * np.exp(-1J * phi)],
             [0, 0, 1J*Idx_ctr_2 * np.exp(1J * phi), Idx_ctr_1],
             [Idx_ctr_1, -1J*Idx_ctr_2 * np.exp(-1J * phi), 0, 0],
             [-1J*Idx_ctr_2 * np.exp(1J * phi), Idx_ctr_1, 0, 0]]
        )

        # Variances and covariances for depolarization Itô processes depending on [tensor(Y,ID)](t)
        Idy_ctr_1, Idy_ctr_2 = NoisyGate._ito_integrals_for_depolarization_process(omega, phi, a)
        Idy_ctr = ed_cr * np.array(
            [[0, 0, -1J*Idy_ctr_1, Idy_ctr_2 * np.exp(-1J * phi)],
             [0, 0, Idy_ctr_2 * np.exp(1J * phi), -1J*Idy_ctr_1],
             [1J*Idy_ctr_1, Idy_ctr_2 * np.exp(-1J * phi), 0, 0],
             [Idy_ctr_2 * np.exp(1J * phi), 1J*Idy_ctr_1, 0, 0]]
        )

        # Variances and covariances for depolarization Itô processes depending on [tensor(Z,ID)](t)
        Wdz_ctr = np.random.normal(0, np.sqrt(a))
        Idz_ctr = ed_cr * np.array(
            [[Wdz_ctr, 0, 0, 0],
             [0, Wdz_ctr, 0, 0],
             [0, 0, -Wdz_ctr, 0],
             [0, 0, 0, -Wdz_ctr]]
        )

        # Variances and covariances for depolarization Itô processes depending on [tensor(ID,X)](t)
        Idx_trg_1, Idx_trg_2, Wdx_trg = NoisyGate._ito_integrals_for_depolarization_process_reversed_tensor(omega, a)

        Idx_trg = ed_cr * np.array(
            [[Idx_trg_1 * np.sin(phi), Wdx_trg + (np.exp(-2*1J*phi)-1)*Idx_trg_2, 0, 0],
             [Wdx_trg + (np.exp(2*1J*phi)-1)*Idx_trg_2, -Idx_trg_1*np.sin(phi), 0, 0],
             [0,  0, -Idx_trg_1 * np.sin(phi), Wdx_trg + (np.exp(-2*1J*phi)-1)*Idx_trg_2],
             [0, 0, Wdx_trg + (np.exp(2*1J*phi)-1)*Idx_trg_2, Idx_trg_1 * np.sin(phi)]]
        )

        # Variances and covariances for depolarization Itô processes depending on [tensor(ID,Y)](t)
        Idy_trg_1, Idy_trg_2,  Wdy_trg = NoisyGate._ito_integrals_for_depolarization_process_reversed_tensor(omega, a)
        Idy_trg = ed_cr * np.array(
            [[-Idy_trg_1*np.cos(phi), -1J*Wdy_trg + 1J * (np.exp(-2*1J*phi)+1)*Idy_trg_2, 0, 0],
             [1J*Wdy_trg - 1J * (np.exp(2*1J*phi)+1)*Idy_trg_2, Idy_trg_1*np.cos(phi), 0, 0],
             [0, 0, Idy_trg_1*np.cos(phi), -1J*Wdy_trg + 1J * (np.exp(-2*1J*phi)+1)*Idy_trg_2],
             [0, 0, 1J*Wdy_trg - 1J * (np.exp(2*1J*phi)+1)*Idy_trg_2, -Idy_trg_1*np.cos(phi)]]
        )

        # Variances and covariances for depolarization Itô processes depending on [tensor(ID,Z)](t)
        Idz_trg_1, Idz_trg_2 = NoisyGate._ito_integrals_for_depolarization_process(omega, phi, a)
        Idz_trg = ed_cr * np.array(
            [[Idz_trg_1, -1J*Idz_trg_2*np.exp(-1J*phi), 0, 0],
             [1J*Idz_trg_2*np.exp(1J*phi), -Idz_trg_1, 0, 0],
             [0, 0, Idz_trg_1, 1J*Idz_trg_2*np.exp(-1J*phi)],
             [0, 0, -1J*Idz_trg_2*np.exp(1J*phi), -Idz_trg_1]]
        )

        """ 4) COMBINE CONTRIBUTIONS """
        return U @ scipy.linalg.expm(deterministic_r_ctr + deterministic_r_trg) \
                 @ scipy.linalg.expm(
            1J * Ir_ctr + 1J * Ir_trg
            + 1J * Ip_ctr + 1J * Ip_trg
            + 1J * Idx_ctr + 1J * Idy_ctr + 1J * Idz_ctr
            + 1J * Idx_trg + 1J * Idy_trg + 1J * Idz_trg
        )
    
    @staticmethod
    def __get_relaxation_gate_contribution(Dt, T1, T2):
        """Generates the noisy gate for combined amplitude and phase damping.

        This is the exact solution, a non-unitary matrix. It implements the single-qubit relaxation error on idle
        qubits.

        Args:
            Dt (float): idle time in ns.
            T1 (float): qubit's amplitude damping time in ns.
            T2 (float): qubit's dephasing time in ns.

        Returns:
              Array representing the amplitude and phase damping noise gate.
        """
        # Constants
        # tg = 561.778 # Gate execution time in nanoseconds as provided by Qiskit's ibmb_kyiv device gate time median
        tg = 35 * 10**(-9)
        Dt = Dt / tg

        # Helper function
        def V(Dt) -> float:
            return 1-np.exp(-e1**2 * Dt)

        # Calculations
        if T1 == 0:
            e1 = 0
        else:
            e1 = np.sqrt(tg/T1)

        if T2 == 0:
            ep = 0
        else:
            e2 = np.sqrt(tg/T2)
            ep = np.sqrt((1/2) * (e2**2 - e1**2/2))

        W = np.random.normal(0, np.sqrt(Dt))
        I = np.random.normal(0, np.sqrt(V(Dt)))
        result = np.array(
            [[np.exp(1J * ep * W), 1J * I * np.exp(-1J * ep * W)],
             [0, np.exp(-e1**2/2 * Dt) * np.exp(-1J * ep * W)]]
        )
        return result

    @staticmethod
    def construct_cnot(c_phi, t_phi, t_cnot, p_cnot, c_p, t_p,
                  c_T1, c_T2, t_T1, t_T2):
        """Generates a noisy CNOT gate.

        This is a 2nd order approximated solution, a non-unitary matrix. It implements the CNOT two-qubit noisy quantum
        gate with depolarizing and relaxation errors on both qubits during the unitary evolution.

        Args:
            c_phi (float): Control qubit phase of the drive defining axis of rotation on the Bloch sphere.
            t_phi (float): Target qubit phase of the drive defining axis of rotation on the Bloch sphere.
            t_cnot (float): CNOT gate time in ns
            p_cnot (float): CNOT depolarizing error probability.
            c_p (float): Qubit depolarizing error probability for contorl qubit.
            t_p (float): Qubit depolarizing error probability for target qubit.
            c_T1 (float): Qubit's amplitude damping time in ns for control qubit.
            t_T1 (float): Qubit's amplitude damping time in ns for target qubit.
            c_T2 (float): Qubit's dephasing time in ns for control qubit.
            t_T2 (float): Qubit's dephasing time in ns for target qubit.

        Returns:
              Array representing a CNOT two-qubit noisy quantum gate.
        """

        """ 0) CONSTANTS """
        tg = 35*10**(-9)
        t_cr = t_cnot/2-tg
        p_cr = (4/3) * (1 - np.sqrt(np.sqrt((1 - (3/4) * p_cnot)**2 / ((1-(3/4)*c_p)**2 * (1-(3/4)*t_p)))))

        """ 1) CR gate contributions """
        first_cr = NoisyGate.__get_cr_gate_contribution(-np.pi/4, -t_phi, t_cr, p_cr, c_T1, c_T2, t_T1, t_T2)
        second_cr = NoisyGate.__get_cr_gate_contribution(np.pi/4, -t_phi, t_cr, p_cr, c_T1, c_T2, t_T1, t_T2)

        """ 2) X/Sqrt(X) contributions """
        x_gate = NoisyGate.construct(np.pi, -c_phi+np.pi/2, c_p, c_T1, c_T2)
        sx_gate = NoisyGate.construct(np.pi / 2, -t_phi, t_p, t_T1, t_T2)
        Y_Rz = NoisyGate.construct(-np.pi, -c_phi + np.pi/2 + np.pi/2, c_p, c_T1, c_T2)

        """ 3) Relaxation contribution """
        relaxation_gate = NoisyGate.__get_relaxation_gate_contribution(tg, t_T1, t_T2)

        """ 4) COMBINE CONTRIBUTIONS """
        return first_cr @ np.kron(x_gate, relaxation_gate) @ second_cr @ np.kron(Y_Rz, sx_gate)
    
    @staticmethod
    def construct_cnot_inverse(c_phi, t_phi, t_cnot, p_cnot, c_p, t_p,
                  c_T1, c_T2, t_T1, t_T2):
        """Generates an reversed noisy CNOT gate.

        This is a 2nd order approximated solution, a non-unitary matrix. It implements the CNOT two-qubit noisy quantum
        gate with depolarizing and relaxation errors on both qubits during the unitary evolution.

        Args:
            c_phi (float): Control qubit phase of the drive defining axis of rotation on the Bloch sphere.
            t_phi (float): Target qubit phase of the drive defining axis of rotation on the Bloch sphere.
            t_cnot (float): CNOT gate time in ns
            p_cnot (float): CNOT depolarizing error probability.
            c_p (float): Qubit depolarizing error probability for contorl qubit.
            t_p (float): Qubit depolarizing error probability for target qubit.
            c_T1 (float): Qubit's amplitude damping time in ns for control qubit.
            t_T1 (float): Qubit's amplitude damping time in ns for target qubit.
            c_T2 (float): Qubit's dephasing time in ns for control qubit.
            t_T2 (float): Qubit's dephasing time in ns for target qubit.

        Returns:
              Array representing the reverse CNOT two-qubit noisy quantum gate.
        """

        """ 0) CONSTANTS """
        tg = 35*10**(-9)
        t_cr = (t_cnot-3*tg)/2
        p_cr = (4/3) * (1 - np.sqrt(np.sqrt((1 - (3/4) * p_cnot)**2 / ((1-(3/4)*c_p)**2 * (1-(3/4)*t_p)**3))))

        """ 1) CR gate contributions """
        first_cr = NoisyGate.__get_cr_gate_contribution(-np.pi/4, -c_phi-np.pi, t_cr, p_cr, c_T1, c_T2, t_T1, t_T2)
        second_cr = NoisyGate.__get_cr_gate_contribution(np.pi/4, -c_phi-np.pi, t_cr, p_cr, c_T1, c_T2, t_T1, t_T2)

        """ 2) X/Sqrt(X) contributions """
        Ry = NoisyGate.construct(-np.pi/2, -t_phi-np.pi/2+np.pi/2, t_p, t_T1, t_T2)
        Y_Z = NoisyGate.construct(np.pi/2, -c_phi-np.pi+np.pi/2, c_p, c_T1, c_T2)
        x_gate = NoisyGate.construct(np.pi, -t_phi-np.pi/2, t_p, t_T1, t_T2)
        first_sx_gate = NoisyGate.construct(np.pi/2, -c_phi - np.pi - np.pi/2, c_p, c_T1, c_T2)
        second_sx_gate = NoisyGate.construct(np.pi/2, -t_phi - np.pi/2, c_p, c_T1, c_T2)

        """ 3) Relaxation contribution """
        relaxation_gate = NoisyGate.__get_relaxation_gate_contribution(tg, c_T1, c_T2)

        """ 4) COMBINE CONTRIBUTIONS """
        return np.kron(Ry, first_sx_gate) @ first_cr @ np.kron(x_gate, relaxation_gate) @ second_cr @ np.kron(second_sx_gate, Y_Z)

    @staticmethod
    def construct_ecr(c_phi, t_phi, t_ecr, p_ecr, c_p, t_p,
                  c_T1, c_T2, t_T1, t_T2):
        """Generates a noisy ECR gate.

            This is a 2nd order approximated solution, a non-unitary matrix. It implements the ECR two-qubit noisy quantum
            gate with depolarizing and relaxation errors on both qubits during the unitary evolution.

            Args:
                c_phi (float): Control qubit phase of the drive defining axis of rotation on the Bloch sphere.
                t_phi (float): Target qubit phase of the drive defining axis of rotation on the Bloch sphere.
                t_ecr (float): ECR gate time in ns.
                p_ecr (float): ECR depolarizing error probability.
                c_p (float): Control qubit depolarizing error probability.
                t_p (float): Target qubit depolarizing error probability.
                c_T1 (float): Control qubit's amplitude damping time in ns.
                c_T2 (float): Control qubit's dephasing time in ns.
                t_T1 (float): Target qubit's amplitude damping time in ns.
                t_T2 (float): Target qubit's dephasing time in ns.

            Returns:
                Array representing a ECR two-qubit noisy quantum gate.
            """
        
        """ 0) CONSTANTS """
        tg = 35*10**(-9)
        t_cr = t_ecr/2-tg
        p_cr = (4/3) * (1 - np.sqrt(np.sqrt((1 - (3/4) * p_ecr)**2 / ((1-(3/4)*c_p)**2 * (1-(3/4)*t_p)))))

        """ 1) CR gate contributions """
        first_cr = NoisyGate.__get_cr_gate_contribution(np.pi/4, np.pi-t_phi, t_cr, p_cr, c_T1, c_T2, t_T1, t_T2)
        second_cr = NoisyGate.__get_cr_gate_contribution(-np.pi/4, np.pi-t_phi, t_cr, p_cr, c_T1, c_T2, t_T1, t_T2)

        """ 2) X contribution """
        x_gate = -1J* NoisyGate.construct(np.pi, np.pi-c_phi, c_p, c_T1, c_T2)

        """ 3) Relaxation contribution """
        relaxation_gate = NoisyGate.__get_relaxation_gate_contribution(tg, t_T1, t_T2)
        
        """ 4) COMBINE CONTRIBUTIONS """
        return (first_cr @ np.kron(x_gate , relaxation_gate) @ second_cr)
    
    @staticmethod
    def construct_ecr_inverse(c_phi, t_phi, t_ecr, p_ecr, c_p, t_p,
                  c_T1, c_T2, t_T1, t_T2):
        """Generates a noisy inverse ECR gate.

        This is a 2nd order approximated solution, a non-unitary matrix. It implements the reverse ECR two-qubit noisy quantum
        gate with depolarizing and relaxation errors on both qubits during the unitary evolution.

        Args:
            c_phi (float): Control qubit phase of the drive defining axis of rotation on the Bloch sphere.
            t_phi (float): Target qubit phase of the drive defining axis of rotation on the Bloch sphere.
            t_ecr (float): ECR gate time in ns.
            p_ecr (float): ECR depolarizing error probability.
            c_p (float): Control qubit depolarizing error probability.
            t_p (float): Target qubit depolarizing error probability.
            c_T1 (float): Control qubit's amplitude damping time in ns.
            c_T2 (float): Control qubit's dephasing time in ns.
            t_T1 (float): Target qubit's amplitude damping time in ns.
            t_T2 (float): Target qubit's dephasing time in ns.

        Returns:
              Array representing a reverse ECR two-qubit noisy quantum gate.
        """
        """ 0) CONSTANTS """
        tg = 35*10**(-9)
        t_cr = t_ecr/2-tg
        p_cr = (4/3) * (1 - np.sqrt(np.sqrt((1 - (3/4) * p_ecr)**2 / ((1-(3/4)*c_p)**2 * (1-(3/4)*t_p)))))

        """ 1) CR gate contributions """
        first_cr = NoisyGate.__get_cr_gate_contribution(np.pi/4, np.pi-t_phi, t_cr, p_cr, c_T1, c_T2, t_T1, t_T2)
        second_cr = NoisyGate.__get_cr_gate_contribution(-np.pi/4, np.pi-t_phi, t_cr, p_cr, c_T1, c_T2, t_T1, t_T2)

        """ 2) X/Sqrt(X) contributions """
        x_gate = -1J* NoisyGate.construct(np.pi, np.pi-c_phi, c_p, c_T1, c_T2)
        sx_gate_ctr_1 =  NoisyGate.construct(np.pi/2, -np.pi/2-c_phi, c_p, c_T1, c_T2)
        sx_gate_trg_1 =  NoisyGate.construct(np.pi/2, -np.pi/2-t_phi, t_p, t_T1, t_T2)
        sx_gate_ctr_2 =  NoisyGate.construct(np.pi/2, -np.pi/2-c_phi, c_p, c_T1, c_T2)
        sx_gate_trg_2 =  NoisyGate.construct(np.pi/2, -np.pi/2-t_phi, t_p, t_T1, t_T2)

        """ 3) Relaxation contribution """
        relaxation_gate = NoisyGate.__get_relaxation_gate_contribution(tg, t_T1, t_T2)

        """ 4) COMBINE CONTRIBUTIONS """
        return 1j * np.kron(sx_gate_ctr_1, sx_gate_trg_1) @ (first_cr @ np.kron(x_gate , relaxation_gate) @ second_cr ) @ np.kron(sx_gate_ctr_2, sx_gate_trg_2)

