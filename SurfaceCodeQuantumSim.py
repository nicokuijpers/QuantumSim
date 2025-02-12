from quantumsim import Circuit

class Q(): # Quantum register naming aliases 
    @staticmethod
    def D1() -> int:
        return 0
    @staticmethod
    def D2() -> int:
        return 1
    @staticmethod
    def D3() -> int:
        return 2
    @staticmethod
    def D4() -> int:
        return 3
    @staticmethod
    def D5() -> int:
        return 4
    @staticmethod
    def D6() -> int:
        return 5
    @staticmethod
    def D7() -> int:
        return 6
    @staticmethod
    def D8() -> int:
        return 7
    @staticmethod
    def D9() -> int:
        return 8
    @staticmethod
    def X1() -> int:
        return 9
    @staticmethod
    def X2() -> int:
        return 9
    @staticmethod
    def X3() -> int:
        return 9
    @staticmethod
    def X4() -> int:
        return 9
    @staticmethod
    def Z1() -> int:
        return 9
    @staticmethod
    def Z2() -> int:
        return 9
    @staticmethod
    def Z3() -> int:
        return 9
    @staticmethod
    def Z4() -> int:
        return 9

class C(): # Classical register naming aliases 
    @staticmethod
    def X1() -> int:
        return 0
    @staticmethod
    def X2() -> int:
        return 1
    @staticmethod
    def X3() -> int:
        return 2
    @staticmethod
    def X4() -> int:
        return 3
    @staticmethod
    def Z1() -> int:
        return 4
    @staticmethod
    def Z2() -> int:
        return 5
    @staticmethod
    def Z3() -> int:
        return 6
    @staticmethod
    def Z4() -> int:
        return 7
    @staticmethod
    def D1() -> int:
        return 8
    @staticmethod
    def D2() -> int:
        return 9
    @staticmethod
    def D3() -> int:
        return 10
    @staticmethod
    def D4() -> int:
        return 11
    @staticmethod
    def D5() -> int:
        return 12
    @staticmethod
    def D6() -> int:
        return 13
    @staticmethod
    def D7() -> int:
        return 14
    @staticmethod
    def D8() -> int:
        return 15
    @staticmethod
    def D9() -> int:
        return 16

class SurfaceCodePart:
    def __init__(self, name: str, startIndexGate: int, totalGates: int):
        self.name = name
        self.startIndexGate = startIndexGate
        self.totalGates = totalGates

    def toString(self) -> str:
        return self.name + ", starting from index: " + str(self.startIndexGate) + ", total gates: " + str(self.totalGates) + "\n"

"""
This class offers functions for simulating the rotated surface 17 code implementation using QuantumSim
"""
class SurfaceCode:
    # Sources used for creating this class:
    # https://errorcorrectionzoo.org/c/surface-17#citation-3
    # https://arxiv.org/pdf/2303.17211
    # http://arxiv.org/pdf/1612.08208
    
    def __init__(self):

        # Simulating many qubits is expensive resource wise, the rotated surface 17 code is therefore simulated using only 10 qubits using QuantumSim.
        self.qubits = int(10)
        # All 17 simulated qubits are eventually measured, therefore 17 classical bits are required.
        self.bits = int(17)
        # Memory optimization flag, when simulating many qubits (10 or more) this flag should always be set to TRUE
        self.save_instructions_flag = True

        self.circuit = Circuit(self.qubits, self.bits, self.save_instructions_flag)
        self.circuit.classicalBitRegister.create_partition(0, 3, "AncX")
        self.circuit.classicalBitRegister.create_partition(4, 7, "AncZ")
        self.circuit.classicalBitRegister.create_partition(8, 16, "Data")

        # Keeps track of the circuit parts added to the SurfaceCode class
        self.parts = []

    def has(self, name) -> bool:
        for part in self.parts:
            if(part.name == name):
                return True
        return False
    
    def toString(self) -> str:
        output = "Surface code circuitry consists of: \n"
        for part in self.parts:
            output += part.toString()
        output += "Total gates in SurfaceCode object: " + str(self.circuit.gates.__len__())
        return output

    def add_encoder_circuit(self):
        if(self.has("Encoder")):
            raise Exception("SurfaceCode already has an encoder circuit")
        
        self.parts.append(SurfaceCodePart("Encoder", self.circuit.gates.__len__(), 12))

        # Step 1. initialize D2, D4, D6, D8 in Hadamard basis.
        self.circuit.hadamard(Q.D2())
        self.circuit.hadamard(Q.D4())
        self.circuit.hadamard(Q.D6())
        self.circuit.hadamard(Q.D8())

        # Step 2. Make four different Bell and Greenberger-Horne-Zeilinger states.
        self.circuit.cnot(Q.D2(), Q.D1())
        self.circuit.cnot(Q.D6(), Q.D3())
        self.circuit.cnot(Q.D6(), Q.D5())
        self.circuit.cnot(Q.D4(), Q.D5())
        self.circuit.cnot(Q.D4(), Q.D7())
        self.circuit.cnot(Q.D8(), Q.D9())

        # Step 3. Entangle all Bell and Greenberger-Horne-Zeilinger states.
        self.circuit.cnot(Q.D3(), Q.D2())
        self.circuit.cnot(Q.D7(), Q.D8())

    def remove_encoder_circuit(self):
        for part in self.parts:
            if part.name == "Encoder":
                self.circuit.remove_circuit_part(part.startIndexGate, (part.startIndexGate + part.totalGates))
                self.parts.remove(part)
                return
        raise Exception("No encoder exists in SurfaceCode object")

    def add_decoder_circuit(self):

        if(self.has("Decoder")):
            raise Exception("SurfaceCode already has an decoder circuit")
        
        self.parts.append(SurfaceCodePart("Decoder", self.circuit.gates.__len__(), 12))

        # Decoding is the opposite of encoding, reverting the encoding steps results in the initiliazed state
        # Step 1. Dentangle all Bell and Greenberger-Horne-Zeilinger states.
        self.circuit.cnot(Q.D3(), Q.D2())
        self.circuit.cnot(Q.D7(), Q.D8())

        # Step 2. Dentagle different Bell and Greenberger-Horne-Zeilinger states.
        self.circuit.cnot(Q.D2(), Q.D1())
        self.circuit.cnot(Q.D6(), Q.D3())
        self.circuit.cnot(Q.D6(), Q.D5())
        self.circuit.cnot(Q.D4(), Q.D5())
        self.circuit.cnot(Q.D4(), Q.D7())
        self.circuit.cnot(Q.D8(), Q.D9())

        # Step 3. Put D2, D4, D6, D8 out of the Hadamard basis.
        self.circuit.hadamard(Q.D2())
        self.circuit.hadamard(Q.D4())
        self.circuit.hadamard(Q.D6())
        self.circuit.hadamard(Q.D8())

    def remove_decoder_circuit(self):
        for part in self.parts:
            if part.name == "Decoder":
                self.circuit.remove_circuit_part(part.startIndexGate, (part.startIndexGate + part.totalGates))
                self.parts.remove(part)
                return
        raise Exception("No decoder exists in SurfaceCode object")

    def __add_x1_syndrome_extraction(self):
        self.circuit.hadamard(Q.X1())
        self.circuit.cnot(Q.X1(), Q.D1())
        self.circuit.cnot(Q.X1(), Q.D2())
        self.circuit.hadamard(Q.X1())
        self.circuit.measurement(Q.X1(), C.X1())
        self.circuit.reset(Q.X1(), C.X1())

    def __add_x2_syndrome_extraction(self):
        self.circuit.hadamard(Q.X2())
        self.circuit.cnot(Q.X2(), Q.D7())
        self.circuit.cnot(Q.X2(), Q.D4())
        self.circuit.cnot(Q.X2(), Q.D8())
        self.circuit.cnot(Q.X2(), Q.D5())
        self.circuit.hadamard(Q.X2())
        self.circuit.measurement(Q.X2(), C.X2())
        self.circuit.reset(Q.X2(), C.X2())

    def __add_x3_syndrome_extraction(self):
        self.circuit.hadamard(Q.X3())
        self.circuit.cnot(Q.X3(), Q.D5())
        self.circuit.cnot(Q.X3(), Q.D2())
        self.circuit.cnot(Q.X3(), Q.D6())
        self.circuit.cnot(Q.X3(), Q.D3())
        self.circuit.hadamard(Q.X3())
        self.circuit.measurement(Q.X3(), C.X3())
        self.circuit.reset(Q.X3(), C.X3())

    def __add_x4_syndrome_extraction(self):
        self.circuit.hadamard(Q.X4())
        self.circuit.cnot(Q.X4(), Q.D8())
        self.circuit.cnot(Q.X4(), Q.D9())
        self.circuit.hadamard(Q.X4())
        self.circuit.measurement(Q.X4(), C.X4())
        self.circuit.reset(Q.X4(), C.X4())

    def add_x_stabilizer_syndrome_extraction(self):
        self.parts.append(SurfaceCodePart("Stabilizer X syndrome measurement", self.circuit.gates.__len__(), 28))

        self.__add_x1_syndrome_extraction()
        self.__add_x2_syndrome_extraction()
        self.__add_x3_syndrome_extraction()
        self.__add_x4_syndrome_extraction()

    def __add_z1_syndrome_extraction(self):
        self.circuit.cnot(Q.D7(), Q.Z1())
        self.circuit.cnot(Q.D4(), Q.Z1())
        self.circuit.measurement(Q.Z1(), C.Z1())
        self.circuit.reset(Q.Z1(), C.Z1())

    def __add_z2_syndrome_extraction(self):
        self.circuit.cnot(Q.D4(), Q.Z2())
        self.circuit.cnot(Q.D5(), Q.Z2())
        self.circuit.cnot(Q.D1(), Q.Z2())
        self.circuit.cnot(Q.D2(), Q.Z2())
        self.circuit.measurement(Q.Z2(), C.Z2())
        self.circuit.reset(Q.Z2(), C.Z2())

    def __add_z3_syndrome_extraction(self):
        self.circuit.cnot(Q.D8(), Q.Z3())
        self.circuit.cnot(Q.D9(), Q.Z3())
        self.circuit.cnot(Q.D5(), Q.Z3())
        self.circuit.cnot(Q.D6(), Q.Z3())
        self.circuit.measurement(Q.Z3(), C.Z3())
        self.circuit.reset(Q.Z3(), C.Z3())

    def __add_z4_syndrome_extraction(self):
        self.circuit.cnot(Q.D6(), Q.Z4())
        self.circuit.cnot(Q.D3(), Q.Z4())
        self.circuit.measurement(Q.Z4(), C.Z4())
        self.circuit.reset(Q.Z4(), C.Z4())

    def add_z_stabilizer_syndrome_extraction(self):
        self.parts.append(SurfaceCodePart("Stabilizer Z syndrome measurement", self.circuit.gates.__len__(), 28))

        self.__add_z1_syndrome_extraction()
        self.__add_z2_syndrome_extraction()
        self.__add_z3_syndrome_extraction()
        self.__add_z4_syndrome_extraction()

    def add_bit_flip(self, q: int):
        if(q < 0 or q > 8):
            raise Exception("q: qubit parameter must be within boundaries 0(D1) and 8(D9)")
        self.circuit.pauli_x(q)

    def add_phase_flip(self, q: int):
        if(q < 0 or q > 8):
            raise Exception("q: qubit parameter must be within boundaries 0(D1) and 8(D9)")
        self.circuit.pauli_z(q)

    def add_recovery_from_syndrome_x_stabilizer(self):
        """
        Calculates an appropriate recovery action based on the stabilizer Z syndrome measurement
        """
        self.parts.append(SurfaceCodePart("Syndrome x recovery", self.circuit.gates.__len__(), 1))
        self.circuit.recovery_phase_flip(0)

    def add_recovery_from_syndrome_z_stabilizer(self):
        """
        Calculates an appropriate recovery action based on the stabilizer Z syndrome measurement
        """
        self.parts.append(SurfaceCodePart("Syndrome z recovery", self.circuit.gates.__len__(), 1))

        self.circuit.recovery_bit_flip(4)

    def add_measure_all_data_qubits(self):
        self.parts.append(SurfaceCodePart("Data qubit measurement", self.circuit.gates.__len__(), 9))

        self.circuit.measurement(Q.D1(), C.D1())
        self.circuit.measurement(Q.D2(), C.D2())
        self.circuit.measurement(Q.D3(), C.D3())
        self.circuit.measurement(Q.D4(), C.D4())
        self.circuit.measurement(Q.D5(), C.D5())
        self.circuit.measurement(Q.D6(), C.D6())
        self.circuit.measurement(Q.D7(), C.D7())
        self.circuit.measurement(Q.D8(), C.D8())
        self.circuit.measurement(Q.D9(), C.D9())

    def add_pauli_x_on_all_data_qubits(self):
        self.parts.append(SurfaceCodePart("Pauli X on all data qubits", self.circuit.gates.__len__(), 9))

        self.circuit.pauli_x(Q.D1())
        self.circuit.pauli_x(Q.D2())
        self.circuit.pauli_x(Q.D3())
        self.circuit.pauli_x(Q.D4())
        self.circuit.pauli_x(Q.D5())
        self.circuit.pauli_x(Q.D6())
        self.circuit.pauli_x(Q.D7())
        self.circuit.pauli_x(Q.D8())
        self.circuit.pauli_x(Q.D9())
     
    def add_pauli_z_on_all_data_qubits(self):
        self.parts.append(SurfaceCodePart("Pauli Z on all data qubits", self.circuit.gates.__len__(), 9))

        self.circuit.pauli_z(Q.D1())
        self.circuit.pauli_z(Q.D2())
        self.circuit.pauli_z(Q.D3())
        self.circuit.pauli_z(Q.D4())
        self.circuit.pauli_z(Q.D5())
        self.circuit.pauli_z(Q.D6())
        self.circuit.pauli_z(Q.D7())
        self.circuit.pauli_z(Q.D8())
        self.circuit.pauli_z(Q.D9())

    def add_noisy_pauli_x_on_all_data_qubits(self):
        self.parts.append(SurfaceCodePart("Noisy pauli X on all data qubits", self.circuit.gates.__len__(), 9))

        self.circuit.noisy_pauli_x(Q.D1(), 0.01)
        self.circuit.noisy_pauli_x(Q.D2(), 0.01)
        self.circuit.noisy_pauli_x(Q.D3(), 0.01)
        self.circuit.noisy_pauli_x(Q.D4(), 0.01)
        self.circuit.noisy_pauli_x(Q.D5(), 0.01)
        self.circuit.noisy_pauli_x(Q.D6(), 0.01)
        self.circuit.noisy_pauli_x(Q.D7(), 0.01)
        self.circuit.noisy_pauli_x(Q.D8(), 0.01)
        self.circuit.noisy_pauli_x(Q.D9(), 0.01)




    
