import quantumsim as sim

class Shors:

    def GetPhaseEncoderCircuit():
       phase_encoder_circuit = sim.Circuit(9)
       phase_encoder_circuit.cnot(0, 3)
       phase_encoder_circuit.cnot(0, 6)
       phase_encoder_circuit.hadamard(0)
       phase_encoder_circuit.hadamard(3)
       phase_encoder_circuit.hadamard(6)
       return phase_encoder_circuit
        
    def GetBitEncoderCircuit(start_qubit):
        bit_encoder_circuit = sim.Circuit(9)
        if(start_qubit == 0 or start_qubit == 3 or start_qubit == 6 ):
            bit_encoder_circuit.cnot(start_qubit, start_qubit+1)
            bit_encoder_circuit.cnot(start_qubit, start_qubit+2)
        else:
            raise ValueError("Bit encoder value invalid, must be 0, 3 or 6")
        return bit_encoder_circuit
    
    def GetBitDecoderCircuit(start_qubit):
        bit_decoder_circuit = sim.Circuit(9)
        if(start_qubit == 0 or start_qubit == 3 or start_qubit == 6 ):
            bit_decoder_circuit.cnot(start_qubit, start_qubit+1)
            bit_decoder_circuit.cnot(start_qubit, start_qubit+2)
            bit_decoder_circuit.toffoli(start_qubit+1, start_qubit+2, start_qubit)
        else:
            raise ValueError("Bit decoder value invalid, must be 0, 3 or 6")
        return bit_decoder_circuit
    
    def GetPhaseDecoderCircuit():
        phase_decoder_circuit = sim.Circuit(9)
        phase_decoder_circuit.hadamard(0)
        phase_decoder_circuit.hadamard(3)
        phase_decoder_circuit.hadamard(6)
        phase_decoder_circuit.cnot(0, 3)
        phase_decoder_circuit.cnot(0, 6)
        phase_decoder_circuit.toffoli(3, 6, 0)
        return phase_decoder_circuit

    def GetBitFlipErrorCircuit(start_qubit):
        bit_flip_circuit = sim.Circuit(9)
        bit_flip_circuit.bitflip_error_random()
        # bit_flip_circuit.bitflip_error_random(start_qubit, (start_qubit+2))
        return bit_flip_circuit

    def GetPhaseFlipErrorCircuit(start_qubit):
        phase_flip_circuit = sim.Circuit(9)
        phase_flip_circuit.phaseflip_error_random()
        # phase_flip_circuit.phaseflip_error_random(start_qubit, (start_qubit+2))
        return phase_flip_circuit

