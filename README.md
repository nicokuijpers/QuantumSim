# QuantumSim
QuantumSim is a quantum computer simulator programmed in Python. This repository contains the python code along with several Jupyter notebooks to explain the code and to illustrate possible usage.

## Quantum algorithms
The following quantum algorithms are implemented using QuantumSim:
* Quantum Teleportation
* Quantum Fourier Transform
* Quantum Phase Estimation
* Shor's Algorithm for finding prime factors of an integer
* Grover's search algorithm
* Deutsch-Josza algorithm
* Bernstein-Vazirani algorithm
* A counter to count ones in a binary string
* Application of a counter in Grover's search algorithm to generate binary strings with given number of ones
* A three-qubit adder
* Application of a three-qubit adder in Grover's search algorithm to generate all possible $a$ and $b$ for which $a+b=S$, for given $S$
* Quantum $2 \times 2$ sudoku solver

## Decomposition of complex quantum gates
It is possible describe complex quantum gates by circuits composed of basic gates. A decomposition of complex gates is needed when noise
is simulated. Examples implemented in QuantumSim are the multi-qubit controlled Pauli X and Pauli Z gates. 
See this research paper by Tycho de Laat:
[Can complex quantum gates be described using basic quantum gates](https://github.com/nicokuijpers/QuantumSim/blob/main/assets/research/TychoDeLaatComplexQuantumGates.pdf)

Jupyter notebook: [QuantumSimDecomposeMultiControlledXandZ.ipynb](https://github.com/nicokuijpers/QuantumSim/blob/main/QuantumSimDecomposeMultiControlledXandZ.ipynb)

## Simulation of noisy quantum gates
An accurate simulation of quantum noise is implemented as described by Di Bartolomeo et al. 
[Noisy gates for simulating quantum computers](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.5.043210)

Noisy quantum gates are implemented in QuantumSim using this approach. See this research paper by Tycho de Laat:
[Integrating quantum noise into quantum gates for enhanced quantum simulation]
https://github.com/nicokuijpers/QuantumSim/blob/main/assets/research/TychoDeLaatQuantumNoise.pdf

Jupyter notebook: [QuantumSimNoisyGates.ipynb](https://github.com/nicokuijpers/QuantumSim/blob/main/QuantumSimNoisyGates.ipynb)

## Simulation of quantum error correction
Two approaches of quantum error correction (QEC) are implemented in QuantumSim: Peter Shor's nine qubit code and surface codes. 
See this research paper by Michel Meulen:
[Exploratory Investigation of Surface Code Implementation for Fault-Tolerant Quantum Computing](https://github.com/nicokuijpers/QuantumSim/blob/main/assets/research/MichelMeulenQuantumErrorCorrection.pdf)

Jupyter notebook describing Shor's nine qubit code:
[QuantumSimShorNineQubit.ipynb](https://github.com/nicokuijpers/QuantumSim/blob/main/ShorsNineQubitCode.py)

Jupyter notebook describing surface codes:
[QuantumSimSurfaceCodes.ipynb](https://github.com/nicokuijpers/QuantumSim/blob/main/QuantumSimSurfaceCodes.ipynb)

## Visualization of quantum computing with and without noise
In addition to the QuantumSim simulator, we developed a tool to support interactive visualization of quantum algorithms. 
This tool accepts a quantum circuit written in QASM and simulates the circuit with and without noise using Qiskit or
other simulators. See this research paper by Jelle Maas:
[QNEX: An intuitive platform for visulizing and analyzing quantum noise in quantum circuits](https://github.com/Typiqally/qnex/blob/main/paper.pdf)

GitHub repository: https://github.com/Typiqally/qnex

## Getting started
The main purpose of this simulator is to explain the connection between quantum information theory and its implementation in Python code.
Intermediate quantum states are visualised to get insight in the effect of quantum operations on the state of a quantum circuit.
A number of Jupyter notebooks with common quantum algorithms is provided. In addition, the concept of incoherence and quantum noise
is introduced and visualised with animations using Bloch spheres.

Contents of this repository:

| File | Description |
|------|-------------|
| [quantumsim.py](https://github.com/nicokuijpers/QuantumSim/blob/main/quantumsim.py) | Contains the Python code of QuantumSim |
| [QuantumSimIntroduction.ipynb](https://github.com/nicokuijpers/QuantumSim/blob/main/QuantumSimIntroduction.ipynb) | Introduction to QuantumSim |
| [QuantumSimVisualization.ipynb](https://github.com/nicokuijpers/QuantumSim/blob/main/QuantumSimVisualization.ipynb) | Visualisation of intermediate quantum states |
| [QuantumSimMoreOperations.ipynb](https://github.com/nicokuijpers/QuantumSim/blob/main/QuantumSimMoreOperations.ipynb) | More quantum operations with examples |
| [QuantumSimBellStates.ipynb](https://github.com/nicokuijpers/QuantumSim/blob/main/QuantumSimBellStates.ipynb) | Implementation of the four Bell states |
| [QuantumSimUGate.ipynb](https://github.com/nicokuijpers/QuantumSim/blob/main/QuantumSimUGate.ipynb) | U-gate with examples |
| [QuantumSimQubitMeasurement.ipynb](https://github.com/nicokuijpers/QuantumSim/blob/main/QuantumSimQubitMeasurement.ipynb) | Measuring a single qubit |
| [QuantumSimQubitReset.ipynb](https://github.com/nicokuijpers/QuantumSim/blob/main/QuantumSimQubitReset.ipynb) | Resetting a single qubit |
| [QuantumSimTeleportation.ipynb](https://github.com/nicokuijpers/QuantumSim/blob/main/QuantumSimTeleportation.ipynb) | Quantum teleportation |
| [QuantumSimFourierTransform.ipynb](https://github.com/nicokuijpers/QuantumSim/blob/main/QuantumSimFourierTransform.ipynb) | Quantum Fourier Transform (QFT) and inverse QFT |
| [QuantumSimPhaseEstimation.ipynb](https://github.com/nicokuijpers/QuantumSim/blob/main/QuantumSimPhaseEstimation.ipynb) | Quantum Phase Estimation (QPE) |
| [QuantumSimShorAlgorithm.ipynb](https://github.com/nicokuijpers/QuantumSim/blob/main/QuantumSimShorAlgorithm.ipynb) | Shor's Algorithm |
| [QuantumSimGroverAlgorithm.ipynb](https://github.com/nicokuijpers/QuantumSim/blob/main/QuantumSimGroverAlgorithm.ipynb) | Grover's Algorithm |
| [QuantumSimDeutschJozsa.ipynb](https://github.com/nicokuijpers/QuantumSim/blob/main/QuantumSimDeutschJozsa.ipynb) | Deutsch-Jozsa Algorithm |
| [QuantumSimBernsteinVazirani.ipynb](https://github.com/nicokuijpers/QuantumSim/blob/main/QuantumSimBernsteinVazirani.ipynb) | Bernstein-Vazirani Algorithm |
| [QuantumSimCounter.ipynb](https://github.com/nicokuijpers/QuantumSim/blob/main/QuantumSimCounter.ipynb) | Counter circuit to count ones in binary strings |
| [QuantumSimCounterGrover.ipynb](https://github.com/nicokuijpers/QuantumSim/blob/main/QuantumSimCounterGroverSearch.ipynb) | Generating binary strings using Counter and Grover |
| [QuantumSimThreeQubitAdder.ipynb](https://github.com/nicokuijpers/QuantumSim/blob/main/QuantumSimThreeQubitAdder.ipynb) | 3-Qubit adder circuit to compute the sum of two $3$-bit numbers |
| [QuantumSimThreeQubitAdderGrover.ipynb](https://github.com/nicokuijpers/QuantumSim/blob/main/QuantumSimThreeQubitAdderGrover.ipynb) | Finding $a$ and $b$ in $a+b=S$ using 3-qubit adder and Grover |
| [QuantumSimSudokuSolver.ipynb](https://github.com/nicokuijpers/QuantumSim/blob/main/QuantumSimSudokuSolver.ipynb) | 2 x 2 Sudoku solver using Grover's search algorithm |
| [QuantumSimDecomposeMultiControlledXandZ.ipynb](https://github.com/nicokuijpers/QuantumSim/blob/main/QuantumSimDecomposeMultiControlledXandZ.ipynb) | Decomposition of multi-qubit controlled X and Z gates |
| [QuantumSimNoise.ipynb](https://github.com/nicokuijpers/QuantumSim/blob/main/QuantumSimNoise.ipynb) | Incoherence and quantum noise |
| [QuantumSimNoiseBlochSphere.ipynb](https://github.com/nicokuijpers/QuantumSim/blob/main/QuantumSimNoiseBlochSphere.ipynb) | Visualisation of (noisy) circuits using Bloch spheres |
| [QuantumSimNoisyGates.ipynb](https://github.com/nicokuijpers/QuantumSim/blob/main/QuantumSimNoisyGates.ipynb) | Simulation of noisy quantum gates |
| [QuantumSimShorNineQubit.ipynb](https://github.com/nicokuijpers/QuantumSim/blob/main/QuantumSimShorNineQubit.ipynb) | Shor's nine qubit code |
| [QuantumSimSurfaceCodes.ipynb](https://github.com/nicokuijpers/QuantumSim/blob/main/QuantumSimSurfaceCodes.ipynb) | Surface codes |
| [SurfaceCodeQuantumSim.py](https://github.com/nicokuijpers/QuantumSim/blob/main/SurfaceCodeQuantumSim.py) | Contains the Python code for surface code |


[bloch_sphere_animation_fourier.mp4](https://github.com/nicokuijpers/QuantumSim/blob/main/assets/videos/bloch_sphere_animation_fourier.mp4)
is an example animation of (inverse) Quantum Fourier Transform (QFT) with Bloch spheres. Green arrows represent the state of an ideal circuit and red arrows the state of a noisy circuit. The circuit is composed of 5 qubits. The qubits are brought into Fourier state $|\tilde{19}\rangle$ and inverse QFT is applied. 
After measuring, the resulting classical state will the binary represention of $19$ which is $|10011\rangle$.

This code requires QuTiP for the visualisation of Bloch spheres, see https://qutip.org/.

Copyright (c) 2024 Nico Kuijpers

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
