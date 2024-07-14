# QuantumSim
This is a quantum computer simulator in Python with several Jupyter notebooks to illustrate possible usage.

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
| [QuantumSimFourierTransform.ipynb](https://github.com/nicokuijpers/QuantumSim/blob/main/QuantumSimFourierTransform.ipynb) | Quantum Fourier Transform (QFT) and inverse QFT |
| [QuantumSimShorAlgorithm.ipynb](https://github.com/nicokuijpers/QuantumSim/blob/main/QuantumSimShorAlgorithm.ipynb) | Shor's Algorithm |
| [QuantumSimGroverAlgorithm.ipynb](https://github.com/nicokuijpers/QuantumSim/blob/main/QuantumSimGroverAlgorithm.ipynb) | Grover's Algorithm |
| [QuantumSimNoise.ipynb](https://github.com/nicokuijpers/QuantumSim/blob/main/QuantumSimNoise.ipynb) | Incoherence and quantum noise |
| [QuantumSimNoiseBlochSphere.ipynb](https://github.com/nicokuijpers/QuantumSim/blob/main/QuantumSimNoiseBlochSphere.ipynb) | Visualisation of (noisy) circuits using Bloch spheres |

[bloch_sphere_animation_fourier.mp4](https://github.com/nicokuijpers/QuantumSim/blob/main/bloch_sphere_animation_fourier.mp4)
is an example animation of (inverse) Quantum Fourier Transform (QFT) with Bloch spheres. Green arrows represent the state of an ideal circuit and red arrows the state of a noisy circuit. The circuit is composed of 5 qubits. The qubits are brought into Fourier state $|\tilde{19}\rangle$ and inverse QFT is applied. 
After measuring, the resulting classical state will the binary represention of $19$ which is $|10011\rangle$.

This code requires QuTiP for the visualisation of Bloch spheres, see https://qutip.org/.

Copyright (c) 2024 Nico Kuijpers

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
