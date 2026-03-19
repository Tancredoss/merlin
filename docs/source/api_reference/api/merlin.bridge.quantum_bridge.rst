merlin.bridge.quantum_bridge module
===================================
.. automodule:: merlin.bridge.quantum_bridge
   :no-members:

.. currentmodule:: merlin.bridge.quantum_bridge

.. autoclass:: QuantumBridge
   :members:
   :undoc-members:
   :show-inheritance:



Overview
--------

The **QuantumBridge** provides a passive interface between PyTorch-compatible qubit statevector
simulators (PennyLane, Qiskit, custom modules, …) and Merlin's photonic quantum processors. It
encodes computational-basis amplitudes into photonic spaces according to a one-photon-per-group
scheme, enabling hybrid quantum machine learning workflows that combine qubit and photonic paradigms.

Key Features
------------

* **Automatic State Conversion**: Applies a predetermined transition matrix that maps qubit
  statevectors to photonic amplitudes ordered according to the selected computation space
* **Flexible Qubit Grouping**: Supports arbitrary partitioning of qubits into groups for photonic encoding
* **Differentiable Pipeline**: Maintains gradient flow from Merlin back to PennyLane for end-to-end training
* **Batch Processing**: Handles batched inputs efficiently
* **Endianness Control**: Supports both little-endian and big-endian qubit wire ordering
* **Inspectable Mapping**: :meth:`QuantumBridge.qubit_to_fock_state` reveals how individual bitstrings map
  to photonic occupancies for debugging or educational purposes

Architecture
------------

The bridge operates in three stages:

1. **State Preparation**: A PennyLane circuit or function generates a qubit statevector :math:`| \psi \rangle \in \mathbb{C}^{2^n}`
2. **Encoding**: Each computational basis state :math:`| \text{bitstring} \rangle` is mapped to a Fock state with one photon per qubit group
3. **Photonic Processing**: The encoded superposition is fed to a Merlin QuantumLayer for photonic computation

Encoding Scheme
~~~~~~~~~~~~~~~

For a qubit system partitioned into groups of sizes [g₁, g₂, ..., gₖ]:

* Total qubits: n = g₁ + g₂ + ... + gₖ
* Total photonic modes: m = 2^g₁ + 2^g₂ + ... + 2^gₖ
* Total photons: k (one per group)

Each group of gᵢ qubits is encoded as one photon distributed across 2^gᵢ modes in a one-hot fashion.


Parameters
~~~~~~~~~~

See the class signature above for the full constructor, including ``qubit_groups``,
``n_modes``, ``n_photons``, ``computation_space``, ``device``, ``dtype``,
``wires_order``, and ``normalize``.

Helper Functions
----------------

The :meth:`~merlin.bridge.quantum_bridge.QuantumBridge.qubit_to_fock_state`
helper converts a bitstring to the corresponding photonic basic state under the
current grouping convention.

Usage Examples
--------------

Basic Example: Identity Circuit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example demonstrates the bridge with a simple identity circuit, mapping basis states 
through the photonic layer:

.. code-block:: python

   import torch
   import perceval as pcvl
   from merlin import MeasurementStrategy, QuantumLayer
   from merlin.bridge.quantum_bridge import ComputationSpace, QuantumBridge

   # Create a simple identity circuit (m=4 modes, 2 photons)
   circuit = pcvl.Circuit(4)
   merlin_layer = QuantumLayer(
       input_size=0,
       circuit=circuit,
       n_photons=2,
       measurement_strategy=MeasurementStrategy.probs(computation_space=ComputationSpace.UNBUNCHED),
   )

   # Create the bridge
   bridge = QuantumBridge(
       qubit_groups=[1, 1],  # Two 1-qubit groups
       n_modes=4,
       n_photons=2,
       wires_order='little',
       computation_space=ComputationSpace.UNBUNCHED,
       normalize=True,
   )

   # PennyLane-like state preparation module
   class StatePrep(torch.nn.Module):
       def forward(self, _x):
           psi = torch.zeros(4, dtype=torch.complex64)
           psi[0] = 1.0 + 0.0j  # |00⟩ state
           return psi

   state_prep = StatePrep()
   model = torch.nn.Sequential(state_prep, bridge, merlin_layer)

   dummy_input = torch.zeros(1, 1)
   output = model(dummy_input)
   print(f"Output shape: {output.shape}")
   print(f"Output probabilities: {output}")

The bridge returns a complex amplitude tensor in the exact ordering expected by the
``QuantumLayer``.


Hybrid Classification Task
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A complete example showing hybrid qubit-photonic classification:

.. code-block:: python

   import torch
   import torch.nn as nn
   import pennylane as qml
   import perceval as pcvl
   from merlin import MeasurementStrategy, QuantumLayer
   from merlin.bridge.quantum_bridge import ComputationSpace, QuantumBridge

   class HybridQuantumClassifier(nn.Module):
       def __init__(self, n_qubits=3, n_classes=2):
           super().__init__()
           self.n_qubits = n_qubits
           
           # Classical pre-processing
           self.pre_net = nn.Sequential(
               nn.Linear(4, 8),
               nn.ReLU(),
               nn.Linear(8, n_qubits)
           )
           
           # PennyLane quantum circuit
           self.dev = qml.device('default.qubit', wires=n_qubits, shots=None)
           self.weights = nn.Parameter(torch.randn(n_qubits, 3))
           
           # Photonic processor
           m = 2 ** n_qubits  # One large group
           circuit = pcvl.Circuit(m)
           for i in range(m - 1):
               circuit.add(i, pcvl.BS())
           
           self.merlin_layer = QuantumLayer(
               input_size=0,
               circuit=circuit,
               n_photons=1,
               measurement_strategy=MeasurementStrategy.probs(),
           )
           
           # Quantum bridge
           self.bridge = QuantumBridge(
               qubit_groups=[n_qubits],
               n_modes=m,
               n_photons=1,
               computation_space=ComputationSpace.UNBUNCHED,
               normalize=True,
           )
           
           # Classical post-processing
           self.post_net = nn.Linear(m, n_classes)
       
       def _quantum_state(self, x):
           @qml.qnode(self.dev, interface='torch', diff_method='backprop')
           def circuit(weights, features):
               qml.AngleEmbedding(features, wires=range(self.n_qubits))
               for i in range(self.n_qubits):
                   qml.Rot(*weights[i], wires=i)
               for i in range(self.n_qubits - 1):
                   qml.CNOT(wires=[i, i + 1])
               return qml.state()
           
           # Handle batching
           if x.ndim > 1:
               states = []
               for i in range(x.shape[0]):
                   states.append(circuit(self.weights, x[i]))
               return torch.stack(states)
           return circuit(self.weights, x)
       
       def forward(self, x):
           # Classical preprocessing
           features = self.pre_net(x)
           psi = self._quantum_state(features)
           payload = self.bridge(psi)
           distribution = self.merlin_layer(payload)
           logits = self.post_net(distribution)
           return logits

   # Usage
   model = HybridQuantumClassifier(n_qubits=3, n_classes=2)
   inputs = torch.randn(8, 4)  # Batch of 8 samples
   outputs = model(inputs)
   print(f"Classification outputs: {outputs.shape}")


Notes and Best Practices
-------------------------

Design Considerations
~~~~~~~~~~~~~~~~~~~~~

* **No Trainable Mapping**: The bridge itself contains no trainable parameters. All variational 
  behavior should be implemented in the PennyLane circuit.

* **Mode Requirements**: The Merlin layer must be configured with m = Σ 2^group_size modes and 
  n_photons = len(qubit_groups). No ancilla or post-selected modes are supported.

* **Differentiability**: The bridge uses Merlin's tensor superposition path to maintain gradient 
  flow. Always call with `apply_sampling=False` (handled internally).

Performance Tips
~~~~~~~~~~~~~~~~

* Use `normalize=False` if your PennyLane circuit already outputs normalized states
* For large qubit systems, consider using multiple smaller groups rather than one large group
* Batch multiple samples together for better GPU utilization

See Also
--------

* :class:`merlin.QuantumLayer`: The underlying photonic processor interface
* `PennyLane Documentation <https://pennylane.ai>`_: For quantum circuit design
* `Perceval Documentation <https://perceval.quandela.net>`_: For photonic circuit details
