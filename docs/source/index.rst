:github_url: https://github.com/merlinquantum/merlin

MerLin - Photonic Quantum Machine Learning Framework
====================================================

**MerLin - Photonic Quantum Machine Learning in PyTorch**

MerLin enables researchers to build, train, and benchmark photonic quantum machine learning models using familiar ML tools.

By integrating photonic circuit simulation with PyTorch and scikit-learn, MerLin allows hybrid quantum-classical models to be developed and evaluated within standard machine learning workflows.

**Explore MerLin**

.. merlin-gallery::
   :data: _data/galleries/welcome_intro_cards_3.json
   :columns: 3

What you can do with MerLin
---------------------------

Start simple, then scale: MerLin lets you prototype your first quantum layer and
progress to reproducible, benchmark-ready QML experiments.

With MerLin you can:

- ⚡ Build photonic quantum layers directly in PyTorch.
- 🔬 Benchmark quantum models on real machine learning datasets.
- 🔁 Reproduce state-of-the-art QML papers.
- 🧠 Combine classical neural networks with quantum circuits.
- 🧪 Experiment with kernels, reservoirs, QNNs, and generative models.
- 🖥 Run simulations locally or execute circuits on photonic hardware.

MerLin is designed as a discovery engine for photonic and hybrid quantum machine
learning, with reproducible experimentation across models, datasets, and hardware
constraints.

.. toctree::
   :maxdepth: 3
   :hidden:

   Welcome <self>
   Quickstart <quickstart/index>
   user_guide/index
   quantum_expert_area/index
   examples/index
   notebooks/index
   reproduced_papers/index
   api_reference/index
   performance/index
   QML_library/index

Reproducing State-of-the-art QML Papers
=======================================

A core goal of MerLin is to make quantum machine learning research more reproducible, comparable, and hardware-aware. As the field grows, ensuring that results can be independently reproduced and evaluated across consistent benchmarks becomes increasingly important.
While many QML papers provide valuable algorithmic insights, reproducing published results can still be challenging due to differences in hardware and software stacks, data preprocessing pipelines, experimental settings, or hardware assumptions.
MerLin addresses this by providing a unified framework and a growing catalog of reproduced QML experiments, implemented within a consistent environment and designed to reflect realistic quantum hardware constraints.
These reproductions help:

1. Validate MerLin implementations against published results.
2. Provide reusable baselines and reference implementations for new research.
3. Enable systematic benchmarking across models, datasets, and encodings.
4. Study how algorithms interact with hardware constraints and photonic architectures, supporting algorithm–hardware co-design.

**Our Reproduced Benchmarks**

.. merlin-gallery::
   :data: _data/galleries/welcome_reproduction_cards.json
   :columns: 4
   :contour-color: #5648ED

Who is MerLin For?
==================

Choose your path based on your background and goals:

**Machine Learning Engineers**
  New to quantum? Start with the :doc:`quickstart/index` for a hands-on introduction.
  No quantum background required—MerLin abstracts the complexity so you focus on
  building and training models. See :doc:`examples/index` for common patterns.

**Quantum Researchers**
  Explore :doc:`quantum_expert_area/index` for detailed documentation on photonic
  circuits, gate operations, and hardware integration. MerLin provides flexible APIs
  for designing custom quantum layers and debugging quantum behavior.

**Paper Reproducers**
  Start with :doc:`reproduced_papers/index` to access reference implementations of
  published QML papers. Compare your results, adapt methods, and build on proven
  baselines. See :doc:`performance/index` for benchmarking and hardware constraints.

**Curious Explorers**
  Browse :doc:`notebooks/index` for practical tutorials, or dive into
  :doc:`user_guide/index` for comprehensive guidance on workflows and patterns.

Installation
============

.. code-block:: bash

   pip install merlinquantum

This installs MerLin and core dependencies including PyTorch, Perceval, NumPy,
Pandas, and scikit-learn.

Verify the installation:

.. code-block:: python

   import merlin as ML
   print(ML.__version__)

Minimal Example
===============

Train a photonic quantum layer inside a PyTorch workflow:

.. code-block:: python

   import torch
   import numpy as np
   import merlin as ML
   from sklearn.datasets import make_circles
   from sklearn.model_selection import train_test_split

   # Dataset
   X, y = make_circles(n_samples=400)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # Normalize using training data statistics (prevent data leakage)
   min_vals = X_train.min(axis=0, keepdims=True)
   max_vals = X_train.max(axis=0, keepdims=True)
   X_train = (X_train - min_vals) / np.clip(max_vals - min_vals, a_min=1e-6, a_max=None)
   X_test = (X_test - min_vals) / np.clip(max_vals - min_vals, a_min=1e-6, a_max=None)

   X_train = torch.tensor(X_train, dtype=torch.float32)
   y_train = torch.tensor(y_train, dtype=torch.long)

   # Quantum layer
   quantum_layer = ML.QuantumLayer.simple(
       input_size=2,
       output_size=2,
   )

   # Training loop
   optimizer = torch.optim.Adam(quantum_layer.parameters(), lr=0.01)
   criterion = torch.nn.CrossEntropyLoss()

   for _ in range(100):
       optimizer.zero_grad()
       logits = quantum_layer(X_train)
       loss = criterion(logits, y_train)
       loss.backward()
       optimizer.step()

This example:

1. Builds a photonic circuit.
2. Wraps it as a PyTorch module.
3. Trains it with standard ML tooling.

**Kickstart Your Journey**

.. merlin-gallery::
   :data: _data/galleries/welcome_next_steps_cards.json
   :columns: 3

The MerLin Model
================

In MerLin, a quantum circuit behaves like a regular PyTorch layer, so you can plug
it into familiar pipelines and train end-to-end with gradient descent.

.. code-block:: python

   model = torch.nn.Sequential(
       torch.nn.Linear(10, 2),
       ML.QuantumLayer.simple(input_size=2),
       torch.nn.Linear(2, 2),
   )

This hybrid architecture can combine:

- **Classical preprocessing**: Encode raw features into quantum-friendly dimensions
- **Quantum processing**: Let the photonic circuit learn quantum-specific patterns
- **Classical output**: Map quantum results to predictions

Common patterns include quantum kernels, variational quantum classifiers (VQC), and
quantum feature maps. Explore :doc:`examples/index` and :doc:`notebooks/index`
for working implementations and use cases.

Next Steps
==========

- **Get started**: :doc:`quickstart/index`
- **Understand MerLin**: :doc:`user_guide/index`
- **Explore examples**: :doc:`examples/index` or :doc:`notebooks/index`
- **Reproduce papers**: :doc:`reproduced_papers/index`
- **API reference**: :doc:`api_reference/index`
