:github_url: https://github.com/merlinquantum/merlin

=================
Reproduced Papers
=================

MerLin enables researchers to reproduce and build upon published quantum machine learning research.
This section provides implementations of key papers in the quantum ML field, complete with working code, analysis, and extensions.

Overview
--------

Each reproduction may include:

* **Original paper implementation** - Faithful recreation of the paper's methodology
* **Reproduction status** - Indicating whether the reproduction is partial or complete
* **Jupyter notebooks** - Interactive exploration of results and concepts
* **Full code** - Available on GitHub for easy access and modification
* **Performance analysis** - Comparison with paper results
* **Extension opportunities** - Ideas for building upon the work

.. note::
   All reproductions are implemented using MerLin's high-level API, making them accessible to ML practitioners without deep quantum expertise.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reproduced Papers

   reproductions/fock_state_expressivity
   reproductions/quantum_reservoir_computing
   reproductions/qllm_finetuning
   reproductions/photonic_qcnn
   reproductions/photonic_kernel
   reproductions/QCNN_data_classification
   reproductions/qssl
   reproductions/photonic_memristor
   reproductions/hqnn-myth
   reproductions/data_reuploading
   reproductions/distributed_nn
   reproductions/quantum_adversarial_ml
   reproductions/quantum_transfer_learning
   reproductions/nearest_centroids_merlin
   reproductions/template

Available Reproductions
-----------------------

The reproductions are organized by topic. Each card opens the corresponding paper-reproduction page.

Kernel Methods
~~~~~~~~~~~~~~

.. merlin-gallery::
   :data: _data/galleries/reproduced_papers/reproduced_papers_kernel_methods.json
   :columns: 3
   :contour-color: #5648ED

For a Better Understanding of Photonic QML Theory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. merlin-gallery::
   :data: _data/galleries/reproduced_papers/reproduced_papers_variational_methods.json
   :columns: 3
   :contour-color: #5648ED

Computer Vision
~~~~~~~~~~~~~~~

.. merlin-gallery::
   :data: _data/galleries/reproduced_papers/reproduced_papers_computer_vision.json
   :columns: 3
   :contour-color: #5648ED

Advanced Training Paradigms
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. merlin-gallery::
   :data: _data/galleries/reproduced_papers/reproduced_papers_advanced_training.json
   :columns: 3
   :contour-color: #5648ED

Distributed Training
~~~~~~~~~~~~~~~~~~~~

.. merlin-gallery::
   :data: _data/galleries/reproduced_papers/reproduced_papers_distributed_training.json
   :columns: 2
   :contour-color: #5648ED

Future-proofing
~~~~~~~~~~~~~~~

.. merlin-gallery::
   :data: _data/galleries/reproduced_papers/reproduced_papers_future_proofing.json
   :columns: 2
   :contour-color: #5648ED

Contributing Reproductions
--------------------------

We welcome contributions of additional paper reproductions.

**Requirements**:

* High-impact quantum ML papers (>50 citations preferred)
* Photonic/optical quantum computing focus
* Implementable with current MerLin features
* Clear experimental validation

**Submission Process**:

1. **Propose** the paper in our `GitHub Discussions <https://github.com/merlinquantum/merlin/discussions>`_
2. **Implement** using MerLin following our guidelines
3. **Validate** results against original paper
4. **Document** in Jupyter notebook format
5. **Submit** via pull request a complete reproduction folder and a summary page in :code:`docs/source/reproductions/` directory

**Template Structure**:

.. code-block:: text

   paper_reproduction/
   ├── README.md             # Paper overview and results
   ├── implementation.py     # Core implementation
   ├── notebook.ipynb        # Interactive exploration showing the key concepts, not necessarily the full implementation
   ├── data/                 # Datasets and preprocessing
   ├── results/              # Figures and analysis
   └── tests/                # Validation tests

**Template Summary Page**: :doc:`this document <reproductions/template>`

Recognition
-----------

Contributors to reproductions are recognized in:

* Paper reproduction documentation
* MerLin project contributors list
* Academic citations in MerLin publications

*Have a paper you'd like to see reproduced?* `Start a discussion <https://github.com/merlinquantum/merlin/discussions/new>`_.
