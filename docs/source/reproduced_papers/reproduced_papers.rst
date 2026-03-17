:github_url: https://github.com/merlinquantum/merlin

=================
Reproduced Papers
=================

MerLin provides reproducible implementations of published quantum machine learning papers.
Each card links to a dedicated reproduction page with paper metadata, implementation details, code access, and results.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reproduced Papers

   reproductions/fock_state_expressivity
   reproductions/amplitude_limitations
   reproductions/nearest_centroids
   reproductions/quantum_reservoir_computing
   reproductions/quantum_adversarial_ml
   reproductions/quantum_transfer_learning
   reproductions/qllm_finetuning
   reproductions/photonic_QGAN
   reproductions/photonic_qcnn
   reproductions/photonic_kernel
   reproductions/QCNN_data_classification
   reproductions/qssl
   reproductions/photonic_memristor
   reproductions/hqnn-myth
   reproductions/data_reuploading
   reproductions/distributed_nn
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


Sequential Tasks
~~~~~~~~~~~~~~~~

.. merlin-gallery::
   :data: _data/galleries/reproduced_papers/reproduced_papers_sequential.json
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
5. **Submit** via pull request a complete reproduction folder and a summary page in :code:`docs/source/reproduced_papers/reproductions/` directory

**Mandatory Structure for a Reproduction**:

.. code-block:: text

   papers/NAME/            # Non-ambiguous acronym or fullname of the reproduced paper
   ├── .gitignore          # specific .gitignore rules for clean repository
   ├── notebook.ipynb      # Interactive exploration of key concepts
   ├── README.md           # Paper overview and results overview
   ├── requirements.txt    # additional requirements for the scripts
   ├── configs/            # defaults + CLI/runtime descriptors consumed by the repo root runner
   ├── lib/                # code used by the shared runner and notebooks - as an integrated library (import shared data helpers from papers/shared/<paper>/)
   ├── models/             # Trained models
   ├── results/            # Selected generated figures, tables, or outputs from trained models
   ├── tests/              # Validation tests
   └── utils/              # additional commandline utilities for visualization, launch of multiple trainings, etc...

**Template Summary Page**: :doc:`this document <reproductions/template>`

Recognition
-----------

Contributors to reproductions are recognized in:

* Paper reproduction documentation
* MerLin project contributors list
* Academic citations in MerLin publications

*Have a paper you'd like to see reproduced?* `Start a discussion <https://github.com/merlinquantum/merlin/discussions/new>`_.
