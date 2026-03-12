:github_url: https://github.com/merlinquantum/merlin

======================================================================
Quantum Convolutional Neural Network for Classical Data Classification
======================================================================

.. admonition:: Paper Information
   :class: note

   **Title**: Quantum convolutional neural network for classical data classification

   **Authors**: Tak Hur, Leeseok Kim, Daniel K. Park

   **Published**: Quantum Machine Intelligence, Volume 4, Number 1, Page 3 (2022)

   **DOI**: `10.1007/s42484-021-00061-x <https://doi.org/10.1007/s42484-021-00061-x>`_

   **Paper URL**: `arXiv:2108.00661 <https://arxiv.org/abs/2108.00661>`_

   **Original Repository**: `takh04/QCNN <https://github.com/takh04/QCNN>`_

   **Reproduction Status**: ✅ Complete

   **Reproducer**: Cassandre Notton (cassandre.notton@quandela.com)

Project Repository
==================

.. merlin-gallery::
   :data: _data/galleries/reproduced_papers/qcnn_data_classification_external_links.json
   :columns: 3
   :contour-color: #5648ED

Abstract
========

This reproduction studies a quantum convolutional neural network (QCNN) pipeline for binary classical-image classification.
The approach compresses MNIST-family inputs with PCA and applies a quantum pseudo-convolution stage before a classification head.

The reproduced setup includes two complementary workflows: a Merlin/Perceval implementation built with ``QuantumLayer`` and wrappers around the authors' original TensorFlow/Keras QCNN code for benchmarking.
The experiments compare quantum and classical pseudo-convolution variants under matched receptive-field settings.

Significance
============

The paper demonstrates that QCNN-style architectures can be adapted to classical data classification and can outperform simple classical baselines in selected settings.
In this reproduction, the same idea is explored in a photonic workflow with explicit parameter/accuracy trade-offs across kernel modes, kernel sizes, strides, and number of kernels.

MerLin Implementation
=====================

The MerLin side of the project is executed through the repo-level CLI with ``--paper QCNN_data_classification`` and supports two model families:

* ``--model qconv``: quantum pseudo-convolution model (parallel kernels)
* ``--model single``: single Gaussian interferometer baseline

Core run options include quantum kernel topology (``nb_kernels``, ``kernel_size``, ``kernel_modes``, ``stride``), encoding mode (angle or amplitude), and optimisation controls (``steps``, ``batch``, ``seeds``, ``lr``).

.. figure:: ../../_static/reproduced_papers/QCNN_data_classification/Photonic_QConv.png
   :alt: Photonic pseudo-convolution architecture used in QCNN reproduction
   :align: center
   :width: 85%

   Photonic pseudo-convolution structure used by the MerLin QCNN reproduction.

Key Contributions Reproduced
============================

**Hybrid quantum/classical benchmark pipeline**
  * Reproduced the MNIST/FashionMNIST 0-vs-1 workflow with PCA-compressed inputs.
  * Exposed matched quantum and classical pseudo-convolution baselines for direct comparison.

**Hyperparameter and efficiency analysis**
  * Performed sweeps over kernel modes, number of kernels, kernel size, and stride.
  * Reported both accuracy and parameter efficiency to identify Pareto-optimal settings.

**Multiple encoding regimes**
  * Evaluated angle-encoding and amplitude-encoding variants.
  * Measured robustness across PCA dimensions (8 and 16 components).

Implementation Details
======================

Main execution examples (from the reproduction README):

.. code-block:: bash

   # Show all options
   python ../implementation.py --paper QCNN_data_classification --help

   # Quantum pseudo-convolution with classical comparison
   python ../implementation.py --paper QCNN_data_classification \
     --dataset mnist --pca_dim 8 --steps 200 --seeds 3 \
     --nb_kernels 4 --kernel_size 2 --kernel_modes 8 --compare_classical

   # Single-GI baseline
   python ../implementation.py --paper QCNN_data_classification \
     --model single --n_modes 8 --n_features 8 --n_photons 4 \
     --steps 200 --seeds 3

Datasets are resolved via the shared data root and stored under ``data/QCNN_data_classification/`` by default.

Experimental Results
====================

Hyperparameter analysis (MNIST, PCA=8)
--------------------------------------

Highlights from the sweep summary:

* Kernel modes have the strongest correlation with accuracy (about 0.65).
* Increasing modes from 4 to 16 improves mean accuracy from about 0.58 to 0.85, with parameter growth from roughly 90 to 1,000.
* Three kernels outperform one kernel on average (about 0.81 vs 0.68 mean accuracy) at higher parameter cost.
* Stride 2 slightly outperforms stride 1 while reducing overlap-related parameter cost.

.. list-table::
   :widths: 50 50

   * - .. figure:: ../../_static/reproduced_papers/QCNN_data_classification/hyperparameter_impacts.png
          :alt: Hyperparameter impacts on QCNN metrics
          :width: 100%

     - .. figure:: ../../_static/reproduced_papers/QCNN_data_classification/correlation_matrix.png
          :alt: Correlation matrix of QCNN hyperparameters and metrics
          :width: 100%

   * - .. figure:: ../../_static/reproduced_papers/QCNN_data_classification/quantum_vs_classical.png
          :alt: Quantum versus classical comparison
          :width: 100%

     - .. figure:: ../../_static/reproduced_papers/QCNN_data_classification/pareto_frontier.png
          :alt: Pareto frontier for accuracy and parameter efficiency
          :width: 100%

A compact heatmap view of sweep outcomes:

.. figure:: ../../_static/reproduced_papers/QCNN_data_classification/heatmaps.png
   :alt: QCNN sweep heatmaps
   :align: center
   :width: 95%

Angle-encoding benchmark (3 kernels, kernel size 3, stride 2)
-------------------------------------------------------------

.. list-table:: Validation accuracy (mean ± std)
   :header-rows: 1
   :widths: 34 33 33

   * - Model
     - 8 PCA components
     - 16 PCA components
   * - Quantum convolution (830 trainable parameters) on MNIST
     - 96.08 ± 3.64
     - 80.11 ± 23.29
   * - Quantum convolution (830 trainable parameters) on FashionMNIST
     - 93.18 ± 1.20
     - 82.75 ± 19.07
   * - Classical convolution (32 trainable parameters) on MNIST
     - 76.78 ± 11.16
     - 72.84 ± 15.04
   * - Classical convolution (32 trainable parameters) on FashionMNIST
     - 81.35 ± 6.38
     - 76.85 ± 23.14

Amplitude-encoding benchmark (6 kernel modes, 3 kernels, kernel size 3, stride 2)
-----------------------------------------------------------------------------------

.. list-table:: Validation accuracy (mean ± std)
   :header-rows: 1
   :widths: 34 33 33

   * - Model
     - 8 PCA components
     - 16 PCA components
   * - Quantum convolution (128/176 trainable parameters) on MNIST
     - 73.51 ± 14.07
     - 66.89 ± 12.08
   * - Quantum convolution (128/176 trainable parameters) on FashionMNIST
     - 71.48 ± 12.53
     - 74.13 ± 21.65

Training curves
===============

.. list-table:: MNIST and FashionMNIST (8 PCA components)
   :widths: 50 50

   * - .. figure:: ../../_static/reproduced_papers/QCNN_data_classification/MNIST_accuracy_curves.png
          :alt: MNIST accuracy curves for QCNN reproduction
          :width: 100%

     - .. figure:: ../../_static/reproduced_papers/QCNN_data_classification/MNIST_loss_curves.png
          :alt: MNIST loss curves for QCNN reproduction
          :width: 100%

   * - .. figure:: ../../_static/reproduced_papers/QCNN_data_classification/FMNIST-accuracy_curves.png
          :alt: FashionMNIST accuracy curves for QCNN reproduction
          :width: 100%

     - .. figure:: ../../_static/reproduced_papers/QCNN_data_classification/FMNIST-loss_curves.png
          :alt: FashionMNIST loss curves for QCNN reproduction
          :width: 100%

Performance Analysis
====================

**Advantages**

* The quantum pseudo-convolution reaches higher peak accuracies than the matched classical baseline in the reported angle-encoding setup.
* Hyperparameter sweeps provide actionable operating points balancing accuracy and parameter budget.

**Current limitations**

* Variance increases in some 16-PCA settings, indicating sensitivity to configuration and seed choice.
* Higher-performing quantum settings typically require substantially more trainable parameters than compact classical baselines.

Citation
========

.. code-block:: bibtex

   @article{hur2022quantum,
     title={Quantum convolutional neural network for classical data classification},
     author={Hur, Tak and Kim, Leeseok and Park, Daniel K.},
     journal={Quantum Machine Intelligence},
     volume={4},
     number={1},
     pages={3},
     year={2022},
     publisher={Springer},
     doi={10.1007/s42484-021-00061-x}
   }

----
