:github_url: https://github.com/merlinquantum/merlin

==============================================
Noisy Simulations
==============================================

To run your first noisy simulation, consult the :doc:`/user_guide/noisy_simulations.rst` page to understand the different noise types and run your first noisy :class:`~merlin.algorithms.layer.QuantumLayer`.

----------------------------------------------
Noisy Simulation implementation
----------------------------------------------

In this section, we will discuss the genral implementation details of the different noise calculations inside of SLOS.

Brightness and transmittance
----------------------------------------------

These two noises are implemented in the same workflow. As mentionned in this page :doc:`/user_guide/noisy_simulations.rst`, the survival probability of the photons are defined by the product of these two noises. This survival probability is then used to **TODO complete once I understand the end of the pipeline**.


Noisy Simulations Limitations
----------------------------------------------

Noisy simulations are significantly less efficient than ideal ones. You can profile the memory requirements of noisy simulations with source noise using the benchmark script: :file:`../../benchmarks/benchmark_noisy_slos_cache_memory.py`.

Memory and computational complexity grow significantly with the number of modes and photons. For example, a 5-photon 2-mode circuit requires around 200 MB, while a 20-mode 3-photon experiment requires around 3 GB. To profile memory consumption in your specific use case, run the benchmark script with:

.. code-block:: bash

    python benchmarks/benchmark_noisy_slos_cache_memory.py --modes 6 7 8 9 --photons 1 2 3 4 5 --backward

Here is an example of the output graph of this run.

.. figure:: images/benchmark_noisy_slos_cache_memory.png
   :align: center
   :width: 600px
   :alt: Memory need for the QuantumLayer with distinguishable photons per output size