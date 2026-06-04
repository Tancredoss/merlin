:github_url: https://github.com/merlinquantum/merlin

==============================================
Noisy Simulations
==============================================

To run your first noisy simulation, consult the :doc:`/user_guide/noisy_simulations.rst` page to understand the different noise types and run your first noisy :class:`~merlin.algorithms.layer.QuantumLayer`.

----------------------------------------------
Noisy Simulation implementation
----------------------------------------------

In this section, we will discuss the general implementation details of the different noise calculations inside of SLOS.

Brightness and Transmittance
----------------------------------------------

These two noises are implemented in the same workflow. As mentioned in this page :doc:`/user_guide/noisy_simulations.rst`, the survival probability of the photons are defined by the product of these two noises. This survival probability is then used to create a transition matrix for the end probabilities of the interferometer. Indeed, the whole simulation of the pipeline is done without any consideration for these noises and generate a tensor of dimension (batch size, n and m Fock space), where n is the number of photons and m is the number of modes. We then apply the brightness and transmittance noise to these results. First we need to compute the transition matrix like so:

1. Define l as the sum of the Fock space dimensions of m modes and 0 to n photons.
2. Define the transition matrix as a tensor full of zeros of dimension (n and m Fock space, l)
3. For every basis state in the n, m Fock space:
    a. For each possible output state, compute the associated probabilities: :math:`\binom{n}{n_{survived}}(b\cdot t)^{n_{survived}}(1-b\cdot t)^{n-n_{survived}}` where :math:`n_{survived}` photons survived in the basis state, :math:`b` is the brightness and :math:`t` is the transmittance.
    b. Those probabilities are then assigned to the corrected column index  and correct keys associated to the basis state

The possible output states are all of the possible combinations of losing photons in the basis state.

For g2 simulations, the photon loss algorithm is applied per n photon photon sector at the output of the simulation. Indeed, the transition matrix is different per sector. After this noise is applied, the probabilities are then reclassified and returned as a big tensor.

Phase Error and Imprecision
----------------------------------------------

**TODO, complete when implemented** 

Indistinguishability
----------------------------------------------

This noise is implemented on the one-bad-bit (OBB) principle first implemented in Merlin in this `reproduced paper <https://github.com/merlinquantum/reproduced_papers/blob/main/papers/photonic_quantum_enhanced_kernels/utils/noise.py>`_. Indeed, since distinguishable photons can be tracked independently, each of those independant photons can be simulated on their own. This implementation is done in the :class:`~merlin.pcvl_pytorch.noisy_slos.NoisySLOSComputeGraph` class. Here are the main steps:

1. Create a :class:`~merlin.pcvl_pytorch.slos_torchscript.SLOSComputeGraph` object for m modes and 1 to n photons. n photons is the number of photons in the input state.
2. For each combination of possible distinguishable photons:
    a. Run a 1 photon SLOS for each of the distinguishable photons.
    b. Run a SLOS simulation for all of the remaining photons since they are indistinguishable.
    c. Convolve the output distributions of these runs and multiply it by its corresponding probability: :math:`p^{n-n_{dist}}(1- p)^{n_{dist}}` where :math:`p` is the indistinguishability, :math:`n` is the number of photons in the input state and :math:`n_{survived}` is the number of distinguishable photons.
3. Add all of those probabilities into the output tensor

The combinations of possible distinguishable photons are simply the ways of choosing 0 to n photons from the input state.


g2 and g2 distinguishable
----------------------------------------------

These params

----------------------------------------------
Noisy Simulations Limitations
----------------------------------------------

Noisy simulations are significantly less efficient than ideal ones. You can profile the memory requirements of noisy simulations with source noise using the benchmark script: :file:`../../benchmarks/benchmark_noisy_slos_cache_memory.py`.

Memory and computational complexity grow significantly with the number of modes and photons. For example, a 5-photon 2-mode circuit requires around 200 MB, while a 20-mode 3-photon experiment requires around 3 GB. This is only with indistinguishable photons. The memory need is even bigger for g2 simulations. To profile memory consumption in your specific use case, run the benchmark script with:

.. code-block:: bash

    python benchmarks/benchmark_noisy_slos_cache_memory.py --modes 6 7 8 9 --photons 1 2 3 4 5 --backward

Here is an example of the output graph of this run.

.. figure:: images/benchmark_noisy_slos_cache_memory.png
   :align: center
   :width: 600px
   :alt: Memory need for the QuantumLayer with distinguishable photons per output size


Also, as a reminder, here are the guidelines of the noisy simulations.
1. All noisy simulations must be run with the probabilities measurement strategy. Indeed, we can only change the probabilities as computing the actual amplitude after the noise breaks the current implementation of SLOS simulations: density matrix representation would be needed instead of just vector states.
2. Noisy simulations cannot use ``return_object=True``. It will be implemented in a future version.
3. Noisy simulations with source noise must be run in the Fock computation space. If a different space is chosen, it will be changed automatically with a warning. Indeed, since the noises can remove or add photons, remaining in a constrained space may remove some of the ffects of the noise.