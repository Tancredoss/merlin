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

**Algorithm**

1. Compute

   .. math::

      l = \sum_{i=0}^{n} \dim(\mathcal{F}_{m,i})

   where :math:`\mathcal{F}_{m,i}` denotes the Fock space of
   ``m`` modes and ``i`` photons.

2. Initialize the transition matrix as a tensor of zeros with shape

   .. math::

      (\dim(\mathcal{F}_{m,n}), l)

3. For each basis state in the ``(m, n)`` Fock space:

   * For each possible output state:

     1. Compute the probability

        .. math::

           \binom{n}{n_{\mathrm{survived}}}
           (b t)^{n_{\mathrm{survived}}}
           (1-b t)^{n-n_{\mathrm{survived}}}

        where:

        * :math:`n_{\mathrm{survived}}` is the number of photons that
          survive,
        * :math:`b` is the source brightness,
        * :math:`t` is the transmittance.

     2. Assign the probability to the appropriate column of the
        transition matrix using the output state's corresponding index
        and key.

4. Return the completed transition matrix.

The possible output states are all of the possible combinations of losing photons in the basis state.

For g2 simulations, the photon loss algorithm is applied per n photon photon sector at the output of the simulation. Indeed, the transition matrix is different per sector. After this noise is applied, the probabilities are then reclassified and returned as a big tensor.

Phase Error and Imprecision
----------------------------------------------

**TODO, complete when implemented** 

Indistinguishability
----------------------------------------------

This noise is implemented on the one-bad-bit (OBB) principle first implemented in Merlin in this `reproduced paper <https://github.com/merlinquantum/reproduced_papers/blob/main/papers/photonic_quantum_enhanced_kernels/utils/noise.py>`_. Indeed, since distinguishable photons can be tracked independently, each of those independant photons can be simulated on their own. This implementation is done in the :class:`~merlin.pcvl_pytorch.noisy_slos.NoisySLOSComputeGraph` class. Here are the main steps:

**Algorithm**

1. Create a
   :class:`~merlin.pcvl_pytorch.slos_torchscript.SLOSComputeGraph`
   object for ``m`` modes and photon numbers ranging from ``1`` to ``n``,
   where ``n`` is the number of photons in the input state.

2. For each possible configuration of distinguishable photons:

   * Run a single-photon SLOS simulation for each distinguishable
     photon.

   * Run a SLOS simulation for all remaining photons, treating them as
     indistinguishable.

   * Convolve the resulting output distributions.

   * Multiply the convolved distribution by

     .. math::

        p^{\,n-n_{\mathrm{dist}}}
        (1-p)^{\,n_{\mathrm{dist}}}

     where:

     * :math:`p` is the photon indistinguishability,
     * :math:`n` is the number of photons in the input state,
     * :math:`n_{\mathrm{dist}}` is the number of distinguishable
       photons.

3. Sum the weighted distributions into the output tensor.

4. Return the resulting output tensor.

The combinations of possible distinguishable photons are simply the ways of choosing 0 to n photons from the input state.


g2 and g2 distinguishable
----------------------------------------------

These noise build on the :class:`~merlin.pcvl_pytorch.noisy_slos.NoisySLOSComputeGraph` class to create the :class:`~merlin.pcvl_pytorch.noisy_slos.NoisyG2SLOSComputeGraph` object. Indeed, the noisy simulation needs to be done for multiple input states with photon duplication. Here is the main implementation steps.

**Algorithm**

1. Create a :class:`~merlin.pcvl_pytorch.noisy_slos.NoisySLOSComputeGraph`
   object for ``m`` modes and ``i`` photons, with ``i`` ranging from ``n``
   to ``2n``.

2. If ``g2_distinguishable`` is ``True``, create a
   :class:`~merlin.pcvl_pytorch.slos_torchscript.SLOSComputeGraph`
   object for ``m`` modes and a single photon.

3. Generate all possible photon-addition vectors. We want the vectors of all possibles sequences of possible duplicated photons.

   For example, for the input state ``[1,1,0]``:

   * ``[[0,0,0]]`` — no duplicated photons.
   * ``[[1,0,0], [0,1,0]]`` — one duplicated photon.
   * ``[[1,1,0]]`` — both photons duplicated.

   Group the vectors according to the number of added photons.

4. For each group corresponding to ``i`` added photons:

   * For each photon-addition vector:

     * If ``g2_distinguishable`` is ``True``:

       1. Run the original input state on the
          :class:`~merlin.pcvl_pytorch.slos_torchscript.SLOSComputeGraph`
          with ``n`` photons.

       2. Run the single-photon SLOS computation for each added photon.

       3. Convolve the resulting output distributions.

     * Otherwise:

       1. Run the augmented input state on the
          :class:`~merlin.pcvl_pytorch.slos_torchscript.SLOSComputeGraph`
          with ``n+i`` photons.

     * Multiply the resulting distribution by

       .. math::

          p^{i}(1-p)^{n-i}

       where ``p`` is the probability that two photons are emitted and
       ``n`` is the photon number of the desired input state.

   * Combine all distributions into the tensor representing the
     ``n_i`` photon sector.

5. Return the probability distribution for each photon sector.

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