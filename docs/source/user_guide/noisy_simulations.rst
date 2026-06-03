=============================
Noisy simulations with SLOS
=============================

Introduction
=============

The current quantum architecture are in the NISQ (noisy intermediate scale quantum) regime, making the computation noisy. That means that your current MerLin simulations, which are theoretically correct, are not going to give the same results as a run on the current photonic devices. 

This can create a problem as models trained on simulator will not have good performance on the hardware since the output will be modified. The solution is to have tunable noisy simulations to train your models locally and can perform well on the hardware.

Since MerLin 0.4, noise was added to the SLOS background to allow this type of training to eventually run on hardware.


Add Noise to the SLOS simulation
============================================

To add noise to your :class:`~merlin.algorithms.layer.QuantumLayer` simulation, the :class:`pcvl.NoiseModel` class needs to be used. It defines the value of each of the 7 noise source. You can either add this noise model to a :class:`pcvl.Experiment` that is then used at the initialization of the :class:`~merlin.algorithms.layer.QuantumLayer` or you can directly pass this noise model to the :class:`~merlin.algorithms.layer.QuantumLayer`'s ``noise`` parameter in the constructor. Here are a couple of examples

.. code-block:: python

    import perceval as pcvl
    import torch
    import merlin as ML

    noise=pcvl.NoiseModel(
            brightness=0.1,
            indistinguishability=0.2,
            g2=0.3,
            g2_distinguishable=False,
            transmittance=0.4,
            phase_imprecision=0.5,
            phase_error=0.6,
        ),

    circuit = pcvl.Circuit(3)
    circuit.add((0, 1), pcvl.BS())
    circuit.add(0, pcvl.PS(pcvl.P("px")))
    circuit.add((1, 2), pcvl.BS())
    
    # Option 1: define the noise model with an experiment
    experiment = pcvl.Experiment(circuit, noise=noise)

    layer = ML.QuantumLayer(
        input_size=1,
        experiment=experiment,
        input_parameters=["px"],
        input_state=[1, 1, 1],
        computation_space=ML.ComputationSpace.FOCK  # Fock space used for noisy simulations
    )

    x = torch.rand(3, 1)
    probs = layer(x)

    # Option 2: define the noise model with the noise parameter
    layer = ML.QuantumLayer(
        input_size=1,
        experiment=experiment,
        input_parameters=["px"],
        input_state=[1, 1, 1],
        computation_space=ML.ComputationSpace.FOCK,  # Fock space used for noisy simulations
        noise=noise
    )

    x = torch.rand(3, 1)
    probs = layer(x)



Noise Types on Quandela's Quantum Computers
============================================

There is 7 different noises split in 3 different categories on Quandela's quantum computer. We will explain here each one of them and their impact on the quantum computations.

-----------------------
Post-Measurement Noise
-----------------------

These noises only affect the probabilities of measurement at the end of the interferometer. 

1. Brightness and 2. Transmittance
-----------------------------------

The brightness noise identifies the probability that the photon source actually emits when it is triggered. The value of this parameter is bounded between ``0.0`` and ``1.0``. It is directly interpreted to be the probability that the photon source emits the photon. The default value is ``1.0`` as it is the perfect case where the source always emits photons. The brightness can be defined in the ``NoiseModel`` with the ``brightness`` parameter.

The transmittance is directly is the probability that the photon is transmitted through the whole interferometer without being ejected. Because it is a probability, it is also bounded between ``0.0`` and ``1.0``. The default value is ``1.0`` as it is the perfect case where no photon is lost. The transmittance can be defined in the ``NoiseModel`` with the ``transmittance`` parameter.

The noise affect the output probabilities by inducing a photon survival probability. So, in other words, the probability that a single photon will be emitted and transmitted is defined by the product of the brightness and transmittance. So, the output probabilities or amplitudes of the quantum layer are not normalized. **TODO Complete When I understand the work exacly, add an example**


-----------------------
Circuit Noise
-----------------------

These noises affect the precision of the operations of the quantum layer.

3. Phase Imprecision
-----------------------------------

This noise type reflects the maximum precision of the phase shifters of the interferometer in radians. By default, this parameter, ``phase_imprecision`` in the ``NoiseModel``, is set to ``0``for infinite precision. **TODO: add the concrete implementation transformation for angles too precise with an example**

4. Phase Error
-----------------------------------

This noise type reflects the maximum random noise applied to the phase shifters of the interferometer in radians. By default, this parameter, ``phase_error`` in the ``NoiseModel``, is set to ``0``for the noiseless case. **TODO: add the concrete implementation transformation for angles too precise with an example**


-----------------------
Source Noise
-----------------------

These noises describe the imperfections of the photon emitter (source).

5. Indistinguishability
-----------------------------------

This noise describes the probability that the photon emitters generate photons that are indistinguishable form one another. In the perfect case, all the photons are indistiguishable to be able to observe intrication effects. Indeed, intrication is one of the two main quantum phenomena that are the foundation of quantum computing. To see the impact of indistiguishability on intrication, a simple beam-splitter that has a 50:50 reflection/transmittance rate  is necessary:

.. image:: ../_static/img/simple_bs.png
   :alt: Simple beam-splitter

We will then use the :math:`\ket{1,1}` input state (one photon per mode) in the Fock basis. If the two photons are indistiguishable, by the Hong-Ou-Mendel (HOM) effect, the output state should be :math:`\frac{1}{\sqrt{2}}\bigg(\ket{2,0} + \ket{0,2} \bigg)`. From a probabilities stand point, that means that the two equiprobable outputs are measuring both photons in the first mode or both in the second mode. We can observe this phenomena with the following code.

.. code-block:: python

    import perceval as pcvl
    import merlin as ml

    #Creating the BS circuit
    circuit = pcvl.Circuit(2)
    circuit.add([0, 1], pcvl.BS.H())

    #Running the circuit
    layer = ml.QuantumLayer(
        input_size=0,
        circuit=circuit,
        input_state=[1, 1],
        measurement_strategy=ml.MeasurementStrategy.probs(
            computation_space=ml.ComputationSpace.FOCK
        ),
    )
    output = layer()

    #Printing the probabilities
    for key, prob in zip(layer.output_keys, output.flatten()):
        print(f"Output probability of state {key} is {prob}")

Output:
- Output probability of state (2, 0) is 0.49999991059303284
- Output probability of state (1, 1) is 0.0
- Output probability of state (0, 2) is 0.49999991059303284

This is caused by intrication because, classically, if each photon has a 50% chance to be reflected or transmitted, the output probabilities should be the following:
- The photons are a 25% chance of of both being measured in the first mode.
- The photons are a 25% chance of of both being measured in the second mode.
- The photons are a 50% chance of of both being measured in different modes.

The intrication phenomena between two indistiguishable photon (they interact between one another) is the cause behind the discrepancy.

With completely distinguishable photons, we find the same expected classical distribution as distinguishable photons don't interact with one another. It can be observed in the following code:

.. code-block:: python

    import perceval as pcvl
    import merlin as ml

    #Creating the BS circuit
    circuit = pcvl.Circuit(2)
    circuit.add([0, 1], pcvl.BS.H())

    #Running the circuit
    layer = ml.QuantumLayer(
        input_size=0,
        circuit=circuit,
        input_state=[1, 1],
        measurement_strategy=ml.MeasurementStrategy.probs(
            computation_space=ml.ComputationSpace.FOCK
        ),
        noise=pcvl.NoiseModel(indistinguishability=1.0), #Completely distinguishable photons
    )
    output = layer()

    #Printing the probabilities
    for key, prob in zip(layer.output_keys, output.flatten()):
        print(f"Output probability of state {key} is {prob}")

Output:
- Output probability of state (2, 0) is 0.25
- Output probability of state (1, 1) is 0.5
- Output probability of state (0, 2) is 0.25

The default value of the ``indistinguishability`` parameter of a the ``NoiseModel`` is 1.0, as in the perfect case all photons are indistinguishable.



g2_distinguishable parameter
----------------------------------------------
The ``g2_distinguishable`` parameter in the noise model is a boolean that identifies if the photons generated by g2 emissions (multi-photon emissions) are distinguishable or not. By default, in Perceval, this parameter is ``True``. In MerLin's QuantumLayer, the parameter is considered ``False`` if it can be ignored (indistinguishability=1.0 or g2=0.0: the default value of these noise sources). So, even if this parameter is set to True, which is the case with Perceval's :class:`pcvl.NoiseModel`'s object, if there is not a simulation with g2 emissions and indistinguishable photons, the ``g2_distinguishable`` parameter will be set to ``False`` in the :class:`~merlin.algorithms.layer.QuantumLayer`. If ``indistinguishability=1.0`` and ``g2>0.0``, a warning will indicate that ``g2_distinguishable`` is set to ``False``, otherwise, since the parameter does not have an impact on the simulation, the switch is done silently. 

Run a noisy simulation
----------------------------------------------

Here 

1. Circuit noise

   a. ``phase_imprecision``: Maximum precision of the phase shifters (0 means infinite precision).
   b. ``phase_error``: Maximum random noise on the phase shifters (in radian). The default value (noiseless case) is 0.

2. Source noise

   a. ``indistinguishability``: Chance two photons are indistinguishable. The default value (noiseless case) is 1.
   b. ``g2``: :math:`g^2(0)` - second order intensity autocorrelation at zero time delay. This parameter is correlated with how often two photons are emitted by the source instead of a single one. The default value (noiseless case) is 0.
   c. ``g2_distinguishable``: g2-generated photons indistinguishability. This parameter can not be False if ``indistinguishability=1.0`` and ``g2>0.0``. The default value (noiseless case) is True.

3. Post-measurement noise

   a. ``brightness``: First lens brightness of a quantum dot. The default value (noiseless case) is 1.
   b. ``transmittance``: System-wide transmittance (warning, can interfere with the brightness parameter). The default value (noiseless case) is 1.


Noisy Simulations Guidelines
----------------------------------------------

For noisy simulations, there are a couple of rules that need to be followed:

1. All noisy simulations must be run with the probabilities measurement strategy.
2. Noisy simulations cannot use ``return_object=True``.
3. Noisy simulations with source noise must be run in the Fock computation space. If a different space is chosen, it will be changed automatically with a warning.
4. Noisy simulation with ``g2>0`` cannot use a grouping strategy. Indeed, since this noise creates input states with more photons than expected, multiple photon sectors are explored. The fock spaces explored are m modes and n_photons to 2*n_photons that all have different space dimensions. To still apply a grouping strategy, you can iterate over the :class:`~merlin.core.sectored_distribution.SectorResult` objects of the :class:`~merlin.core.sectored_distribution.SectoredDistribution` and apply one grouping per sector.


**Useful**: :math:`\sum_i 2^{k_i}`