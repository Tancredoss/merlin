#!/usr/bin/env python
import torch
import merlin as ML

def update_rule(state: torch.Tensor, output) -> torch.Tensor:
    # Handle different output types
    if isinstance(output, torch.Tensor):
        output_tensor = output
    elif isinstance(output, ML.core.ProbabilityDistribution):
        output_tensor = output.tensor
    elif isinstance(output, ML.core.StateVector):
        output_tensor = output.tensor
    elif isinstance(output, ML.core.PartialMeasurement):
        output_tensor = output.tensor
    else:
        output_tensor = output
    
    return state + output_tensor[:, 0]

layer = ML.algorithms.QuantumLayer(
    builder=lambda: (
        b := ML.CircuitBuilder(n_modes=3),
        b.add_entangling_layer(trainable=True, name="U1"),
        b.add_memristive_ps(
            mode=1,
            update_rule=update_rule,
            initial_state=0.25,
            name="mem",
            num_backprop_steps=2,
        ),
        b.add_angle_encoding(modes=[0, 2], name="input"),
        b.add_entangling_layer(trainable=True, name="U2"),
        b
    )[-1],
    input_size=2,
    input_state=[1, 1, 0],
    measurement_strategy=ML.MeasurementStrategy.probs(
        computation_space=ML.ComputationSpace.FOCK
    ),
)
layer.reset(batch_size=1)

# Run 5 forwards
inputs = [torch.randn(1, 2, requires_grad=True) for _ in range(5)]
outputs = [layer(input_batch) for input_batch in inputs]

# Check history
print(f"History length: {len(layer.memristive_history[0])}")

# Compute loss and backward
loss = outputs[-1][:, 0].sum()
loss.backward()

# Check gradient norms
grad_norms = [
    0.0 if inp.grad is None else inp.grad.abs().max().item()
    for inp in inputs
]
print(f"Gradient norms: {grad_norms}")
print(f"Expected: inputs 2, 3, 4 should have non-zero gradients")
