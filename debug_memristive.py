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

# Build layer
builder = ML.CircuitBuilder(n_modes=3)
builder.add_entangling_layer(trainable=True, name="U1")
builder.add_memristive_ps(
    mode=1,
    update_rule=update_rule,
    initial_state=0.5,
    name="memPS",
    num_backprop_steps=1,
)
builder.add_angle_encoding(modes=[0, 2], name="input")
builder.add_entangling_layer(trainable=True, name="U2")

layer = ML.QuantumLayer(
    builder=builder,
    input_size=2,
    n_photons=2,
    measurement_strategy=ML.MeasurementStrategy.probs(
        computation_space=ML.ComputationSpace.FOCK
    ),
    return_object=True,
)
layer.reset(batch_size=2)

print(f"Initial memristive_state length: {len(layer.memristive_state)}")
print(f"Initial memristive_history[0] length: {len(layer.memristive_history[0])}")
print(f"Initial memristive_history[0]: {layer.memristive_history[0]}")

input_data = torch.randn(2, 2, requires_grad=True)

# First forward
output1 = layer(input_data)
print(f"\nAfter forward 1:")
print(f"memristive_history[0] length: {len(layer.memristive_history[0])}")
print(f"memristive_history[0]: {layer.memristive_history[0]}")

# Second forward
output2 = layer(input_data.detach())
print(f"\nAfter forward 2:")
print(f"memristive_history[0] length: {len(layer.memristive_history[0])}")
print(f"memristive_history[0]: {layer.memristive_history[0]}")
