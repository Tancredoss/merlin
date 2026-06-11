# MIT License
#
# Copyright (c) 2025 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import perceval as pcvl
import pytest

from merlin import MeasurementStrategy, QuantumLayer
from merlin.core.computation_space import ComputationSpace


class TestMeasurementStrategyDeprecations:
    def test_probabilities_enum_raises_deprecation_error(self):
        with pytest.raises(
            AttributeError,
            match="v0.4",
        ):
            _ = MeasurementStrategy.PROBABILITIES

    def test_mode_expectations_enum_raises_deprecation_error(self):
        with pytest.raises(
            AttributeError,
            match="v0.4",
        ):
            _ = MeasurementStrategy.MODE_EXPECTATIONS

    def test_amplitudes_enum_raises_deprecation_error(self):
        with pytest.raises(
            AttributeError,
            match="v0.4",
        ):
            _ = MeasurementStrategy.AMPLITUDES

    def test_none_enum_raises_no_error(self):
        _ = MeasurementStrategy.NONE

    def test_deprecation_error_includes_migration_hint(self):
        with pytest.raises(
            AttributeError,
            match="Use MeasurementStrategy.probs",
        ):
            _ = MeasurementStrategy.PROBABILITIES

    def test_deprecated_enum_fails_in_quantum_layer(self):
        circuit = pcvl.Circuit(2)
        with pytest.raises(
            AttributeError,
            match="Use MeasurementStrategy.probs",
        ):
            layer = QuantumLayer(
                input_size=0,
                circuit=circuit,
                input_state=[1, 0],
                computation_space=ComputationSpace.FOCK,
                measurement_strategy=MeasurementStrategy.PROBABILITIES,
            )

        with pytest.raises(
            AttributeError,
            match="Use MeasurementStrategy.amplitudes",
        ):
            layer = QuantumLayer(
                input_size=0,
                circuit=circuit,
                input_state=[1, 0],
                computation_space=ComputationSpace.FOCK,
                measurement_strategy=MeasurementStrategy.AMPLITUDES,
            )

        with pytest.raises(
            AttributeError,
            match="Use MeasurementStrategy.mode_expectations",
        ):
            layer = QuantumLayer(
                input_size=0,
                circuit=circuit,
                input_state=[1, 0],
                computation_space=ComputationSpace.FOCK,
                measurement_strategy=MeasurementStrategy.MODE_EXPECTATIONS,
            )

    def test_deprecated_enum_none_sill_passes_in_quantum_layer(self):
        circuit = pcvl.Circuit(2)
        layer = QuantumLayer(
            input_size=0,
            circuit=circuit,
            input_state=[1, 0],
            measurement_strategy=MeasurementStrategy.NONE,
        )

    def test_probabilities_str_raises_deprecation_error(self):
        circuit = pcvl.Circuit(2)
        with pytest.raises(
            TypeError,
            match="Passing measurement_strategy as a string is no longer supported as of v0.4.",
        ):
            layer = QuantumLayer(
                input_size=0,
                circuit=circuit,
                input_state=[1, 0],
                computation_space=ComputationSpace.FOCK,
                measurement_strategy="PROBABILITIES",
            )

    def test_amplitudes_str_raises_deprecation_error(self):
        circuit = pcvl.Circuit(2)
        with pytest.raises(
            TypeError,
            match="Passing measurement_strategy as a string is no longer supported as of v0.4.",
        ):
            layer = QuantumLayer(
                input_size=0,
                circuit=circuit,
                input_state=[1, 0],
                computation_space=ComputationSpace.FOCK,
                measurement_strategy="AMPLITUDES",
            )

    def test_mode_expectations_str_raises_deprecation_error(self):
        circuit = pcvl.Circuit(2)
        with pytest.raises(
            TypeError,
            match="Passing measurement_strategy as a string is no longer supported as of v0.4.",
        ):
            layer = QuantumLayer(
                input_size=0,
                circuit=circuit,
                input_state=[1, 0],
                computation_space=ComputationSpace.FOCK,
                measurement_strategy="MODE_EXPECTATIONS",
            )

    def test_computation_space_in_constructor_fails(self):
        circuit = pcvl.Circuit(2)
        with pytest.raises(
            AttributeError,
            match="Passing 'computation_space' without an explicit measurement_strategy is no longer supported as of v0.4.",
        ):
            layer = QuantumLayer(
                input_size=0,
                circuit=circuit,
                input_state=[1, 0],
                computation_space=ComputationSpace.FOCK,
            )
        with pytest.raises(
            AttributeError,
            match="Cannot specify 'computation_space' in QuantumLayer's constructor.",
        ):
            layer = QuantumLayer(
                input_size=0,
                circuit=circuit,
                input_state=[1, 0],
                computation_space=ComputationSpace.FOCK,
                measurement_strategy=MeasurementStrategy.probs(),
            )
        with pytest.raises(
            AttributeError,
            match="Cannot specify 'computation_space' in QuantumLayer's constructor.",
        ):
            layer = QuantumLayer(
                input_size=0,
                circuit=circuit,
                input_state=[1, 0],
                computation_space=ComputationSpace.FOCK,
                measurement_strategy=MeasurementStrategy.amplitudes(),
            )

        with pytest.raises(
            AttributeError,
            match="Cannot specify 'computation_space' in QuantumLayer's constructor.",
        ):
            layer = QuantumLayer(
                input_size=0,
                circuit=circuit,
                input_state=[1, 0],
                computation_space=ComputationSpace.FOCK,
                measurement_strategy=MeasurementStrategy.mode_expectations(),
            )

        with pytest.raises(
            AttributeError,
            match="Cannot specify 'computation_space' in QuantumLayer's constructor.",
        ):
            layer = QuantumLayer(
                input_size=0,
                circuit=circuit,
                input_state=[1, 0],
                computation_space=ComputationSpace.FOCK,
                measurement_strategy=MeasurementStrategy.partial(modes=[0]),
            )

        with pytest.raises(
            AttributeError,
            match="Cannot specify 'computation_space' in QuantumLayer's constructor.",
        ):
            layer = QuantumLayer(
                input_size=0,
                circuit=circuit,
                input_state=[1, 0],
                computation_space=ComputationSpace.FOCK,
                measurement_strategy=MeasurementStrategy.NONE,
            )

    def modern_factory_raises_no_error():
        circuit = pcvl.Circuit(2)

        layer = QuantumLayer(
            input_size=0,
            circuit=circuit,
            input_state=[1, 0],
        )

        layer = QuantumLayer(
            input_size=0,
            circuit=circuit,
            input_state=[1, 0],
            measurement_strategy=MeasurementStrategy.NONE,
        )

        # probs
        layer = QuantumLayer(
            input_size=0,
            circuit=circuit,
            input_state=[1, 0],
            measurement_strategy=MeasurementStrategy.probs(),
        )

        layer = QuantumLayer(
            input_size=0,
            circuit=circuit,
            input_state=[1, 0],
            measurement_strategy=MeasurementStrategy.probs(
                computation_space=ComputationSpace.FOCK
            ),
        )

        # amplitudes
        layer = QuantumLayer(
            input_size=0,
            circuit=circuit,
            input_state=[1, 0],
            measurement_strategy=MeasurementStrategy.amplitudes(),
        )

        layer = QuantumLayer(
            input_size=0,
            circuit=circuit,
            input_state=[1, 0],
            measurement_strategy=MeasurementStrategy.amplitudes(
                computation_space=ComputationSpace.FOCK
            ),
        )

        # mode expectation
        layer = QuantumLayer(
            input_size=0,
            circuit=circuit,
            input_state=[1, 0],
            measurement_strategy=MeasurementStrategy.mode_expectations(),
        )

        layer = QuantumLayer(
            input_size=0,
            circuit=circuit,
            input_state=[1, 0],
            measurement_strategy=MeasurementStrategy.mode_expectations(
                computation_space=ComputationSpace.FOCK
            ),
        )

        # Partial
        layer = QuantumLayer(
            input_size=0,
            circuit=circuit,
            input_state=[1, 0],
            measurement_strategy=MeasurementStrategy.partial(modes=[0]),
        )

        layer = QuantumLayer(
            input_size=0,
            circuit=circuit,
            input_state=[1, 0],
            measurement_strategy=MeasurementStrategy.partial(
                modes=[0], computation_space=ComputationSpace.FOCK
            ),
        )
