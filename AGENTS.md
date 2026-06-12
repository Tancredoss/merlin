# AGENTS.md

Repository instructions for coding agents working on Merlin.

## General Rules

- Be direct and concise.
- Read the surrounding code before changing behavior.
- Keep changes scoped to the user request.
- Do not rewrite unrelated code.
- Do not add abstractions unless they remove real complexity.
- Use clear, descriptive variable and function names.
- Code should be self-explanatory. Add inline comments only when they explain non-obvious intent.

## Maintainability

- Write code for future maintainers, not only for the current task.
- Prefer simple, explicit control flow over clever or compact code.
- Keep functions and methods focused on one responsibility.
- Preserve existing public behavior unless the change explicitly requires otherwise.
- Use existing project patterns before introducing new ones.
- Avoid hidden state, implicit side effects, and surprising mutation.
- Make invalid states impossible when reasonable, and raise clear errors when they occur.
- Do not duplicate logic. Extract a helper only when it improves readability or reduces real repetition.
- Keep type annotations accurate and update them when behavior changes.
- Keep tests close to the behavior they protect.

## Failure Policy

- Do not hide failures.
- Do not add silent fallbacks during development.
- Do not catch broad exceptions to continue execution unless the caller explicitly expects that behavior.
- Do not convert errors into warnings when the operation cannot produce correct results.
- If something fails, let it fail clearly so the root cause can be fixed.
- Use `# TODO` only when explicitly leaving known unfinished behavior visible.

## Warnings

- Warnings are acceptable for deprecations, compatibility notices, or genuinely recoverable behavior.
- Warnings must not replace exceptions for invalid inputs, corrupted state, missing required dependencies, failed computation, or unsupported behavior.
- If new warning behavior is added, add or update tests that assert the warning is emitted.

## Tests

- When a test fails, determine the root cause before changing code.
- Do not skip, xfail, weaken, or delete tests just to make the suite pass.
- Do not silence failing tests by broadening assertions.
- If a test expectation is wrong, explain why in the change.
- Run the relevant tests after code changes whenever possible.
- If tests cannot be run, state that clearly.

## Documentation And Docstrings

All public classes, methods, functions, aliases, and data types must have:

- a typed function signature;
- a complete NumPy-style docstring;
- parameters, returns, raises, and defaults documented in the project format.

Documentation is required even when the implementation is explicit. Clear code does not replace public API documentation.

In-code documentation is required as well:

- Add module docstrings when a module defines public behavior or non-trivial internal machinery.
- Add docstrings to internal helpers when their purpose, inputs, outputs, assumptions, or failure modes are not immediately obvious.
- Document important invariants, shape conventions, dtype expectations, units, and numerical assumptions close to the code that depends on them.
- Document algorithmic choices when a future maintainer would need context to change them safely.
- Keep inline comments focused on intent, constraints, and edge cases. Do not narrate obvious statements.

Use this docstring structure:

```python
def func(x: dtype1, y: QuantumLayer | None = None) -> dtype3:
    """Description of the function.

    Parameters
    ----------
    x : dtype1
        Description of the first parameter respecting the
        indent
    y : QuantumLayer|None
        Description of an optional parameter. If omitted, description.
        Default value is 0.

    Returns
    -------
    dtype3
        Description of the return value.

    Raises
    ------
    ImportError
        Description of the error.
    """
```

Docstring rules:

- Do not write default values in the dtype field.
- Do not write `optional` next to the dtype.
- Use `Type | None` for nullable values.
- Do not wrap dtypes or return types in double backticks.
- Use the object name directly for Merlin types when Sphinx can resolve it.
- Use full object paths when Sphinx cannot resolve a type, for example `merlin.core.state_vector.StateVector`.
- Use full import paths for external types, for example `numpy.ndarray`, `pandas.DataFrame`, or `torch.Tensor`.
- Use explicit Sphinx references only when normal type resolution does not work.
- New public APIs must be added to the relevant files under `docs/source/api_reference`.
- Do not rely on `automodule` to generate the full API automatically.
- Every new public object must be declared once in the corresponding API `.rst` file.
- Every file under `docs/source` must appear in at least one `toctree`.

## Sphinx

For documentation-affecting changes, verify with:

```bash
SPHINXOPTS="-W --keep-going -n" make -C docs clean html
```

The docs should build without warnings or errors.

Do not suppress Sphinx warnings unless the warning is understood and the suppression is intentional.
