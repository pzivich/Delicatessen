# Contributing to delicatessen

Thank you for taking the time to contribute to `delicatessen`. The following is a set of guidelines
for contributing to `delicatessen`. These are meant to be general guidelines and not a strict set of
rules. Therefore, use your best judgement or ask if you are unsure.

Below are a few ways you can contribute. If you are contributing code, please be sure to review the
code guidelines below.

-----------------------------

## How can I contribute?

### Reporting bugs

If you come across a bug or other issue, consider opening an issue. When opening an issue, please 
provide the full requested information. A minimal replicable example will also be of a benefit. This
will allow us to design a test to be placed in `tests/` to cover these potential errors.

### Suggesting enhancements

Suggesting ways `delicatessen` could be enhanced is also something we appreciate. In particular, if
there is an estimating equation you think would be valuable to add support for, please consider 
opening an issue with the feature request template. Providing a source that describes the estimating
equation (or gradient / score function for parametric models) will also be a large help.

### Review available documentation

If reviewing documentation and you come across typos, mistakes, or unclear information, we would 
appreciate calling our attention to these areas. If you are inclined, you can also open a pull request
that corrects these mistakes.

### Providing examples

Providing use cases or demonstrations of application of `delicatessen` is another way to contribute.
Of greatest interest are scenarios where multiple estimating equations (can be pre-built or custom) are
stacked together. These examples more readily highlight the benefits of M-Estimation.

Examples can be suggested via an issue, or can be directly added in `docs/`. Please be aware that
any provided examples would need the data source to be freely available. However, we may consider 
simulated data for more complex applications of `delicatessen` if data is not or cannot be made 
freely available.

-----------------------------

## Code Guidelines

When proposing updates to the code, please be sure that the proposed code is well documented. By well
documented, we mean that there the code is clearly annotated with comments, indicating what each line
is doing. 

Furthermore, readability should be prioritized. A single line should do up to two major calculations or
function calls at most. When nesting calls, break each call into a clean line. This will allow adequate
space for comments on what the function call is meant to do.

`delicatessen` is meant to be rely on as few dependencies as possible. Currently, only NumPy and 
SciPy are needed. If your code requires additional dependencies, it is more likely to be rejected.

Finally, please adhere to the Python Enhancement Proposals (PEP) guidance when contributing code.

### Estimating equation contributions

Regarding contribution of an estimating equations, please be sure your contribution includes the 
following items:
 - Documentation of the estimating equation, including the mathematical expression and a source
 - Example of function call in the documentation
 - Relevant tests in `tests/`
