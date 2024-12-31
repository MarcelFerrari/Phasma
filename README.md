# Phasma

**Phasma** is a minimalistic Python library for fast sparse linear algebra, built on top of the Eigen C++ library. Designed for high performance and simplicity, Phasma exposes Eigen's sparse matrix classes and provides compatibility wrappers to integrate other software with Eigen Types, enabling seamless access from Python.

## Features

- **Built on Eigen**: Access Eigen's mature and efficient sparse matrix classes directly from Python.
- **Compatibility Wrappers**: Integrates with other software, exposing them to Eigen Types for streamlined workflows.
- **Minimalistic Design**: Provides a small, well-optimized set of operations that emphasize performance best practices.
- **Python Bindings**: Simplifies the use of advanced sparse linear algebra operations in Python.

## Philosophy

Phasma prioritizes:
- **Performance**: Every operation is optimized for speed and efficiency.
- **Simplicity**: Avoids redundant methods, providing a single, fast way to perform each operation.
- **Best Practices**: Encourages users to adapt to high-performance patterns by design.

## Building Phasma

Building Phasma from source is straight forward:

```
git clone https://github.com/MarcelFerrari/Phasma.git
cd Phasma
mkdir build
cd build
cmake ..
make install
```
This will compile the source code and generate a shared object file in `/lib`.

## Getting Started

To import Phasma in your Python script, it will be sufficient to set your `$PYTHONPATH` environment variable to included Phasma's `/lib` folder. E.g.:

```
export PYTHONPATH=$PYTHONPATH:$(realpath Phasma/lib)
```

Here's an example of how to use Phasma to solve a sparse linear system:

```python
import numpy as np
import phasma as ph

# Assemble matrix in COO format
# Numpy arrays:
# i_idx = i indices
# j_idx = j indices
# vals = values
i_idx, j_idx, vals = assemble_matrix()

# Create CCS matrix
A = ph.CCSMatrix(i_idx, j_idx, vals)

# Perform LU factorization using Eigen's SparseLU
# Use full column/row scaling as preconditioning strategy
solver = ph.SparseLU(ph.ScalingType.Full)
solver.compute(A)

# Solve sparse linear system
rhs = ...
x = solver.solve(rhs)
```

## Documentation

Coming soon

## Copyright notice

Copyright &copy; 2024 Marcel Ferrari

## License

Phasma is licensed under the [Mozilla Public License 2.0](https://www.mozilla.org/MPL/2.0/).

## Acknowledgments

Phasma is powered by the Eigen library and inspired by the need for fast, reliable sparse linear algebra in Python. We also thank the open-source community for providing tools and libraries that make projects like this possible.
