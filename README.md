# Phasma

**Phasma** is the high-performance C++ linear algebra backend of the [Pyroclast](https://github.com/MarcelFerrari/Pyroclast) Geophysics Solver. Built on top of the Eigen C++ library, Phasma exposes fast and efficient sparse matrix operations to Python through minimal, high-quality bindings.

Designed specifically to support the demanding requirements of large-scale geodynamic simulations, Phasma is lightweight, modular, and focused on performance-critical sparse linear algebra tasks.

## Features

- **Backend for Pyroclast**: Phasma powers the sparse linear algebra computations in Pyroclast, enabling fast and scalable geophysical simulations.
- **Built on Eigen**: Leverages Eigen’s mature and highly optimized sparse matrix solvers.
- **Clean Python Bindings**: Exposes Eigen types and solvers to Python via a minimal interface.
- **Compatibility Wrappers**: Allows integration of additional C++ solvers and modules using Eigen's data structures.
- **Minimalistic Design**: Offers a lean set of operations that emphasize speed, clarity, and good performance practices.

## Philosophy

Phasma is built around the following principles:

- **Performance First**: Every component is tuned for speed and efficient memory usage.
- **Simplicity by Design**: A small, clear API that avoids unnecessary abstraction or duplication.
- **Interoperability**: Facilitates clean integration between Python, C++, and other HPC components within the Pyroclast ecosystem.

## Building Phasma

To build Phasma from source:

```bash
git clone https://github.com/MarcelFerrari/Phasma.git
cd Phasma
mkdir build
cd build
cmake ..
make install
```

This will compile the core C++ code and install the shared Python extension into `/lib`.

## Getting Started

To use Phasma in Python, add the `/lib` folder to your `$PYTHONPATH`:

```bash
export PYTHONPATH=$PYTHONPATH:$(realpath Phasma/lib)
```

### Example: Solving a Sparse Linear System

```python
import numpy as np
import phasma as ph

# Assemble a sparse matrix in COO format
i_idx, j_idx, vals = assemble_matrix()

# Convert to compressed column storage (CCS)
A = ph.CCSMatrix(i_idx, j_idx, vals)

# Use Eigen's SparseLU solver
solver = ph.SparseLU(ph.ScalingType.Full)
solver.compute(A)

# Solve Ax = b
rhs = ...
x = solver.solve(rhs)
```

## Documentation

Documentation is in progress and will be published soon.

## License

Phasma is licensed under the [Mozilla Public License 2.0](https://www.mozilla.org/MPL/2.0/).

## Acknowledgments

Phasma is built on the Eigen C++ library and is developed as part of the Pyroclast project. Special thanks to the open-source community for creating the tools that make this possible.