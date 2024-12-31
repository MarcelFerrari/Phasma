/*
Phasma: fast sparse linear algebra for Python.
https://github.com/MarcelFerrari/phasma

File: bindings.hpp
Description: Forward declarations for the bindings module.
             This is done to reduce compile time.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

See LICENSE file in the project root for full license information.
*/

#ifndef PHASMA_BINDINGS_HPP
#define PHASMA_BINDINGS_HPP

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace Phasma::bindings {
    void init_sparse_matrix_module(nb::module_ &m);
    void init_scaler_module(nb::module_ &m);
    void init_direct_solver_module(nb::module_ &m);
}

#endif // PHASMA_BINDINGS_HPP