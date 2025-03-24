/*
Phasma: fast sparse linear algebra for Python.
https://github.com/MarcelFerrari/phasma

File: bindings.cpp
Description: Main file for binding Phasma to Python using NanoBind.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

#include <nanobind/nanobind.h>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "phasma/bindings/bindings.hpp"

NB_MODULE(phasma, m) {
    // Initialize Types module
    Phasma::bindings::init_types_module(m);

    // Initialize Sparse Matrix module
    Phasma::bindings::init_sparse_matrix_module(m);

    // Init Scaler module
    Phasma::bindings::init_scaler_module(m);

    // Initialize direct solvers module
    auto m_direct = m.def_submodule("direct");
    Phasma::bindings::init_direct_solver_module(m_direct);

    // Initialize iterative solvers module
    auto m_iterative = m.def_submodule("iterative");
    Phasma::bindings::init_iterative_solver_module(m_iterative);

    // Initialize matrix-free solvers module
    auto m_matfree = m.def_submodule("matfree");
    Phasma::bindings::init_matfree_solver_module(m_matfree);
}