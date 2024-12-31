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
    // Initialize Sparse Matrix module
    Phasma::bindings::init_sparse_matrix_module(m);

    // Init Scaler module
    Phasma::bindings::init_scaler_module(m);

    // Initialize direct solvers module
    Phasma::bindings::init_direct_solver_module(m);
}