/*
Phasma: fast sparse linear algebra for Python.
https://github.com/MarcelFerrari/phasma

File: sparse_matrix_module.cpp
Description: Exposes Eigen's SparseMatrix class to Python using Nanobind.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

// Nanobind
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>

#include "phasma/types.hpp"
#include "phasma/bindings/sparse_matrix_module.hpp"

namespace nb = nanobind;
using namespace nb::literals;
namespace Phasma::bindings {
void init_sparse_matrix_module(nb::module_ & m){
    Phasma::bindings::bind_sparse_matrix<double, Phasma::ColMajor>(m, "CCSMatrix");
    Phasma::bindings::bind_sparse_matrix<double, Phasma::RowMajor>(m, "CRSMatrix");
    Phasma::bindings::bind_sparse_matrix<float,  Phasma::ColMajor>(m, "CCSMatrix_f");
    Phasma::bindings::bind_sparse_matrix<float,  Phasma::RowMajor>(m, "CRSMatrix_f");
    Phasma::bindings::bind_sparse_matrix<Phasma::float128,  Phasma::ColMajor>(m, "CCSMatrix_ld");
    Phasma::bindings::bind_sparse_matrix<Phasma::float128,  Phasma::RowMajor>(m, "CRSMatrix_ld");
}
} // namespace Phasma::bindings