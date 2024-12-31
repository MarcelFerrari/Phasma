/*
Phasma: fast sparse linear algebra for Python.
https://github.com/MarcelFerrari/phasma

File: sparse_matrix_module.cpp
Description: Exposes Eigen's SparseMatrix class to Python using Nanobind.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

See LICENSE file in the project root for full license information.
*/

// Nanobind
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "phasma/bindings/sparse_matrix_module.hpp"

namespace nb = nanobind;
using namespace nb::literals;
namespace Phasma::bindings {
void init_sparse_matrix_module(nb::module_ & m){
    Phasma::bindings::bind_sparse_matrix<double, Eigen::ColMajor>(m, "CCSMatrix");
    Phasma::bindings::bind_sparse_matrix<double, Eigen::RowMajor>(m, "CRSMatrix");
    Phasma::bindings::bind_sparse_matrix<float,  Eigen::ColMajor>(m, "CCSMatrix_f");
    Phasma::bindings::bind_sparse_matrix<float,  Eigen::RowMajor>(m, "CRSMatrix_f");
}
} // namespace Phasma::bindings