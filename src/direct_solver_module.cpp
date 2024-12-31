/*
Phasma: fast sparse linear algebra for Python.
https://github.com/MarcelFerrari/phasma

File: direct_solver_module.cpp
Description: Direct solver module for binding Phasma to Python using NanoBind.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

See LICENSE file in the project root for full license information.
*/

#include <nanobind/nanobind.h>

#include <Eigen/Core>
#include <Eigen/Sparse>

// C++ Standard Library
#include <string>
#include <iostream>
#include <optional>

#include "phasma/types.hpp"
#include "phasma/bindings/direct_solver_module.hpp"

namespace nb = nanobind;
namespace Phasma::bindings{
void init_direct_solver_module(nb::module_& m) {
    // Initialize Eigen 3 direct solvers
    // Most use ColMajor format so we create a convenient alias
    using SparseMatrix = Phasma::SparseMatrix<double>;

    // SparseLU
    Phasma::bindings::bind_direct_solver<Eigen::SparseLU<SparseMatrix>, double>(m, "SparseLU");

    // SparseQR
    Phasma::bindings::bind_direct_solver<Eigen::SparseQR<SparseMatrix, Eigen::COLAMDOrdering<Phasma::Index>>, double>(m, "SparseQR");
    
    // PardisoLU
    #ifdef PHASMA_PARDISO_SUPPORT
    #pragma message("Pardiso support is enabled.")
    Phasma::bindings::bind_direct_solver<Eigen::PardisoLU<SparseMatrix>, double>(m, "PardisoLU");
    #endif

    // SuiteSparse
    #ifdef PHASMA_SUITESPARSE_SUPPORT
    #pragma message("SuiteSparse support is enabled.")
    Phasma::bindings::bind_direct_solver<Eigen::UmfPackLU<SparseMatrix>, double>(m, "UmfPackLU");
    Phasma::bindings::bind_direct_solver<Eigen::KLU<SparseMatrix>, double>(m, "KLU");
    #endif
}
} // namespace Phasma::bindings