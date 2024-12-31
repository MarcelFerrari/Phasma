/*
Phasma: fast sparse linear algebra for Python.
https://github.com/MarcelFerrari/phasma

File: direct_solvers_module.hpp
Description: Exposes direct solvers to Python using Nanobind.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

See LICENSE file in the project root for full license information.
*/

#ifndef PHASMA_DIRECT_SOLVER_MODULE_HPP
#define PHASMA_DIRECT_SOLVER_MODULE_HPP

// Nanobind
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>

// Eigen
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/OrderingMethods>

// Phasma
#include "phasma/types.hpp"
#include "phasma/scaler.hpp"
#include "phasma/scaled_direct_solver.hpp"

// Eigen 3 built-in direct solvers
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>

// Pardiso
#ifdef PHASMA_PARDISO_SUPPORT
#include <Eigen/PardisoSupport>
#endif

// SuiteSparse
#ifdef PHASMA_SUITESPARSE_SUPPORT
#include <Eigen/UmfPackSupport>
#include <Eigen/KLUSupport>
#endif

namespace nb = nanobind;
using namespace nb::literals;

namespace Phasma::bindings {

template <typename Solver, typename Scalar, int Order = Eigen::ColMajor>
void bind_direct_solver(nb::module_& m, const std::string& class_name) {
    using ScaledDirectSolver = Phasma::ScaledDirectSolver<Solver, Scalar, Order>;
    using SparseMatrix = typename ScaledDirectSolver::SparseMatrix;
    using Vector = typename ScaledDirectSolver::Vector;

    nb::class_<ScaledDirectSolver>(m, class_name.c_str())
        .def(nb::init<Phasma::ScalingType>(), "type"_a = Phasma::ScalingType::NONE, "Create a DirectSolver object with the given scaling type.")
        .def("analyze_pattern", &ScaledDirectSolver::analyze_pattern, "A"_a, "Analyze the sparsity pattern of the matrix A.")
        .def("factorize", &ScaledDirectSolver::factorize, "A"_a, "Factorize the matrix A.")
        .def("compute", &ScaledDirectSolver::compute, "A"_a, "Analyze the sparsity pattern and factorize the matrix A.")
        .def("solve", &ScaledDirectSolver::solve, "b"_a, "Solve the linear system Ax = b.")
        .def("solve", &ScaledDirectSolver::factorize_and_solve, "A"_a, "b"_a, "Factorize the matrix A and solve the linear system Ax = b.");
}

} // namespace Phasma::bindings

#endif // PHASMA_DIRECT_SOLVER_MODULE_HPP