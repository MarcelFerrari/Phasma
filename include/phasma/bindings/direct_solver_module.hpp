/*
Phasma: fast sparse linear algebra for Python.
https://github.com/MarcelFerrari/phasma

File: direct_solvers_module.hpp
Description: Exposes direct solvers to Python using Nanobind.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

#ifndef PHASMA_DIRECT_SOLVER_MODULE_HPP
#define PHASMA_DIRECT_SOLVER_MODULE_HPP

// Nanobind
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>

// Standard Library
#include <string>

// Phasma
#include "phasma/types.hpp"
#include "phasma/scaler.hpp"
#include "phasma/direct_solvers/direct_solver_wrapper.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace Phasma::bindings {

template <typename Solver, typename Scalar, int Order = Eigen::ColMajor>
void bind_eigen_direct_solver(nb::module_& m, const std::string& class_name) {
    using DirectSolverWrapper = Phasma::DirectSolverWrapper<Solver, Scalar, Order>;
    using SparseMatrix = typename DirectSolverWrapper::SparseMatrix;
    using Vector = typename DirectSolverWrapper::Vector;

    nb::class_<DirectSolverWrapper>(m, class_name.c_str())
        .def(nb::init<Phasma::ScalingType>(), "type"_a = Phasma::ScalingType::None, "Create a DirectSolver object with the given scaling type.")
        .def("analyze_pattern", &DirectSolverWrapper::analyze_pattern, "A"_a, "Analyze the sparsity pattern of the matrix A.")
        .def("factorize", &DirectSolverWrapper::factorize, "A"_a, "Factorize the matrix A.")
        .def("compute", &DirectSolverWrapper::compute, "A"_a, "Analyze the sparsity pattern and factorize the matrix A.")
        .def("solve", &DirectSolverWrapper::solve, "b"_a, "Solve the linear system Ax = b.")
        .def("solve", &DirectSolverWrapper::factorize_and_solve, "A"_a, "b"_a, "Factorize the matrix A and solve the linear system Ax = b.");
}

} // namespace Phasma::bindings

#endif // PHASMA_DIRECT_SOLVER_MODULE_HPP
