/*
Phasma: fast sparse linear algebra for Python.
https://github.com/MarcelFerrari/phasma

File: iterative_solver_module.hpp
Description: Exposes iterative solvers to Python using Nanobind.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

#ifndef PHASMA_ITERATIVE_SOLVER_MODULE_HPP
#define PHASMA_ITERATIVE_SOLVER_MODULE_HPP

// Nanobind
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>

// Eigen
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

// Phasma
#include "phasma/types.hpp"
#include "phasma/scaler.hpp"
#include "phasma/scaled_iterative_solver.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace Phasma::bindings {

template <typename Solver, typename Scalar, int Order = Phasma::RowMajor>
void bind_iterative_solver(nb::module_& m, const std::string& class_name) {
    using ScaledIterativeSolver = Phasma::ScaledIterativeSolver<Solver, Scalar, Order>;
    using SparseMatrix = typename ScaledIterativeSolver::SparseMatrix;
    using Vector = typename ScaledIterativeSolver::Vector;

    nb::class_<ScaledIterativeSolver>(m, class_name.c_str())
        .def(nb::init<Phasma::ScalingType, bool>(), "type"_a = Phasma::ScalingType::None, "check_convergence"_a = false, "Create an IterativeSolver object with the given scaling type.")
        .def("set_tolerance", &ScaledIterativeSolver::set_tolerance, "tol"_a, "Set the tolerance for the iterative solver.")
        .def("set_max_iterations", &ScaledIterativeSolver::set_max_iterations, "max_iter"_a, "Set the maximum number of iterations.")
        .def("tolerance", &ScaledIterativeSolver::tolerance, "Get the tolerance of the solver.")
        .def("max_iterations", &ScaledIterativeSolver::max_iterations, "Get the maximum number of iterations allowed.")
        .def("iterations", &ScaledIterativeSolver::iterations, "Get the number of iterations performed in the last solve.")
        .def("compute", &ScaledIterativeSolver::compute, "A"_a, "Initialize the solver with the matrix A.")
        .def("solve", &ScaledIterativeSolver::solve, "b"_a, "Solve the linear system Ax = b.")
        .def("solve", &ScaledIterativeSolver::compute_and_solve, "A"_a, "b"_a, "Initialize the solver with A and solve Ax = b.");
}

} // namespace Phasma::bindings

#endif // PHASMA_ITERATIVE_SOLVER_MODULE_HPP
