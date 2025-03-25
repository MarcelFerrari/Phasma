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
#include <nanobind/stl/function.h>

// C++ Standard Library
#include <functional>

// Eigen
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

// Phasma
#include "phasma/types.hpp"
#include "phasma/scaler.hpp"
#include "phasma/iterative_solvers/eigen_matrix_free_operator.hpp"
#include "phasma/iterative_solvers/eigen_iterative_solver_wrapper.hpp"


namespace nb = nanobind;
using namespace nb::literals;

namespace Phasma::bindings {

template <typename Solver, typename Scalar, int Order = Phasma::RowMajor>
void bind_eigen_iterative_solver(nb::module_& m, const std::string& class_name) {
    using EigenIterativeSolverWrapper = Phasma::EigenIterativeSolverWrapper<Solver, Scalar, Order>;
    using SparseMatrix = typename EigenIterativeSolverWrapper::SparseMatrix;
    using Vector = typename EigenIterativeSolverWrapper::Vector;

    nb::class_<EigenIterativeSolverWrapper>(m, class_name.c_str())
        .def(nb::init<Phasma::Scale, bool>(), "type"_a = Phasma::Scale::Identity, "check_convergence"_a = false, "Create an IterativeSolver object with the given scaling type.")
        .def("set_tolerance", &EigenIterativeSolverWrapper::set_tolerance, "tol"_a, "Set the tolerance for the iterative solver.")
        .def("set_max_iterations", &EigenIterativeSolverWrapper::set_max_iterations, "max_iter"_a, "Set the maximum number of iterations.")
        .def("tolerance", &EigenIterativeSolverWrapper::tolerance, "Get the tolerance of the solver.")
        .def("max_iterations", &EigenIterativeSolverWrapper::max_iterations, "Get the maximum number of iterations allowed.")
        .def("iterations", &EigenIterativeSolverWrapper::iterations, "Get the number of iterations performed in the last solve.")
        .def("compute", &EigenIterativeSolverWrapper::compute, "A"_a, "Initialize the solver with the matrix A.")
        .def("solve", &EigenIterativeSolverWrapper::solve, "b"_a, "Solve the linear system Ax = b.")
        .def("solve", &EigenIterativeSolverWrapper::solve_with_guess, "b"_a, "guess"_a, "Solve the linear system Ax = b with an initial guess.")
        .def("solve", &EigenIterativeSolverWrapper::compute_and_solve, "A"_a, "b"_a, "Initialize the solver with A and solve Ax = b.")
        .def("solve", &EigenIterativeSolverWrapper::compute_and_solve_with_guess, "A"_a, "b"_a, "guess"_a, "Initialize the solver with A and solve Ax = b with an initial guess.")
;}


template <typename Solver, typename Scalar>
void bind_eigen_matfree_solver(nb::module_& m, const std::string& class_name) {
    using Vector = Phasma::Vector<Scalar>;
    using MatrixReplacement = Phasma::EigenSupport::MatrixReplacement<Scalar>;
    using MatrixOp = typename MatrixReplacement::MatrixOp;

    nb::class_<Solver>(m, class_name.c_str())
        .def(nb::init<>(), "Create an IterativeSolver object.")
                .def("set_tolerance", [](Solver& solver, double tol) -> void {
                solver.setTolerance(tol);
            }, "tol"_a, "Set the tolerance for the iterative solver.")
        .def("set_max_iterations", [](Solver& solver, int max_iter) -> void {
                solver.setMaxIterations(max_iter);
            }, "max_iter"_a, "Set the maximum number of iterations.")
        .def("tolerance", [](Solver& solver) -> double {
                return solver.tolerance();
            }, "Get the tolerance of the solver.")
        .def("max_iterations", [](Solver& solver) -> int {
                return solver.maxIterations();
            }, "Get the maximum number of iterations allowed.")
        .def("iterations", [](Solver& solver) -> int {
                return solver.iterations();
            }, "Get the number of iterations performed in the last solve.")
        .def("error", [](Solver& solver) -> double {
                return solver.error();
            }, "Get the error of the last solve.")
        .def("solve", [](Solver& solver, const MatrixOp& op, int rows, int cols, const Eigen::Ref<const Vector>& rhs) -> Vector {
            MatrixReplacement op_wrapper(op, rows, cols);
            solver.compute(op_wrapper); 
            return solver.solve(rhs); 
        }, "f"_a, "rows"_a, "cols"_a, "rhs"_a, "Solve the linear system Ax = b.")
        .def("solve", [](Solver& solver, const MatrixOp& op, int rows, int cols, const Eigen::Ref<const Vector>& rhs, const Eigen::Ref<const Vector>& guess) -> Vector {
            MatrixReplacement op_wrapper(op, rows, cols);
            solver.compute(op_wrapper); 
            return solver.solveWithGuess(rhs, guess); 
        }, "f"_a, "rows"_a, "cols"_a, "rhs"_a, "guess"_a, "Solve the linear system Ax = b with an initial guess.")
;}



} // namespace Phasma::bindings

#endif // PHASMA_ITERATIVE_SOLVER_MODULE_HPP
