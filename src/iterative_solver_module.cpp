/*
Phasma: fast sparse linear algebra for Python.
https://github.com/MarcelFerrari/phasma

File: iterative_solver_module.cpp
Description: Iterative solver module for binding Phasma to Python using NanoBind.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

#include <nanobind/nanobind.h>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

// C++ Standard Library
#include <string>
#include <iostream>
#include <optional>

#include "phasma/types.hpp"
#include "phasma/bindings/iterative_solver_module.hpp"

namespace nb = nanobind;
namespace Phasma::bindings {

void init_iterative_solver_module(nb::module_& m) {
    // Initialize Eigen 3 iterative solvers
    // Most use ColMajor format so we create a convenient alias
    using SparseMatrix = Phasma::CRSMatrix<double>;

    // Conjugate Gradient (CG)
    Phasma::bindings::bind_iterative_solver<Eigen::ConjugateGradient<SparseMatrix>, double>(m, "ConjugateGradient");

    // BiCGSTAB
    Phasma::bindings::bind_iterative_solver<Eigen::BiCGSTAB<SparseMatrix>, double>(m, "BiCGSTAB");

    // LeastSquaresConjugateGradient
    Phasma::bindings::bind_iterative_solver<Eigen::LeastSquaresConjugateGradient<SparseMatrix>, double>(m, "LeastSquaresCG");
}

} // namespace Phasma::bindings
