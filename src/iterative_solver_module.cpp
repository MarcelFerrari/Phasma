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
#include <unsupported/Eigen/IterativeSolvers> 

// C++ Standard Library
#include <string>
#include <iostream>
#include <optional>

#include "phasma/types.hpp"
#include "phasma/bindings/iterative_solver_module.hpp"
#include "phasma/iterative_solvers/eigen_matrix_free_operator.hpp"

namespace nb = nanobind;
namespace Phasma::bindings {

template <typename Solver, typename Scalar, int Order = Phasma::RowMajor>
void init_iterative_solver_module(nb::module_& m) {
    // Initialize iterative solvers
    // Most use RowMajor
    using SparseMatrix = Phasma::CRSMatrix<double>;
    using SparseMatrix_f = Phasma::CRSMatrix<float>;
    using SparseMatrix_ld = Phasma::CRSMatrix<Phasma::float128>;

    // Conjugate Gradient (CG)
    Phasma::bindings::bind_eigen_iterative_solver<Eigen::ConjugateGradient<SparseMatrix>, double>(m, "Eigen_CG");
    Phasma::bindings::bind_eigen_iterative_solver<Eigen::ConjugateGradient<SparseMatrix_f>, float>(m, "Eigen_CG_f");
    Phasma::bindings::bind_eigen_iterative_solver<Eigen::ConjugateGradient<SparseMatrix_ld>, Phasma::float128>(m, "Eigen_CG_ld");
    
    // BiCGSTAB
    Phasma::bindings::bind_eigen_iterative_solver<Eigen::BiCGSTAB<SparseMatrix>, double>(m, "Eigen_BiCGSTAB");
    Phasma::bindings::bind_eigen_iterative_solver<Eigen::BiCGSTAB<SparseMatrix_f>, float>(m, "Eigen_BiCGSTAB_f");
    Phasma::bindings::bind_eigen_iterative_solver<Eigen::BiCGSTAB<SparseMatrix_ld>, Phasma::float128>(m, "Eigen_BiCGSTAB_ld");

    // LeastSquaresConjugateGradient
    Phasma::bindings::bind_eigen_iterative_solver<Eigen::LeastSquaresConjugateGradient<SparseMatrix>, double>(m, "LeastSquaresCG");
    Phasma::bindings::bind_eigen_iterative_solver<Eigen::LeastSquaresConjugateGradient<SparseMatrix_f>, float>(m, "LeastSquaresCG_f");
    Phasma::bindings::bind_eigen_iterative_solver<Eigen::LeastSquaresConjugateGradient<SparseMatrix_ld>, Phasma::float128>(m, "LeastSquaresCG_ld");
    
    // GMRES
    Phasma::bindings::bind_eigen_iterative_solver<Eigen::GMRES<SparseMatrix>, double>(m, "Eigen_GMRES");
    Phasma::bindings::bind_eigen_iterative_solver<Eigen::GMRES<SparseMatrix_f>, float>(m, "Eigen_GMRES_f");
    Phasma::bindings::bind_eigen_iterative_solver<Eigen::GMRES<SparseMatrix_ld>, Phasma::float128>(m, "Eigen_GMRES_ld");
}

void init_matfree_solver_module(nb::module_& m) {
    // Initialize matrix-free iterative solvers
    using Phasma::EigenSupport::MatrixReplacement;

    // BiCGSTAB
    Phasma::bindings::bind_eigen_matfree_solver<Eigen::BiCGSTAB<MatrixReplacement<double>, Eigen::IdentityPreconditioner>, double>(m, "Eigen_BiCGSTAB");
    Phasma::bindings::bind_eigen_matfree_solver<Eigen::BiCGSTAB<MatrixReplacement<float>, Eigen::IdentityPreconditioner>, float>(m, "Eigen_BiCGSTAB_f");
    Phasma::bindings::bind_eigen_matfree_solver<Eigen::BiCGSTAB<MatrixReplacement<Phasma::float128>, Eigen::IdentityPreconditioner>, Phasma::float128>(m, "Eigen_BiCGSTAB_ld");

    // GMRES
    Phasma::bindings::bind_eigen_matfree_solver<Eigen::GMRES<MatrixReplacement<double>, Eigen::IdentityPreconditioner>, double>(m, "Eigen_GMRES");
    Phasma::bindings::bind_eigen_matfree_solver<Eigen::GMRES<MatrixReplacement<float>, Eigen::IdentityPreconditioner>, float>(m, "Eigen_GMRES_f");
    Phasma::bindings::bind_eigen_matfree_solver<Eigen::GMRES<MatrixReplacement<Phasma::float128>, Eigen::IdentityPreconditioner>, Phasma::float128>(m, "Eigen_GMRES_ld");
}

} // namespace Phasma::bindings
