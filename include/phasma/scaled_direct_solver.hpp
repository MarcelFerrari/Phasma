/*
Phasma: fast sparse linear algebra for Python.
https://github.com/MarcelFerrari/phasma

File: linear_solver_wrapper.hpp
Description: Wrapper that exposes bindings for linear solver classes following
the Eigen 3 sparse solver API. This class handles matrix scaling.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

See LICENSE file in the project root for full license information.
*/

#ifndef PHASMA_DIRECT_SOLVER_HPP
#define PHASMA_DIRECT_SOLVER_HPP

// Nanobind
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>

// Eigen
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/OrderingMethods>

// C++ Standard Library
#include <string>
#include <iostream>
#include <optional>

// Phasma
#include "phasma/types.hpp"
#include "phasma/scaler.hpp"

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

namespace Phasma {

template <typename Solver, typename Scalar, int Order = Eigen::ColMajor>
class ScaledDirectSolver {
    /*
        Wrapper class for direct solvers that follow the Eigen 3 sparse solver API.
        This class handles matrix scaling without having to reimplement it for each solver.
    */
public:
    using SparseMatrix = Phasma::SparseMatrix<Scalar, Order>;
    using Vector = Phasma::Vector<Scalar>;

    ScaledDirectSolver(Phasma::ScalingType t = Phasma::ScalingType::NONE) : solver_(), scaler_(t) {};

    void analyze_pattern(const SparseMatrix& A) {
        solver_.analyzePattern(A);
        pattern_analyzed_ = true;
    }

    void factorize(const SparseMatrix& A) {
        if (!pattern_analyzed_) {
            std::cout << "DirectSolver: Warning: symbolic factorization has not been performed." << std::endl;
            std::cout << "DirectSolver: Performing symbolic factorization now." << std::endl;
            std::cout << "DirectSolver: Further calls to factorize() will be faster." << std::endl;
            analyze_pattern(A);
        }

        if(scaler_.type() != Phasma::ScalingType::NONE) {
            SparseMatrix A_scaled = scaler_.scale(A);
            solver_.factorize(A_scaled);
        } else {
            solver_.factorize(A);
        }

        if (solver_.info() != Eigen::Success) {
            throw std::runtime_error("DirectSolver: Numerical factorization failed.");
        }

        factorized_ = true;
    }

    void compute(const SparseMatrix& A) {
        analyze_pattern(A);
        factorize(A);
    }

    Vector solve(const Vector& b) const {
        if (!factorized_) {
            throw std::runtime_error("DirectSolver: Matrix has not been factorized.");
        }

        Vector x;
        // Scale input vector and solve if necessary
        if (scaler_.type() == Phasma::ScalingType::FULL || scaler_.type() == Phasma::ScalingType::ROW) {
            Vector b_scaled = scaler_.scale(b);
            x = solver_.solve(b_scaled);
        } else {
            x = solver_.solve(b);
        }

        // Check if the solver failed
        if (solver_.info() != Eigen::Success) {
            throw std::runtime_error("DirectSolver: Solving failed.");
        }

        // Unscale the solution if necessary
        if (scaler_.type() == Phasma::ScalingType::FULL || scaler_.type() == Phasma::ScalingType::COL) {
            return scaler_.unscale(x);
        } else {
            return x;
        }
    }

    Vector factorize_and_solve(const SparseMatrix& A, const Vector& b) {
        compute(A);
        return solve(b);
    }

private:
    Solver solver_;
    Phasma::Scaler<Scalar> scaler_;
    bool pattern_analyzed_ = false;
    bool factorized_ = false;    
};

template <typename Solver, typename Scalar, int Order = Eigen::ColMajor>
void bind_sparse_solver(nb::module_& m, const std::string& class_name) {
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

void init_direct_solver_module(nb::module_& m) {
    // Initialize Eigen 3 direct solvers
    // Most use ColMajor format so we create a convenient alias
    using SparseMatrix = Phasma::SparseMatrix<double>;

    // SparseLU
    Phasma::bind_sparse_solver<Eigen::SparseLU<SparseMatrix>, double>(m, "SparseLU");

    // SparseQR
    Phasma::bind_sparse_solver<Eigen::SparseQR<SparseMatrix, Eigen::COLAMDOrdering<Phasma::Index>>, double>(m, "SparseQR");
    
    // PardisoLU
    #ifdef PHASMA_PARDISO_SUPPORT
    #pragma message("Pardiso support is enabled.")
    Phasma::bind_sparse_solver<Eigen::PardisoLU<SparseMatrix>, double>(m, "PardisoLU");
    #endif

    // SuiteSparse
    #ifdef PHASMA_SUITESPARSE_SUPPORT
    #pragma message("SuiteSparse support is enabled.")
    Phasma::bind_sparse_solver<Eigen::UmfPackLU<SparseMatrix>, double>(m, "UmfPackLU");
    Phasma::bind_sparse_solver<Eigen::KLU<SparseMatrix>, double>(m, "KLU");
    #endif
}

} // namespace Phasma

#endif // PHASMA_LINEAR_SOLVER_WRAPPER_HPP
