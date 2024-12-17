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
#include "phasma/utils.hpp"
#include "phasma/types.hpp"
#include "phasma/sparse_matrix.hpp"

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

template <typename Solver, typename Scalar>
class DirectSolver {
    /*
        Wrapper class for direct solvers that follow the Eigen 3 sparse solver API.
        This class handles matrix scaling without having to reimplement it for each solver.
    */
public:
    using CCSSpMat = Phasma::SparseMatrix<Scalar, Eigen::ColMajor>;
    using CRSSpMat = Phasma::SparseMatrix<Scalar, Eigen::RowMajor>;
    using Vector = Phasma::Vector<Scalar>;

    DirectSolver() : solver_() {};

    void analyze_pattern(const CCSSpMat& A) {
        solver_.analyzePattern(A);
        pattern_analyzed_ = true;
    }

    void factorize(const CCSSpMat& A, const std::string & scale_matrix = "none") {
        if (!pattern_analyzed_) {
            std::cout << "DirectSolver: Warning: symbolic factorization has not been performed." << std::endl;
            std::cout << "DirectSolver: Performing symbolic factorization now." << std::endl;
            std::cout << "DirectSolver: Further calls to factorize() will be faster." << std::endl;
            analyze_pattern(A);
        }

        if (scale_matrix != "none") {
            if (scale_matrix != "full" && scale_matrix != "row" && scale_matrix != "col") {
                throw std::runtime_error("DirectSolver: Invalid scaling option. Use 'none', 'full', 'row' or 'col'.");
            }
                if(scale_matrix == "full") {
                    CRSSpMat A_crs = A;
                    Dr_inv_ = compute_inverse_row_norms(A_crs);
                    Dc_inv_ = compute_inverse_col_norms(A);
                    CCSSpMat A_scaled = (Dr_inv_->asDiagonal() * A * Dc_inv_->asDiagonal()).eval();
                    solver_.factorize(A_scaled);
                } else if(scale_matrix == "row") {
                    CRSSpMat A_crs = A;
                    Dr_inv_ = compute_inverse_row_norms(A_crs);
                    CCSSpMat A_scaled = (Dr_inv_->asDiagonal() * A).eval();
                    solver_.factorize(A_scaled);
                } else if(scale_matrix == "col") {
                    Dc_inv_ = compute_inverse_col_norms(A);
                    CCSSpMat A_scaled = (A * Dc_inv_->asDiagonal()).eval();
                    solver_.factorize(A_scaled);
                }
        } else {
            solver_.factorize(A);
        }

        if (solver_.info() != Eigen::Success) {
            throw std::runtime_error("DirectSolver: Numerical factorization failed.");
        }

        scale_ = scale_matrix;
        factorized_ = true;
    }

    void compute(const CCSSpMat& A, const std::string & scale_matrix = "none") {
        analyze_pattern(A);
        factorize(A, scale_matrix);
    }

    Vector solve(const Vector& b) const {
        if (!factorized_) {
            throw std::runtime_error("DirectSolver: Matrix has not been factorized.");
        }

        Vector x;
        if (scale_ != "none") {
            if(scale_ == "full"){
                Vector b_scaled = Dr_inv_->cwiseProduct(b);
                x = solver_.solve(b_scaled);
                x = (Dc_inv_->cwiseProduct(x)).eval();
            } else if (scale_ == "row") {
                Vector b_scaled = Dr_inv_->cwiseProduct(b);
                x = solver_.solve(b_scaled);
            } else if (scale_ == "col") {
                x = solver_.solve(b);
                x = (Dc_inv_->cwiseProduct(x)).eval();
            } else {
                throw std::runtime_error("DirectSolver: Invalid scaling option. Use 'none', 'full', 'row' or 'col'.");
            }
        } else {
            x = solver_.solve(b);
        }

        if (solver_.info() != Eigen::Success) {
            throw std::runtime_error("DirectSolver: Solving failed.");
        }

        return x;
    }

    Vector factorize_and_solve(const CCSSpMat& A, const Vector& b, const std::string & scale_matrix = "col") {
        compute(A, scale_matrix);
        return solve(b);
    }

private:
    Solver solver_;
    bool pattern_analyzed_ = false;
    bool factorized_ = false;
    std::string scale_ = "none";
    std::optional<Vector> Dr_inv_; // Row scaling factors
    std::optional<Vector> Dc_inv_; // Column scaling factors
};

template <typename Solver, typename Scalar, int Order = Eigen::ColMajor>
void bind_sparse_solver(nb::module_& m, const std::string& class_name) {
    using SparseMatrix = Eigen::SparseMatrix<Scalar, Order, Phasma::Index>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    nb::class_<DirectSolver<Solver, Scalar>>(m, class_name.c_str())
        .def(nb::init<>(), "Default constructor")
        .def("analyze_pattern", &DirectSolver<Solver, Scalar>::analyze_pattern, "A"_a, "Analyze the sparsity pattern of the matrix A.")
        .def("factorize", &DirectSolver<Solver, Scalar>::factorize, "A"_a, "scale_matrix"_a = "none", "Factorize the matrix A.")
        .def("compute", &DirectSolver<Solver, Scalar>::compute, "A"_a, "scale_matrix"_a = "none", "Analyze the sparsity pattern and factorize the matrix A.")
        .def("solve", &DirectSolver<Solver, Scalar>::solve, "b"_a, "Solve the linear system Ax = b.")
        .def("solve", &DirectSolver<Solver, Scalar>::factorize_and_solve, "A"_a, "b"_a, "scale_matrix"_a = "col", "Factorize the matrix A and solve the linear system Ax = b.");
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

#endif // PHADMA_LINEAR_SOLVER_WRAPPER_HPP
