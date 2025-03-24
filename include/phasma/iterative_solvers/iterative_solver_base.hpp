/*
Phasma: fast sparse linear algebra for Python.
https://github.com/MarcelFerrari/phasma

File: iterative_solver_base.hpp
Description: Base class for iterative solvers in Phasma.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

#ifndef PHASMA_ITERATIVE_SOLVER_BASE_HPP
#define PHASMA_ITERATIVE_SOLVER_BASE_HPP

// Eigen
#include <Eigen/Core>

// Standard Lib
#include <optional>
#include <functional>
#include <stdexcept>
#include <limits> // For std::numeric_limits

// Phasma
#include "phasma/types.hpp"

namespace Phasma {

template <typename Scalar>
class IterativeSolverBase {

public:
    using SparseMatrix = Phasma::SparseMatrix<Scalar, Phasma::RowMajor>;
    using Vector = Phasma::Vector<Scalar>;

    IterativeSolverBase() = default;
    IterativeSolverBase(const SparseMatrix& A) { set_matrix(A); }

    void setTolerance(Scalar tol) noexcept {
        tol_ = tol;
    }

    void setMaxIterations(int max_iter) {
        if (max_iter < 0) {
            throw std::invalid_argument("IterativeSolver: Maximum number of iterations must be strictly positive.");
        }
        max_iter_ = max_iter;
    }

    Scalar tolerance() const noexcept {
        return tol_;
    }

    int maxIterations() const noexcept {
        return max_iter_;
    }

    int iterations() const noexcept {
        return iterations_;
    }

    void compute(const SparseMatrix& A) {
        set_matrix(A);
    }

    int info() const noexcept {
        return info_;
    }

    Vector solve(const Vector& b) const {
        const SparseMatrix& A = get_matrix_reference();
        set_max_iter(A);
        return solve_impl(A, b, Vector::Zero(b.size()));
    }

    Vector solveWithGuess(const Vector& b, const Vector& guess) const {
        const SparseMatrix& A = get_matrix_reference();
        set_max_iter(A);
        return solve_impl(A, b, guess);
    }

protected:
    std::optional<std::reference_wrapper<const SparseMatrix>> A_;
    Scalar tol_ = std::numeric_limits<Scalar>::epsilon(); // Default to epsilon for Scalar
    mutable int max_iter_ = -1; // Default to an invalid value
    mutable int iterations_ = 0;
    mutable Eigen::ComputationInfo info_ = Eigen::Success;

    void set_max_iter(const SparseMatrix& A) const {
        if (max_iter_ < 0) {
            max_iter_ = 2 * A.cols();
        }
    }

    void set_matrix(const SparseMatrix& A) {
        // Ensure the matrix is square
        if (A.rows() != A.cols()) {
            throw std::invalid_argument("Matrix A must be square.");
        }

        A_ = A;
    }
    
    const SparseMatrix& get_matrix_reference() const {
        if (!A_) {
            throw std::runtime_error("IterativeSolver: Matrix has not been initialized. Call compute() first.");
        }
        return A_.value().get();
    }

    // Actual solver implementation
    virtual Vector solve_impl(const SparseMatrix& A, const Vector& b, const Vector& guess) const = 0;
};

} // namespace Phasma

#endif // PHASMA_ITERATIVE_SOLVER_BASE_HPP
