/*
Phasma: fast sparse linear algebra for Python.
https://github.com/MarcelFerrari/phasma

File: scaled_iterative_solver.hpp
Description: Wrapper that exposes bindings for iterative solver classes following
the Eigen 3 sparse solver API. This class handles matrix scaling automatically.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

#ifndef PHASMA_ITERATIVE_SOLVER_HPP
#define PHASMA_ITERATIVE_SOLVER_HPP

// Eigen
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/OrderingMethods>

// Phasma
#include "phasma/types.hpp"
#include "phasma/scaler.hpp"

namespace Phasma {

template <typename IterativeSolver, typename Scalar, int Order = Phasma::RowMajor>
class ScaledIterativeSolver {
    /*
        Wrapper class for iterative solvers that follow the Eigen 3 sparse solver API.
        This class handles matrix scaling and iterative solver parameters.
    */
public:
    using SparseMatrix = Phasma::SparseMatrix<Scalar, Order>;
    using Vector = Phasma::Vector<Scalar>;

    ScaledIterativeSolver(Phasma::ScalingType t = Phasma::ScalingType::None,
                          bool check_convergence = False)
    : solver_(),
      scaler_(t), 
      check_convergence_(check_convergence)
    {};

    void set_tolerance(double tol) {
        solver_.setTolerance(tol);
    }

    void set_max_iterations(int max_iter) {
        solver_.setMaxIterations(max_iter);
    }

    double tolerance() const {
        return solver_.tolerance();
    }

    int max_iterations() const {
        return solver_.maxIterations();
    }

    int iterations() const {
        return solver_.iterations();
    }

    void compute(const SparseMatrix& A) {
        if (scaler_.type() != Phasma::ScalingType::None) {
            SparseMatrix A_scaled = scaler_.scale(A);
            solver_.compute(A_scaled);
        } else {
            solver_.compute(A);
        }

        if (solver_.info() != Eigen::Success) {
            throw std::runtime_error("IterativeSolver: Initialization failed.");
        }

        matrix_initialized_ = true;
    }

    Vector solve(const Vector& b) const {
        if (!matrix_initialized_) {
            throw std::runtime_error("IterativeSolver: Matrix has not been initialized.");
        }

        Vector x;
        // Scale input vector and solve if necessary
        if (scaler_.type() == Phasma::ScalingType::Full || scaler_.type() == Phasma::ScalingType::Row) {
            Vector b_scaled = scaler_.scale(b);
            x = solver_.solve(b_scaled);
        } else {
            x = solver_.solve(b);
        }

        // Check if the solver failed
        if (solver_.info() != Eigen::Success) {
            throw std::runtime_error("IterativeSolver: Solving failed.");
        }

        // Unscale the solution if necessary
        if (scaler_.type() == Phasma::ScalingType::Full || scaler_.type() == Phasma::ScalingType::Col) {
            return scaler_.unscale(x);
        } else {
            return x;
        }
    }

    Vector compute_and_solve(const SparseMatrix& A, const Vector& b) {
        compute(A);
        return solve(b);
    }

private:
    IterativeSolver solver_;
    Phasma::Scaler<Scalar> scaler_;
    bool matrix_initialized_ = false;
    const bool check_convergence_;
};

} // namespace Phasma

#endif // PHASMA_ITERATIVE_SOLVER_HPP
