/*
Phasma: fast sparse linear algebra for Python.
https://github.com/MarcelFerrari/phasma

File: scaled_direct_solver.hpp
Description: Wrapper that exposes bindings for direct solver classes following
the Eigen 3 sparse solver API. This class handles matrix scaling automatically.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

#ifndef PHASMA_DIRECT_SOLVER_HPP
#define PHASMA_DIRECT_SOLVER_HPP

// Eigen
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/OrderingMethods>

// C++ Standard Library
#include <string>

// Eigen
#include <Eigen/Core>

// Phasma
#include "phasma/types.hpp"
#include "phasma/scaler.hpp"

namespace Phasma {

template <typename DirectSolver, typename Scalar, int Order = Phasma::ColMajor>
class DirectSolverWrapper {
    /*
        Wrapper class for direct solvers that follow the Eigen 3 sparse solver API.
        This class handles matrix scaling without having to reimplement it for each solver.
    */
public:
    using SparseMatrix = Phasma::SparseMatrix<Scalar, Order>;
    using Vector = Phasma::Vector<Scalar>;

    DirectSolverWrapper(Phasma::Scale t = Phasma::Scale::Identity) : solver_(), scaler_(t) {};

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

        if(scaler_.type() != Phasma::Scale::Identity) {
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
        if (scaler_.type() == Phasma::Scale::Full || scaler_.type() == Phasma::Scale::Row) {
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
        if (scaler_.type() == Phasma::Scale::Full || scaler_.type() == Phasma::Scale::Col) {
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
    DirectSolver solver_;
    Phasma::Scaler<Scalar> scaler_;
    bool pattern_analyzed_ = false;
    bool factorized_ = false;    
};

} // namespace Phasma

#endif // PHASMA_LINEAR_SOLVER_WRAPPER_HPP
