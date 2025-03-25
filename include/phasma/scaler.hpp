/*
Phasma: fast sparse linear algebra for Python.
https://github.com/MarcelFerrari/phasma

File: scaler.hpp
Description: Scaler object that applies left and right scaling to a matrix.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

#ifndef PHASMA_SCALER_HPP
#define PHASMA_SCALER_HPP

// C++ Standard Library
#include <optional>

// Phasma
#include "phasma/types.hpp"

namespace Phasma {

template <typename Scalar>
class Scaler {
public:
    using CCSMatrix = Phasma::SparseMatrix<Scalar, Phasma::ColMajor>;
    using CRSMatrix = Phasma::SparseMatrix<Scalar, Phasma::RowMajor>;
    using Vector = Phasma::Vector<Scalar>;

    Scaler(Scale t) : type_(t) {}
     
    const Scale & type() const { return type_; }

    CCSMatrix scale_ccs(const CCSMatrix& A_ccs) {
        if(type_ == Scale::Full) {
            CRSMatrix A_crs = A_ccs;
            return scale_full_ccs(A_ccs, A_crs);
        } else if (type_ == Scale::Row) {
            CRSMatrix A_crs = A_ccs;
            return scale_row_ccs(A_ccs, A_crs);
        } else if (type_ == Scale::Col) {
            return scale_col_ccs(A_ccs);
        } else {
            throw std::runtime_error("Scaler: Error attempting to scale matrix. Scaling type is 'None'.");
        }
    }

    CRSMatrix scale_crs(const CRSMatrix& A_crs) {
        if(type_ == Scale::Full) {
            CCSMatrix A_ccs = A_crs;
            return scale_full_crs(A_ccs, A_crs);
        } else if (type_ == Scale::Row) {
            return scale_row_crs(A_crs);
        } else if (type_ == Scale::Col) {
            CCSMatrix A_ccs = A_crs;
            return scale_col_crs(A_ccs, A_crs);
        } else {
            throw std::runtime_error("Scaler: Error attempting to scale matrix. Scaling type is 'None'.");
        }
    }

    Vector scale_vec(const Vector& v) const {
        if(type_ == Scale::Full) {
            return scale_vector(v);
        } else if (type_ == Scale::Row) {
            return scale_vector(v);
        } else if (type_ == Scale::Col) {
            throw std::runtime_error("Scaler: Error attempting to scale vector. Scaling type is 'Col'.");
        } else {
            throw std::runtime_error("Scaler: Error attempting to scale vector. Scaling type is 'None'.");
        }
    }

    Vector unscale(const Vector& v) const {
        if(type_ == Scale::Full) {
            return unscale_vector(v);
        } else if (type_ == Scale::Row) {
            throw std::runtime_error("Scaler: Error attempting to unscale vector. Scaling type is 'Row'.");
        } else if (type_ == Scale::Col) {
            return unscale_vector(v);
        } else {
            throw std::runtime_error("Scaler: Error attempting to unscale vector. Scaling type is 'None'.");
        }
    }

    // Convenience overload for calls from C++
    // does not work with Python bindings
    CCSMatrix scale(const CCSMatrix& A_ccs){
        return scale_ccs(A_ccs);
    }

    CRSMatrix scale(const CRSMatrix& A_crs){
        return scale_crs(A_crs);
    }

    Vector scale(const Vector& v) const {
        return scale_vec(v);
    }

private:
    Scale type_;
    std::optional<Vector> Dr_inv_; // Row scaling factors
    std::optional<Vector> Dc_inv_; // Column scaling factors

    // Function to compute inverse row norms of a sparse matrix
    Vector compute_inverse_row_norms(const CRSMatrix &A) {
        Vector norms(A.rows());
        
        #pragma omp parallel for schedule(dynamic)
        for (Phasma::Index i = 0; i < A.rows(); ++i) {
            norms[i] = A.row(i).norm();
        }
        return std::move(norms.array().inverse());
    }

    // Function to compute inverse column norms of a sparse matrix
    Vector compute_inverse_col_norms(const CCSMatrix &A) {
        Vector norms(A.cols());

        #pragma omp parallel for schedule(dynamic)
        for (Phasma::Index i = 0; i < A.cols(); ++i) {
            norms[i] = A.col(i).norm();
        }
        return std::move(norms.array().inverse());
    }

    // ============= Scale CCS Matrix =================
    CCSMatrix scale_full_ccs(const CCSMatrix& A_ccs, const CRSMatrix& A_crs) {
        Dr_inv_ = compute_inverse_row_norms(A_crs);
        Dc_inv_ = compute_inverse_col_norms(A_ccs);
        return (Dr_inv_->asDiagonal() * A_ccs * Dc_inv_->asDiagonal());
    }

    CCSMatrix scale_row_ccs(const CCSMatrix& A_ccs, const CRSMatrix& A_crs) {
        Dr_inv_ = compute_inverse_row_norms(A_crs);
        return Dr_inv_->asDiagonal() * A_ccs;
    }

    CCSMatrix scale_col_ccs(const CCSMatrix& A_ccs) {
        Dc_inv_ = compute_inverse_col_norms(A_ccs);
        return A_ccs * Dc_inv_->asDiagonal();
    }

    // ============= Scale CRS Matrix =================
    CRSMatrix scale_full_crs(const CCSMatrix& A_ccs, const CRSMatrix& A_crs) {
        Dr_inv_ = compute_inverse_row_norms(A_crs);
        Dc_inv_ = compute_inverse_col_norms(A_ccs);
        return (Dr_inv_->asDiagonal() * A_crs * Dc_inv_->asDiagonal());
    }

    CRSMatrix scale_row_crs(const CRSMatrix& A_crs) {
        Dr_inv_ = compute_inverse_row_norms(A_crs);
        return Dr_inv_->asDiagonal() * A_crs;
    }

    CRSMatrix scale_col_crs(const CCSMatrix& A_ccs, const CRSMatrix& A_crs) {
        Dc_inv_ = compute_inverse_col_norms(A_ccs);
        return A_crs * Dc_inv_->asDiagonal();
    }

    // ============= Scale Vector =================
    Vector scale_vector(const Vector& v) const {
        if(!Dr_inv_.has_value()) {
            throw std::runtime_error("Scaler: Error scaling vector. Need to scale matrix first.");
        }

        return Dr_inv_->cwiseProduct(v);
    }

    Vector unscale_vector(const Vector& v) const {
        if(!Dc_inv_.has_value()) {
            throw std::runtime_error("Scaler: Error unscaling vector. Need to scale matrix first.");
        }

        return Dc_inv_->cwiseProduct(v);
    }
};
} // namespace Phasma

#endif // PHASMA_SCALER_HPP