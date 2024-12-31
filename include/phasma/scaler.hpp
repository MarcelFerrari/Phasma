/*
Phasma: fast sparse linear algebra for Python.
https://github.com/MarcelFerrari/phasma

File: scaler.hpp
Description: Scaler object that applies left and right scaling to a matrix.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

See LICENSE file in the project root for full license information.
*/

#ifndef PHASMA_SCALER_HPP
#define PHASMA_SCALER_HPP

// Eigen
#include <Eigen/Sparse>
#include <string>

// C++ Standard Library
#include <optional>

// Phasma
#include "phasma/types.hpp"

namespace nb = nanobind;

namespace Phasma {

enum class ScalingType {
    NONE,
    ROW,
    COL,
    FULL,
};

template <typename Scalar>
class Scaler {
public:
    using CCSMatrix = Phasma::SparseMatrix<Scalar, Eigen::ColMajor>;
    using CRSMatrix = Phasma::SparseMatrix<Scalar, Eigen::RowMajor>;
    using Vector = Phasma::Vector<Scalar>;

    Scaler(ScalingType t) : type_(t) {}
     
    const ScalingType & type() const { return type_; }

    CCSMatrix scale_ccs(const CCSMatrix& A_ccs) {
        if(type_ == ScalingType::FULL) {
            CRSMatrix A_crs = A_ccs;
            return scale_full_ccs(A_ccs, A_crs);
        } else if (type_ == ScalingType::ROW) {
            CRSMatrix A_crs = A_ccs;
            return scale_row_ccs(A_ccs, A_crs);
        } else if (type_ == ScalingType::COL) {
            return scale_col_ccs(A_ccs);
        } else {
            throw std::runtime_error("Scaler: Error attempting to scale matrix. Scaling type is 'NONE'.");
        }
    }

    CRSMatrix scale_crs(const CRSMatrix& A_crs) {
        if(type_ == ScalingType::FULL) {
            CCSMatrix A_ccs = A_crs;
            return scale_full_crs(A_ccs, A_crs);
        } else if (type_ == ScalingType::ROW) {
            return scale_row_crs(A_crs);
        } else if (type_ == ScalingType::COL) {
            CCSMatrix A_ccs = A_crs;
            return scale_col_crs(A_ccs, A_crs);
        } else {
            throw std::runtime_error("Scaler: Error attempting to scale matrix. Scaling type is 'NONE'.");
        }
    }

    Vector scale_vec(const Vector& v) const {
        if(type_ == ScalingType::FULL) {
            return scale_vector(v);
        } else if (type_ == ScalingType::ROW) {
            return scale_vector(v);
        } else if (type_ == ScalingType::COL) {
            throw std::runtime_error("Scaler: Error attempting to scale vector. Scaling type is 'COL'.");
        } else {
            throw std::runtime_error("Scaler: Error attempting to scale vector. Scaling type is 'NONE'.");
        }
    }

    Vector unscale(const Vector& v) const {
        if(type_ == ScalingType::FULL) {
            return unscale_vector(v);
        } else if (type_ == ScalingType::ROW) {
            throw std::runtime_error("Scaler: Error attempting to unscale vector. Scaling type is 'ROW'.");
        } else if (type_ == ScalingType::COL) {
            return unscale_vector(v);
        } else {
            throw std::runtime_error("Scaler: Error attempting to unscale vector. Scaling type is 'NONE'.");
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
    ScalingType type_;
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