/*
Phasma: fast sparse linear algebra for Python.
https://github.com/MarcelFerrari/phasma

File: utils.hpp
Description: Utility functions and classes for Phasma.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

See LICENSE file in the project root for full license information.
*/

#ifndef PHASMA_UTILS_HPP
#define PHASMA_UTILS_HPP

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/eigen/dense.h>
#include <Eigen/Core>

#include "phasma/types.hpp"

namespace nb = nanobind;

namespace Phasma {

/*
Iterator class used to construct a sparse matrix from a vector representation
of a COO matrix instead of using a std::vector of Eigen triplets.
This is faster and easier to use than the Eigen triplet interface.
*/
template <typename Scalar>
class ArrayTripletIterator {
    using IndexArray = nb::DRef<Eigen::Array<Phasma::Index, Eigen::Dynamic, 1>>;
    using ScalarArray = nb::DRef<Eigen::Array<Scalar, Eigen::Dynamic, 1>>;

    IndexArray rows_;
    IndexArray cols_;
    ScalarArray values_;
    Phasma::Index index_;

public:
    ArrayTripletIterator(IndexArray rows,
                         IndexArray cols,
                         ScalarArray values, Phasma::Index index = 0)
        : rows_(rows), cols_(cols), values_(values), index_(index) {}

    Phasma::Index row() const { return rows_[index_]; }
    Phasma::Index col() const { return cols_[index_]; }
    Scalar value() const { return values_[index_]; }

    ArrayTripletIterator& operator++() {
        ++index_;
        return *this;
    }

    bool operator!=(const ArrayTripletIterator& other) const { return index_ != other.index_; }

    // Pointer-like interface for Eigen compatibility
    const ArrayTripletIterator* operator->() const { return this; }
};


// Function to compute inverse row norms of a sparse matrix
template <typename Scalar>
Phasma::Vector<Scalar> compute_inverse_row_norms(const Phasma::SparseMatrix<Scalar, Eigen::RowMajor> &A) {
    Phasma::Vector<Scalar> norms(A.rows());
    for (Phasma::Index i = 0; i < A.rows(); ++i) {
        // norms[i] = A.row(i).cwiseAbs().sum()/A.cols();
        norms[i] = A.row(i).norm();
    }
    return norms.array().inverse();
}

// Function to compute inverse column norms of a sparse matrix
template <typename Scalar>
Phasma::Vector<Scalar> compute_inverse_col_norms(const Phasma::SparseMatrix<Scalar, Eigen::ColMajor> &A) {
    Phasma::Vector<Scalar> norms(A.cols());
    for (Phasma::Index i = 0; i < A.cols(); ++i) {
        // norms[i] = A.col(i).cwiseAbs().sum()/A.rows();
        norms[i] = A.col(i).norm();
    }
    return norms.array().inverse();
}

} // namespace Phasma
#endif // PHASMA_UTILS_HPP