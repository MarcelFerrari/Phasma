/*
Phasma: fast sparse linear algebra for Python.
https://github.com/MarcelFerrari/phasma

File: sparse_matrix.hpp
Description: Exposes Eigen's SparseMatrix class to Python using Nanobind.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

#ifndef PHASMA_SPARSE_MATRIX_MODULE_HPP
#define PHASMA_SPARSE_MATRIX_MODULE_HPP

// Nanobind
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>

// Standard Library
#include <string>

// Eigen
#include <Eigen/Core>

// Phasma
#include "phasma/types.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace Phasma::bindings {

/*
Iterator class used to construct a sparse matrix from a vector representation
of a COO matrix instead of using a std::vector of Eigen triplets.
This is faster and easier to use than the Eigen triplet interface.
*/

template <typename Scalar>
class ArrayTripletIterator {
    using IndexArray = nb::DRef<Phasma::Array<Phasma::Index>>;
    using ScalarArray = nb::DRef<Phasma::Array<Scalar>>;

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


// Set a sparse matrix from arrays of triplets in COO format
template <typename Scalar, int Order>
void set_from_coo_arrays(Phasma::SparseMatrix<Scalar, Order> &mat,
                        nb::DRef<Phasma::Array<Phasma::Index>> idx_i,
                        nb::DRef<Phasma::Array<Phasma::Index>> idx_j,
                        nb::DRef<Phasma::Array<Scalar>> values,
                        Phasma::Index rows = Phasma::Index(0),
                        Phasma::Index cols = Phasma::Index(0)) {

    // Determine matrix size
    if (rows == 0 && cols == 0) {
        // No size provided, determine from indices
        rows = idx_i.maxCoeff() + 1;
        cols = idx_j.maxCoeff() + 1;
    } else if (rows != 0 && cols == 0) {
        // Only rows provided, assume square matrix
        cols = rows;
    }
    
    // Create ArrayTripletIterator object
    ArrayTripletIterator<Scalar> begin(idx_i, idx_j, values);
    ArrayTripletIterator<Scalar> end(idx_i, idx_j, values, values.rows());

    // Set the matrix from triplets
    mat.resize(rows, cols);
    mat.setFromTriplets(begin, end);
}
                         
template <typename Scalar, int Order>
void bind_sparse_matrix(nb::module_ &m, const std::string &class_name) {
    using SparseMatrix = Phasma::SparseMatrix<Scalar, Order>;
    constexpr int OppositeOrder = (Order == Eigen::RowMajor) ? Eigen::ColMajor : Eigen::RowMajor;
    using OppositeSparseMatrix = Phasma::SparseMatrix<Scalar, OppositeOrder>;

    nb::class_<SparseMatrix>(m, class_name.c_str())
        // Constructors
        .def(nb::init<>(), "Default constructor")
        .def("__init__",
            [](SparseMatrix* t,
               nb::DRef<Phasma::Array<Phasma::Index>> idx_i,
               nb::DRef<Phasma::Array<Phasma::Index>> idx_j,
               nb::DRef<Phasma::Array<Scalar>> values,
               Phasma::Index rows = Phasma::Index(0),
               Phasma::Index cols = Phasma::Index(0)) {
                new (t) SparseMatrix();
                set_from_coo_arrays<Scalar, Order>(*t, idx_i, idx_j, values, rows, cols);
               }, "idx_i"_a, "idx_j"_a, "values"_a, "rows"_a = Phasma::Index(0), "cols"_a = Phasma::Index(0),
               "Initialize matrix from arrays of triplets idx_i, idx_j, values in COO format.")

         // Conversion initializer
        .def("__init__",
             [](SparseMatrix* t, const OppositeSparseMatrix& other) {
                new (t) SparseMatrix();
                *t = other;
             },
             "other"_a, "Initialize matrix by converting from an opposite storage order matrix.")

        // ====================== Properties ======================
        .def_prop_ro("rows", &SparseMatrix::rows, "Number of rows")
        .def_prop_ro("cols", &SparseMatrix::cols, "Number of columns")
        .def_prop_ro("non_zeros", &SparseMatrix::nonZeros, "Number of non-zero elements")
        .def_prop_ro("is_compressed", &SparseMatrix::isCompressed, "Check if the matrix is compressed")

        // ====================== Member functions ======================
        .def("set_from_triplets", &set_from_coo_arrays<Scalar, Order>,
             "idx_i"_a, "idx_j"_a, "values"_a, "rows"_a = Phasma::Index(0), "cols"_a = Phasma::Index(0),
             "Set the matrix from arrays of triplets idx_i, idx_j, values in COO format.")

        // Operators
        .def("__repr__", [](const SparseMatrix &self) -> std::string {
            std::ostringstream oss;
            oss << self << std::endl;
            return oss.str();
        }, nb::is_operator())

        // ====================== Mathematical operators ======================
        // -------- Addition --------
        // SparseMatrix + SparseMatrix
        .def("__add__", [](const SparseMatrix &self, const SparseMatrix &other) -> SparseMatrix {
            return self + other;
        }, nb::is_operator())

        // SparseMatrix + Scalar - Not implemented in Eigen yet
        .def("__add__", [](const SparseMatrix &self, double s) -> SparseMatrix {
            SparseMatrix result = self; // Copy the matrix
            // Map the value array to a vector and add the scalar
            Eigen::Map<Phasma::Vector<Scalar>> v(result.valuePtr(), result.nonZeros());
            v += Phasma::Vector<Scalar>::Constant(result.nonZeros(), s);
            return result;
        }, nb::is_operator())
        .def("__radd__", [](const SparseMatrix &self, double s) -> SparseMatrix {
            SparseMatrix result = self; // Copy the matrix
            // Map the value array to a vector and add the scalar
            Eigen::Map<Phasma::Vector<Scalar>> v(result.valuePtr(), result.nonZeros());
            v += Phasma::Vector<Scalar>::Constant(result.nonZeros(), s);
            return result;
        }, nb::is_operator())

        // -------- Subtraction --------
        // SparseMatrix - SparseMatrix
        .def("__sub__", [](const SparseMatrix &self, const SparseMatrix &other) -> SparseMatrix {
            return self - other;
        }, nb::is_operator())

        // SparseMatrix - Scalar
        .def("__sub__", [](const SparseMatrix &self, double s) -> SparseMatrix {
            SparseMatrix result = self; // Copy the matrix
            // Map the value array to a vector and subtract the scalar
            Eigen::Map<Phasma::Vector<Scalar>> v(result.valuePtr(), result.nonZeros());
            v -= Phasma::Vector<Scalar>::Constant(result.nonZeros(), s);
            return result;
        }, nb::is_operator())
        .def("__rsub__", [](const SparseMatrix &self, double s) -> SparseMatrix {
            SparseMatrix result = self; // Copy the matrix
            // Map the value array to a vector and subtract the scalar
            Eigen::Map<Phasma::Vector<Scalar>> v(result.valuePtr(), result.nonZeros());
            v = Phasma::Vector<Scalar>::Constant(result.nonZeros(), s) - v;
            return result;
        }, nb::is_operator())

        // -------- Multiplication --------
        // SparseMatrix * Scalar
        .def("__mul__", [](const SparseMatrix &self, double s) -> SparseMatrix {
            return self * s;
        }, nb::is_operator())
        .def("__rmul__", [](const SparseMatrix &self, double s) -> SparseMatrix {
            return self * s;
        }, nb::is_operator())

        // SparseMatrix * Vector
        .def("__mul__", [](const SparseMatrix &self, nb::DRef<Phasma::Vector<Scalar>> v) -> Phasma::Vector<Scalar> {
            return self * v;
        }, nb::is_operator())

        // SparseMatrix * SparseMatrix
        .def("__mul__", [](const SparseMatrix &self, const SparseMatrix &other) -> SparseMatrix {
            return self * other;
        }, nb::is_operator())
        .def("__rmul__", [](const SparseMatrix &self, const SparseMatrix &other) -> SparseMatrix {
            return self * other;
        }, nb::is_operator())

        // SparseMatrix * DenseMatrix
        .def("__mul__", [](const SparseMatrix &self, nb::DRef<Phasma::Matrix<Scalar>> m) -> Phasma::Matrix<Scalar> {
            return self * m;
        }, nb::is_operator())
        .def("__rmul__", [](const SparseMatrix &self, nb::DRef<Phasma::Matrix<Scalar>> m) -> Phasma::Matrix<Scalar> {
            return self * m;
        }, nb::is_operator())

        // Sparse*Sparse cwise product
        .def("cwiseProduct", [](const SparseMatrix &self, const SparseMatrix &other) -> SparseMatrix {
            return self.cwiseProduct(other);
        }, "other"_a, "Element-wise product of two sparse matrices.")

        // Sparse*Dense cwise product
        .def("cwiseProduct", [](const SparseMatrix &self, nb::DRef<Phasma::Matrix<Scalar>> m) -> Phasma::Matrix<Scalar> {
            return self.cwiseProduct(m);
        }, "m"_a, "Element-wise product of a sparse matrix and a dense matrix.")

        // -------- Unary minus --------
        .def("__neg__", [](const SparseMatrix &self) -> SparseMatrix {
            return -self;
        }, nb::is_operator())

        // -------- Division --------
        // SparseMatrix / Scalar
        .def("__truediv__", [](const SparseMatrix &self, double s) -> SparseMatrix {
            return self / s;
        }, nb::is_operator())
        .def("__rtruediv__", [](const SparseMatrix &self, double s) -> SparseMatrix {
            return self / s;
        }, nb::is_operator())
        
        // ====================== Other operations ======================
        .def("diag", [](const SparseMatrix &self) -> Phasma::Vector<Scalar> {
            return self.diagonal();
        }, "Get the diagonal of the matrix.")

        .def("transpose", [](const SparseMatrix &self) -> SparseMatrix {
            return self.transpose();
        }, "Transpose the matrix.")

        .def("T_prod", [](const SparseMatrix &self, nb::DRef<Phasma::Vector<Scalar>> v) -> Phasma::Vector<Scalar> {
            return self.transpose() * v;
        }, "v"_a, "Perform the transpose matrix-vector product.")

        // Triangular solve
        .def("triangular_solve", [](const SparseMatrix &self, nb::DRef<Phasma::Vector<Scalar>> b, const Phasma::View& view) -> Phasma::Vector<Scalar> {
            if (view == Phasma::Upper) {
                return self.template triangularView<Phasma::Upper>().solve(b);
            } else if (view == Phasma::Lower) {
                return self.template triangularView<Phasma::Lower>().solve(b);
            } else {
                throw std::invalid_argument("Invalid value for 'view'. Use 'Upper' or 'Lower'.");
            }
        }, "b"_a, "view"_a, "Solve a triangular system of equations.")

        .def("triangular_prod", [](const SparseMatrix &self, nb::DRef<Phasma::Vector<Scalar>> b, const Phasma::View& view) -> Phasma::Vector<Scalar> {
            if (view == Phasma::Upper) {
                return self.template triangularView<Phasma::Upper>()*b;
            } else if (view == Phasma::Lower) {
                return self.template triangularView<Phasma::Lower>()*b;
            } else if (view == Phasma::StrictlyUpper) {
                return self.template triangularView<Phasma::StrictlyUpper>()*b;
            } else if (view == Phasma::StrictlyLower) {
                return self.template triangularView<Phasma::StrictlyLower>()*b;
            } else {
                throw std::invalid_argument("Invalid value for 'view'. Use 'Upper', 'Lower', 'StrictlyUpper' or 'StrictlyLower'.");
            }
        }, "b"_a, "view"_a, "Perform a triangular matrix-vector product.")

        // To dense
        .def("dense", [](const SparseMatrix &self) -> Phasma::Matrix<Scalar> {
            return Phasma::Matrix<Scalar>(self);
        }, "Convert the sparse matrix to a dense matrix.")
;}

} // namespace Phasma
#endif // PHASMA_SPARSE_MATRIX_MODULE_HPP