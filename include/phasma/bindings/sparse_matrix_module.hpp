/*
Phasma: fast sparse linear algebra for Python.
https://github.com/MarcelFerrari/phasma

File: sparse_matrix.hpp
Description: Exposes Eigen's SparseMatrix class to Python using Nanobind.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

See LICENSE file in the project root for full license information.
*/

#ifndef PHASMA_SPARSE_MATRIX_MODULE_HPP
#define PHASMA_SPARSE_MATRIX_MODULE_HPP

// Nanobind
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>

#include <Eigen/Sparse>
#include <string>

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


// Set a sparse matrix from arrays of triplets in COO format
template <typename Scalar, int Order>
void set_from_coo_arrays(Phasma::SparseMatrix<Scalar, Order> &mat,
                        nb::DRef<Phasma::Array<Phasma::Index>> idx_i,
                        nb::DRef<Phasma::Array<Phasma::Index>> idx_j,
                        nb::DRef<Phasma::Array<Scalar>> values){

    // Determine matrix size
    Phasma::Index rows = idx_i.maxCoeff() + 1;
    Phasma::Index cols = idx_j.maxCoeff() + 1;

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

    nb::class_<SparseMatrix>(m, class_name.c_str())
        // Constructors
        .def(nb::init<>(), "Default constructor")
        .def(nb::init<Phasma::Index, Phasma::Index>(), "rows"_a, "cols"_a)

        // ====================== Properties ======================
        .def_prop_ro("rows", &SparseMatrix::rows, "Number of rows")
        .def_prop_ro("cols", &SparseMatrix::cols, "Number of columns")
        .def_prop_ro("non_zeros", &SparseMatrix::nonZeros, "Number of non-zero elements")
        .def_prop_ro("is_compressed", &SparseMatrix::isCompressed, "Check if the matrix is compressed")

        // ====================== Member functions ======================
        .def("setFromTriplets", &set_from_coo_arrays<Scalar, Order>,
             "idx_i"_a, "idx_j"_a, "values"_a,
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
        .def("__add__", [](const SparseMatrix &self, Scalar s) -> SparseMatrix {
            SparseMatrix result = self; // Copy the matrix
            // Map the value array to a vector and add the scalar
            Eigen::Map<Phasma::Vector<Scalar>> v(result.valuePtr(), result.nonZeros());
            v += Phasma::Vector<Scalar>::Constant(result.nonZeros(), s);
            return result;
        }, nb::is_operator())
        .def("__radd__", [](const SparseMatrix &self, Scalar s) -> SparseMatrix {
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
        .def("__sub__", [](const SparseMatrix &self, Scalar s) -> SparseMatrix {
            SparseMatrix result = self; // Copy the matrix
            // Map the value array to a vector and subtract the scalar
            Eigen::Map<Phasma::Vector<Scalar>> v(result.valuePtr(), result.nonZeros());
            v -= Phasma::Vector<Scalar>::Constant(result.nonZeros(), s);
            return result;
        }, nb::is_operator())
        .def("__rsub__", [](const SparseMatrix &self, Scalar s) -> SparseMatrix {
            SparseMatrix result = self; // Copy the matrix
            // Map the value array to a vector and subtract the scalar
            Eigen::Map<Phasma::Vector<Scalar>> v(result.valuePtr(), result.nonZeros());
            v = Phasma::Vector<Scalar>::Constant(result.nonZeros(), s) - v;
            return result;
        }, nb::is_operator())

        // -------- Multiplication --------
        // SparseMatrix * Scalar
        .def("__mul__", [](const SparseMatrix &self, Scalar s) -> SparseMatrix {
            return self * s;
        }, nb::is_operator())
        .def("__rmul__", [](const SparseMatrix &self, Scalar s) -> SparseMatrix {
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
        .def("__mul__", [](const SparseMatrix &self, nb::DRef<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> m) -> Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> {
            return self * m;
        }, nb::is_operator())
        .def("__rmul__", [](const SparseMatrix &self, nb::DRef<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> m) -> Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> {
            return self * m;
        }, nb::is_operator())

        // Sparse*Sparse cwise product
        .def("cwiseProduct", [](const SparseMatrix &self, const SparseMatrix &other) -> SparseMatrix {
            return self.cwiseProduct(other);
        }, "other"_a, "Element-wise product of two sparse matrices.")

        // Sparse*Dense cwise product
        .def("cwiseProduct", [](const SparseMatrix &self, nb::DRef<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> m) -> Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> {
            return self.cwiseProduct(m);
        }, "m"_a, "Element-wise product of a sparse matrix and a dense matrix.")

        // -------- Unary minus --------
        .def("__neg__", [](const SparseMatrix &self) -> SparseMatrix {
            return -self;
        }, nb::is_operator())
        // -------- Division --------
        // SparseMatrix / Scalar
        .def("__truediv__", [](const SparseMatrix &self, Scalar s) -> SparseMatrix {
            return self / s;
        }, nb::is_operator())
        .def("__rtruediv__", [](const SparseMatrix &self, Scalar s) -> SparseMatrix {
            return self / s;
        }, nb::is_operator())
        
        // ====================== Other operations ======================
        // Triangular solve
        .def("triangular_solve", [](const SparseMatrix &self, nb::DRef<Phasma::Vector<Scalar>> b, const std::string & uplo) -> Phasma::Vector<Scalar> {
            if (uplo == "U") {
                return self.template triangularView<Eigen::Upper>().solve(b);
            } else if (uplo == "L") {
                return self.template triangularView<Eigen::Lower>().solve(b);
            } else {
                throw std::invalid_argument("Invalid value for 'uplo'. Use 'U' or 'L'.");
            }
        }, "b"_a, "lower"_a = true, "Solve the triangular system Ax = b.")

        .def("triangular_prod", [](const SparseMatrix &self, nb::DRef<Phasma::Vector<Scalar>> b, const std::string & uplo) -> Phasma::Vector<Scalar> {
            if (uplo == "U") {
                return self.template triangularView<Eigen::Upper>()*b;
            } else if (uplo == "L") {
                return self.template triangularView<Eigen::Lower>()*b;
            } else if (uplo == "SU") {
                return self.template triangularView<Eigen::StrictlyUpper>()*b;
            } else if (uplo == "SL") {
                return self.template triangularView<Eigen::StrictlyLower>()*b;
            } else {
                throw std::invalid_argument("Invalid value for 'uplo'. Use 'U', 'L', 'SU' or 'SL'.");
            }
        }, "b"_a, "lower"_a = true, "Multiply the triangular matrix by a vector.")
;}

} // namespace Phasma
#endif // PHASMA_SPARSE_MATRIX_MODULE_HPP