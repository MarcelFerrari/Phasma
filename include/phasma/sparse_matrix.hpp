/*
Phasma: fast sparse linear algebra for Python.
https://github.com/MarcelFerrari/phasma

File: sparse_matrix.hpp
Description: Exposes Eigen's SparseMatrix class to Python using Nanobind.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

See LICENSE file in the project root for full license information.
*/

#ifndef PHASMA_SPARSE_MATRIX_HPP
#define PHASMA_SPARSE_MATRIX_HPP

// Nanobind
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>

#include <Eigen/Sparse>
#include <string>
#include <iostream>

// Phasma
#include "phasma/utils.hpp"
#include "phasma/types.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace Phasma {

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
        .def("set_from_triplets", [](SparseMatrix &self,
                                     nb::DRef<Phasma::Array<Phasma::Index>> idx_i,
                                     nb::DRef<Phasma::Array<Phasma::Index>> idx_j,
                                     nb::DRef<Phasma::Array<Scalar>> values) -> void {
            
            // Determine matrix size
            Phasma::Index rows = idx_i.maxCoeff() + 1;
            Phasma::Index cols = idx_j.maxCoeff() + 1;

            // Create ArrayTripletIterator object
            ArrayTripletIterator<Scalar> begin(idx_i, idx_j, values);
            ArrayTripletIterator<Scalar> end(idx_i, idx_j, values, values.rows());

            // Set the matrix from triplets
            self.resize(rows, cols);
            self.setFromTriplets(begin, end);
        }, "idx_i"_a, "idx_j"_a, "values"_a,
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

        // -------- Division --------
        // SparseMatrix / Scalar
        .def("__truediv__", [](const SparseMatrix &self, Scalar s) -> SparseMatrix {
            return self / s;
        }, nb::is_operator())
        .def("__rtruediv__", [](const SparseMatrix &self, Scalar s) -> SparseMatrix {
            return self / s;
        }, nb::is_operator())        
;}

void init_sparse_matrix_module(nb::module_ & m){
    bind_sparse_matrix<double, Eigen::ColMajor>(m, "CCSDSpmat");
    bind_sparse_matrix<double, Eigen::RowMajor>(m, "CRSDSpmat");
    bind_sparse_matrix<float,  Eigen::ColMajor>(m, "CCSFSpmat");
    bind_sparse_matrix<float,  Eigen::RowMajor>(m, "CRSFSpmat");
}
    
} // namespace Phasma
#endif // PHASMA_SPARSE_MATRIX_HPP