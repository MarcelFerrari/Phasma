/*
Phasma: fast sparse linear algebra for Python.
https://github.com/MarcelFerrari/phasma

File: types.hpp
Description: Defines Phasma types based on Eigen.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

#ifndef PHASMA_TYPES_HPP
#define PHASMA_TYPES_HPP

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace Phasma {

// Index type based on 32 or 64 bits
// Define -DPHASMA_USE_64_BIT_INDEX=true in the CMakeLists.txt to use 64-bit indices
#ifdef PHASMA_USE_64_BIT_INDEX
    using Index = long long int;
#else 
    using Index = int;
#endif

// Define aliases for Phasma types based on Eigen
template <typename Scalar>
using Array = Eigen::Array<Scalar, Eigen::Dynamic, 1>;

template <typename Scalar>
using Vector = Eigen::Vector<Scalar, Eigen::Dynamic>;

template <typename Scalar>
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

template <typename Scalar>
using SparseVector = Eigen::SparseVector<Scalar, Eigen::ColMajor, Phasma::Index>;

template <typename Scalar, int Order = Eigen::ColMajor>
using SparseMatrix = Eigen::SparseMatrix<Scalar, Order, Phasma::Index>;

// Enum classes for Phasma types
enum Order {
    ColMajor = Eigen::ColMajor,
    RowMajor = Eigen::RowMajor
};

enum View {
    Upper = Eigen::Upper,
    Lower = Eigen::Lower,
    StrictlyUpper = Eigen::StrictlyUpper,
    StrictlyLower = Eigen::StrictlyLower
};

enum class ScalingType {
    None,
    Row,
    Col,
    Full,
};

} // namespace Phasma

#endif // PHASMA_TYPES_HPP
