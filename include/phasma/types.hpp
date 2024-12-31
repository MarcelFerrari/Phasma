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

#ifdef PHASMA_USE_64_BIT_INDEX
    using Index = long long int;
#else 
    using Index = int;
#endif

// Define aliases for Phasma types based on Eigen
template <typename Scalar>
using SparseVector = Eigen::SparseVector<Scalar, Eigen::ColMajor, Phasma::Index>;

template <typename Scalar>
using Vector = Eigen::Vector<Scalar, Eigen::Dynamic>;

template <typename Scalar, int Order = Eigen::ColMajor>
using SparseMatrix = Eigen::SparseMatrix<Scalar, Order, Phasma::Index>;

template <typename Scalar>
using Array = Eigen::Array<Scalar, Eigen::Dynamic, 1>;

} // namespace Phasma

#endif // PHASMA_TYPES_HPP
