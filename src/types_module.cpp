/*
Phasma: fast sparse linear algebra for Python.
https://github.com/MarcelFerrari/phasma

File: sparse_matrix_module.cpp
Description: Exposes Eigen's SparseMatrix class to Python using Nanobind.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

// Nanobind
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>

#include "phasma/types.hpp"

namespace nb = nanobind;
using namespace nb::literals;
namespace Phasma::bindings {
void init_types_module(nb::module_ & m){
    // Bind the Scale enum
    nb::enum_<Phasma::Scale>(m, "Scale")
        .value("Identity", Phasma::Scale::Identity)
        .value("Row", Phasma::Scale::Row)
        .value("Col", Phasma::Scale::Col)
        .value("Full", Phasma::Scale::Full)
        .export_values();

    nb::enum_<Phasma::Order>(m, "Order")
        .value("ColMajor", Phasma::Order::ColMajor)
        .value("RowMajor", Phasma::Order::RowMajor)
        .export_values();
    
    nb::enum_<Phasma::View>(m, "View")
        .value("Upper", Phasma::View::Upper)
        .value("Lower", Phasma::View::Lower)
        .value("StrictlyUpper", Phasma::View::StrictlyUpper)
        .value("StrictlyLower", Phasma::View::StrictlyLower)
        .export_values();
}
} // namespace Phasma::bindings

