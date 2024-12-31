/*
Phasma: fast sparse linear algebra for Python.
https://github.com/MarcelFerrari/phasma

File: scaler_module.cpp
Description: Bindings for the Scaler object.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/

#include <nanobind/nanobind.h>

#include "phasma/types.hpp"
#include "phasma/bindings/scaler_module.hpp"

namespace nb = nanobind;
namespace Phasma::bindings {
    void init_scaler_module(nb::module_& m) {
        // Bind the ScalingType enum
        nb::enum_<Phasma::ScalingType>(m, "ScalingType")
            .value("NONE", Phasma::ScalingType::NONE)
            .value("ROW", Phasma::ScalingType::ROW)
            .value("COL", Phasma::ScalingType::COL)
            .value("FULL", Phasma::ScalingType::FULL)
            .export_values();

        // Bind Scaler for double and float
        Phasma::bindings::bind_scaler<double>(m, "Scaler");
        Phasma::bindings::bind_scaler<float>(m, "Scaler_f");
    }
} // namespace Phasma::bindings