
/*
Phasma: fast sparse linear algebra for Python.
https://github.com/MarcelFerrari/phasma

File: scaler_module.hpp
Description: Exposes the Scaler class to Python using Nanobind.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
*/
#ifndef PHASMA_SCALER_MODULE_HPP
#define PHASMA_SCALER_MODULE_HPP

// Nanobind
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>

// Phasma
#include "phasma/types.hpp"
#include "phasma/scaler.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace Phasma::bindings {

template<typename Scalar>
void bind_scaler(nb::module_& m, const std::string& class_name) {
    using ScalerType = Phasma::Scaler<Scalar>;
    using CCSMatrix = typename ScalerType::CCSMatrix;
    using CRSMatrix = typename ScalerType::CRSMatrix;
    using Vector = typename ScalerType::Vector;

    nb::class_<ScalerType>(m, class_name.c_str())
        .def(nb::init<Phasma::ScalingType>(), "type"_a, "Create a Scaler object with the given scaling type.")
        .def("type", &ScalerType::type, "Get the scaling type of this Scaler object.")
        .def("scale", &ScalerType::scale_ccs, "A"_a, "Scale a CCS matrix.")
        .def("scale", &ScalerType::scale_crs, "A"_a, "Scale a CRS matrix.")
        .def("scale", &ScalerType::scale_vec, "v"_a, "Scale a vector.")
        .def("unscale", &ScalerType::unscale, "v"_a, "Unscale a vector.");
}

} // namespace Phasma::bindings

#endif // PHASMA_SCALER_MODULE_HPP