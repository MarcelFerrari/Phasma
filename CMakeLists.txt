# Phasma: fast sparse linear algebra for Python.
# https://github.com/MarcelFerrari/phasma

# Author: Marcel Ferrari
# Copyright (c) 2024 Marcel Ferrari. All rights reserved.

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

cmake_minimum_required(VERSION 3.15...3.27)
project(phasma)

# ---------- BUILD SETTINGS ----------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the default build type to optimized release if not specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Set the installation folder to "lib" within the source directory
set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/lib")

# ---------- PYTHON CONFIGURATION ----------
if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()
find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

# ---------- DEPENDENCY: NANOBIND ----------
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED)

# ---------- MODULE CONFIGURATION ----------
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add the phasma module
nanobind_add_module(phasma NOMINSIZE 
 src/bindings.cpp
 src/sparse_matrix_module.cpp
 src/scaler_module.cpp
 src/direct_solver_module.cpp
)

option(PHASMA_USE_64_BIT_INDEX "Use 64-bit index type." OFF)

if (PHASMA_USE_64_BIT_INDEX)
  # Use 64-bit index type for sparse matrices
  add_compile_definitions(PHASMA_USE_64_BIT_INDEX)
endif()

# ---------- DEPENDENCY: EIGEN ----------
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries(phasma PRIVATE Eigen3::Eigen)
message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")

# ---------- FIND OPENMP ----------
find_package(OpenMP REQUIRED)
target_link_libraries(phasma PRIVATE OpenMP::OpenMP_CXX)

# ---------- OPTIONAL DEPENDENCIES ----------
# Option for MKL
option(USE_MKL "Enable Intel MKL PARDISO." OFF)
if (USE_MKL)
  find_package(MKL REQUIRED)

  # Set MKL-specific flags
  set(MKL_COMPILER_FLAGS "" CACHE STRING "Custom MKL compiler flags.")
  set(MKL_LINKER_FLAGS "" CACHE STRING "Custom MKL linker flags.")

  # Check if MKL-specific flags are set
  if (NOT MKL_COMPILER_FLAGS OR NOT MKL_LINKER_FLAGS)    
    message(FATAL_ERROR "MKL compiler and linker flags must be set."
                        "Please set MKL_COMPILER_FLAGS and MKL_LINKER_FLAGS."
                        "Refer to the MKL link line advisor for more information: "
                        "https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html")
  endif()
    
  # Append MKL flags to the extra flags
  target_compile_definitions(phasma PRIVATE ${MKL_COMPILER_FLAGS})
  target_link_options(phasma PRIVATE ${MKL_LINKER_FLAGS})
  
  message(STATUS "MKL PARDISO enabled.")
  message(STATUS "Using MKL compiler flags: ${MKL_COMPILER_FLAGS}")
  message(STATUS "Using MKL linker flags: ${MKL_LINKER_FLAGS}")

  add_compile_definitions(PHASMA_PARDISO_SUPPORT)
endif()

# Option for SuiteSparse
option(USE_SUITESPARSE "Enable SuiteSparse." OFF)
if(USE_SUITESPARSE)
  # find_package(SuiteSparse REQUIRED NO_MODULE)
  find_package(UMFPACK REQUIRED NO_MODULE)
  find_package(KLU REQUIRED NO_MODULE)
  target_link_libraries(phasma PRIVATE SuiteSparse::UMFPACK SuiteSparse::KLU)
  message(STATUS "SuiteSparse enabled.")
  add_compile_definitions(PHASMA_SUITESPARSE_SUPPORT)
endif()

# ---------- INSTALLATION ----------
install(TARGETS phasma
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
