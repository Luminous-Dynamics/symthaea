# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# CMake build for liblitert_shim.so (Android NDK cross-compilation)
#
# Usage from Gradle (externalNativeBuild) or standalone:
#   cmake -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
#         -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 \
#         -DCMAKE_BUILD_TYPE=Release ..
#   cmake --build .

cmake_minimum_required(VERSION 3.18)
project(litert_shim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(litert_shim SHARED litert_shim.cpp)

# If LiteRT-LM SDK is available, link against it
if(DEFINED ENV{LITERT_LM_SDK})
    message(STATUS "LiteRT-LM SDK found: $ENV{LITERT_LM_SDK}")
    target_compile_definitions(litert_shim PRIVATE LITERT_LM_AVAILABLE)
    target_include_directories(litert_shim PRIVATE $ENV{LITERT_LM_SDK}/include)
    target_link_directories(litert_shim PRIVATE $ENV{LITERT_LM_SDK}/lib)
    target_link_libraries(litert_shim PRIVATE litert_lm)
else()
    message(STATUS "LiteRT-LM SDK not found — building stub mode")
endif()

# Android 15+ requires 16KB page alignment
if(ANDROID)
    target_link_options(litert_shim PRIVATE -Wl,-z,max-page-size=16384)
endif()
