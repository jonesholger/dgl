/**
 *  Copyright (c) 2021 by Contributors
 * @file hip_common.h
 * @brief Wrapper to place hipcub in dgl namespace.
 */

#ifndef DGL_ARRAY_HIP_DGL_CUB_H_
#define DGL_ARRAY_HIP_DGL_CUB_H_

// This should be defined in CMakeLists.txt
#if 0
#ifndef THRUST_CUB_WRAPPED_NAMESPACE
static_assert(false, "THRUST_CUB_WRAPPED_NAMESPACE must be defined for DGL.");
#endif
#endif

#include "hipcub/hipcub.hpp"

#endif  // DGL_ARRAY_HIP_DGL_CUB_H_
