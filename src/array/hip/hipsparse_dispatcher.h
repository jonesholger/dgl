/**
 *  Copyright (c) 2020 by Contributors
 * @file array/hip/dispatcher.cuh
 * @brief Templates to dispatch into different hipsparse routines based on the
 * type argument.
 */
#ifndef DGL_ARRAY_HIPSPARSE_DISPATCHER_H_
#define DGL_ARRAY_HIPSPARSE_DISPATCHER_H_

#include <hipsparse.h>
#include <dgl/runtime/c_runtime_api.h>

#include "bf16.h"
#include "fp16.h"

namespace dgl {
namespace aten {

/** @brief hipsparseXcsrgemm dispatcher */
template <typename DType>
struct CSRGEMM {
  template <typename... Args>
  static inline hipsparseStatus_t bufferSizeExt(Args... args) {
    BUG_IF_FAIL(false) << "This piece of code should not be reached.";
    return static_cast<hipsparseStatus_t>(0);
  }

  template <typename... Args>
  static inline hipsparseStatus_t nnz(Args... args) {
    return hipsparseXcsrgemm2Nnz(args...);
  }

  template <typename... Args>
  static inline hipsparseStatus_t compute(Args... args) {
    BUG_IF_FAIL(false) << "This piece of code should not be reached.";
    return static_cast<hipsparseStatus_t>(0);
  }
};

template <>
struct CSRGEMM<__half> {
  template <typename... Args>
  static inline hipsparseStatus_t bufferSizeExt(Args... args) {
    // TODO(ndickson): There is no hipsparseHcsrgemm2_bufferSizeExt, so a
    // different implementation would be required.
    LOG(FATAL) << "CSRGEMM::bufferSizeExt does not support dtype half (FP16).";
    return static_cast<hipsparseStatus_t>(0);
  }

  template <typename... Args>
  static inline hipsparseStatus_t nnz(Args... args) {
    return hipsparseXcsrgemm2Nnz(args...);
  }

  template <typename... Args>
  static inline hipsparseStatus_t compute(Args... args) {
    // TODO(ndickson): There is no hipsparseHcsrgemm2, so a different
    // implementation would be required.
    LOG(FATAL) << "CSRGEMM::compute does not support dtype half (FP16).";
    return static_cast<hipsparseStatus_t>(0);
  }
};

#if BF16_ENABLED
template <>
struct CSRGEMM<__nv_bfloat16> {
  template <typename... Args>
  static inline hipsparseStatus_t bufferSizeExt(Args... args) {
    // TODO(ndickson): There is no hipsparseHcsrgemm2_bufferSizeExt, so a
    // different implementation would be required.
    LOG(FATAL)
        << "CSRGEMM::bufferSizeExt does not support dtype bfloat16 (BF16).";
    return static_cast<hipsparseStatus_t>(0);
  }

  template <typename... Args>
  static inline hipsparseStatus_t nnz(Args... args) {
    return hipsparseXcsrgemm2Nnz(args...);
  }

  template <typename... Args>
  static inline hipsparseStatus_t compute(Args... args) {
    // TODO(ndickson): There is no hipsparseHcsrgemm2, so a different
    // implementation would be required.
    LOG(FATAL) << "CSRGEMM::compute does not support dtype bfloat16 (BF16).";
    return static_cast<hipsparseStatus_t>(0);
  }
};
#endif  // BF16_ENABLED

template <>
struct CSRGEMM<float> {
  template <typename... Args>
  static inline hipsparseStatus_t bufferSizeExt(Args... args) {
    return hipsparseScsrgemm2_bufferSizeExt(args...);
  }

  template <typename... Args>
  static inline hipsparseStatus_t nnz(Args... args) {
    return hipsparseXcsrgemm2Nnz(args...);
  }

  template <typename... Args>
  static inline hipsparseStatus_t compute(Args... args) {
    return hipsparseScsrgemm2(args...);
  }
};

template <>
struct CSRGEMM<double> {
  template <typename... Args>
  static inline hipsparseStatus_t bufferSizeExt(Args... args) {
    return hipsparseDcsrgemm2_bufferSizeExt(args...);
  }

  template <typename... Args>
  static inline hipsparseStatus_t nnz(Args... args) {
    return hipsparseXcsrgemm2Nnz(args...);
  }

  template <typename... Args>
  static inline hipsparseStatus_t compute(Args... args) {
    return hipsparseDcsrgemm2(args...);
  }
};

/** @brief hipsparseXcsrgeam dispatcher */
template <typename DType>
struct CSRGEAM {
  template <typename... Args>
  static inline hipsparseStatus_t bufferSizeExt(Args... args) {
    BUG_IF_FAIL(false) << "This piece of code should not be reached.";
    return static_cast<hipsparseStatus_t>(0);
  }

  template <typename... Args>
  static inline hipsparseStatus_t nnz(Args... args) {
    return hipsparseXcsrgeam2Nnz(args...);
  }

  template <typename... Args>
  static inline hipsparseStatus_t compute(Args... args) {
    BUG_IF_FAIL(false) << "This piece of code should not be reached.";
    return static_cast<hipsparseStatus_t>(0);
  }
};

template <>
struct CSRGEAM<__half> {
  template <typename... Args>
  static inline hipsparseStatus_t bufferSizeExt(Args... args) {
    // TODO(ndickson): There is no hipsparseHcsrgeam2_bufferSizeExt, so a
    // different implementation would be required.
    LOG(FATAL) << "CSRGEAM::bufferSizeExt does not support dtype half (FP16).";
    return static_cast<hipsparseStatus_t>(0);
  }

  template <typename... Args>
  static inline hipsparseStatus_t nnz(Args... args) {
    return hipsparseXcsrgeam2Nnz(args...);
  }

  template <typename... Args>
  static inline hipsparseStatus_t compute(Args... args) {
    // TODO(ndickson): There is no hipsparseHcsrgeam2, so a different
    // implementation would be required.
    LOG(FATAL) << "CSRGEAM::compute does not support dtype half (FP16).";
    return static_cast<hipsparseStatus_t>(0);
  }
};

#if BF16_ENABLED
template <>
struct CSRGEAM<__nv_bfloat16> {
  template <typename... Args>
  static inline hipsparseStatus_t bufferSizeExt(Args... args) {
    // TODO(ndickson): There is no hipsparseHcsrgeam2_bufferSizeExt, so a
    // different implementation would be required.
    LOG(FATAL)
        << "CSRGEAM::bufferSizeExt does not support dtype bfloat16 (BF16).";
    return static_cast<hipsparseStatus_t>(0);
  }

  template <typename... Args>
  static inline hipsparseStatus_t nnz(Args... args) {
    return hipsparseXcsrgeam2Nnz(args...);
  }

  template <typename... Args>
  static inline hipsparseStatus_t compute(Args... args) {
    // TODO(ndickson): There is no hipsparseHcsrgeam2, so a different
    // implementation would be required.
    LOG(FATAL) << "CSRGEAM::compute does not support dtype bfloat16 (BF16).";
    return static_cast<hipsparseStatus_t>(0);
  }
};
#endif  // BF16_ENABLED

template <>
struct CSRGEAM<float> {
  template <typename... Args>
  static inline hipsparseStatus_t bufferSizeExt(Args... args) {
    return hipsparseScsrgeam2_bufferSizeExt(args...);
  }

  template <typename... Args>
  static inline hipsparseStatus_t nnz(Args... args) {
    return hipsparseXcsrgeam2Nnz(args...);
  }

  template <typename... Args>
  static inline hipsparseStatus_t compute(Args... args) {
    return hipsparseScsrgeam2(args...);
  }
};

template <>
struct CSRGEAM<double> {
  template <typename... Args>
  static inline hipsparseStatus_t bufferSizeExt(Args... args) {
    return hipsparseDcsrgeam2_bufferSizeExt(args...);
  }

  template <typename... Args>
  static inline hipsparseStatus_t nnz(Args... args) {
    return hipsparseXcsrgeam2Nnz(args...);
  }

  template <typename... Args>
  static inline hipsparseStatus_t compute(Args... args) {
    return hipsparseDcsrgeam2(args...);
  }
};

};  // namespace aten
};  // namespace dgl

#endif  // DGL_ARRAY_HIPSPARSE_DISPATCHER_H_