/**
 *  Copyright (c) 2020 by Contributors
 * @file array/hip/spmm.cu
 * @brief SPMM C APIs and definitions.
 */
#include <dgl/array.h>

#include "../../runtime/hip/hip_common.h"
#include "./functor.h"
#include "./ge_spmm.h"
#include "./spmm.h"

namespace dgl {

using namespace hip;

namespace aten {

/**
 * @brief CUDA implementation of g-SpMM on Csr format.
 * @note use hipsparse if the reduce operator is `sum` and there is
 *       no broadcast, use dgl's kernel in other cases.
 */
template <int XPU, typename IdType, typename DType>
void SpMMCsr(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
  bool is_scalar_efeat = efeat.NumElements() == csr.indices->shape[0];
  bool use_efeat = op != "copy_lhs";

  if (reduce == "sum") {
    bool more_nnz = (csr.indices->shape[0] > csr.num_rows * csr.num_cols);
    if (op == "copy_lhs" && hipsparse_available<DType, IdType>(more_nnz)) {
      // hipsparse
      int64_t x_length = 1;
      for (int i = 1; i < ufeat->ndim; ++i) x_length *= ufeat->shape[i];
      hipsparseCsrmm2<DType, IdType>(
          ufeat->ctx, csr, static_cast<DType*>(ufeat->data), nullptr,
          static_cast<DType*>(out->data), x_length);
    } else if (
        op == "mul" && is_scalar_efeat &&
        hipsparse_available<DType, IdType>(more_nnz)) {
      // hipsparse
      int64_t x_length = 1;
      for (int i = 1; i < ufeat->ndim; ++i) x_length *= ufeat->shape[i];
      if (!IsNullArray(csr.data)) {
        efeat = IndexSelect(efeat, csr.data);
      }
      hipsparseCsrmm2<DType, IdType>(
          ufeat->ctx, csr, static_cast<DType*>(ufeat->data),
          static_cast<DType*>(efeat->data), static_cast<DType*>(out->data),
          x_length);
    } else {  // general kernel
      SWITCH_OP(op, Op, {
        hip::SpMMCsr<IdType, DType, Op, hip::reduce::Sum<IdType, DType> >(
            bcast, csr, ufeat, efeat, out, NullArray(), NullArray());
      });
    }
  } else if (reduce == "max") {
    SWITCH_OP(op, Op, {
      hip::SpMMCsr<IdType, DType, Op, hip::reduce::Max<IdType, DType> >(
          bcast, csr, ufeat, efeat, out, out_aux[0], out_aux[1]);
    });
  } else if (reduce == "min") {
    SWITCH_OP(op, Op, {
      hip::SpMMCsr<IdType, DType, Op, hip::reduce::Min<IdType, DType> >(
          bcast, csr, ufeat, efeat, out, out_aux[0], out_aux[1]);
    });
  } else {
    LOG(FATAL) << "Not implemented";
  }
}

/**
 * @brief CUDA implementation of g-SpMM on Coo format.
 */
template <int XPU, typename IdType, typename DType>
void SpMMCoo(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
  if (reduce == "sum") {
    SWITCH_OP(op, Op, {
      hip::SpMMCoo<IdType, DType, Op, hip::reduce::Sum<IdType, DType, true> >(
          bcast, coo, ufeat, efeat, out, NullArray(), NullArray());
    });
  } else if (reduce == "max") {
    SWITCH_OP(op, Op, {
      hip::SpMMCoo<IdType, DType, Op, hip::reduce::Max<IdType, DType, true> >(
          bcast, coo, ufeat, efeat, out, out_aux[0], out_aux[1]);
    });
  } else if (reduce == "min") {
    SWITCH_OP(op, Op, {
      hip::SpMMCoo<IdType, DType, Op, hip::reduce::Min<IdType, DType, true> >(
          bcast, coo, ufeat, efeat, out, out_aux[0], out_aux[1]);
    });
  } else {
    LOG(FATAL) << "Not implemented";
  }
}

#ifdef DGL_ENABLE_HALF
template void SpMMCsr<kDGLROCM, int32_t, __half>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCsr<kDGLROCM, int64_t, __half>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
#endif

#if BF16_ENABLED
template void SpMMCsr<kDGLROCM, int32_t, __nv_bfloat16>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCsr<kDGLROCM, int64_t, __nv_bfloat16>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
#endif  // BF16_ENABLED
template void SpMMCsr<kDGLROCM, int32_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCsr<kDGLROCM, int64_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCsr<kDGLROCM, int32_t, double>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCsr<kDGLROCM, int64_t, double>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);

#ifdef DGL_ENABLE_HALF
template void SpMMCoo<kDGLROCM, int32_t, __half>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCoo<kDGLROCM, int64_t, __half>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
#endif

#if BF16_ENABLED
template void SpMMCoo<kDGLROCM, int32_t, __nv_bfloat16>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCoo<kDGLROCM, int64_t, __nv_bfloat16>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
#endif  // BF16_ENABLED
template void SpMMCoo<kDGLROCM, int32_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCoo<kDGLROCM, int64_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCoo<kDGLROCM, int32_t, double>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCoo<kDGLROCM, int64_t, double>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);

}  // namespace aten
}  // namespace dgl
