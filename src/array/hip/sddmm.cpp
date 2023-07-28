/**
 *  Copyright (c) 2020 by Contributors
 * @file array/hip/sddmm.cu
 * @brief SDDMM C APIs and definitions.
 */
#include <dgl/array.h>

#include "./functor.h"
#include "./sddmm.h"

namespace dgl {
namespace aten {

/**
 * @brief CUDA implementation of g-SDDMM on Csr format.
 */
template <int XPU, typename IdType, typename DType>
void SDDMMCsr(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target) {
  SWITCH_OP(op, Op, {
    SWITCH_TARGET(lhs_target, rhs_target, LhsTarget, RhsTarget, {
      hip::SDDMMCsr<IdType, DType, Op, LhsTarget, RhsTarget>(
          bcast, csr, lhs, rhs, out);
    });
  });
}

/**
 * @brief CUDA implementation of g-SDDMM on Coo format.
 */
template <int XPU, typename IdType, typename DType>
void SDDMMCoo(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target) {
  SWITCH_OP(op, Op, {
    SWITCH_TARGET(lhs_target, rhs_target, LhsTarget, RhsTarget, {
      hip::SDDMMCoo<IdType, DType, Op, LhsTarget, RhsTarget>(
          bcast, coo, lhs, rhs, out);
    });
  });
}

#ifdef DGL_ENABLE_HALF
template void SDDMMCsr<kDGLCUDA, int32_t, __half>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target);
template void SDDMMCsr<kDGLCUDA, int64_t, __half>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target);
#endif

#if BF16_ENABLED
template void SDDMMCsr<kDGLCUDA, int32_t, __nv_bfloat16>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target);
template void SDDMMCsr<kDGLCUDA, int64_t, __nv_bfloat16>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target);
#endif  // BF16_ENABLED
template void SDDMMCsr<kDGLCUDA, int32_t, float>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target);
template void SDDMMCsr<kDGLCUDA, int64_t, float>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target);
template void SDDMMCsr<kDGLCUDA, int32_t, double>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target);
template void SDDMMCsr<kDGLCUDA, int64_t, double>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target);

#ifdef DGL_ENABLE_HALF
template void SDDMMCoo<kDGLCUDA, int32_t, __half>(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target);
template void SDDMMCoo<kDGLCUDA, int64_t, __half>(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target);
#endif

#if BF16_ENABLED
template void SDDMMCoo<kDGLCUDA, int32_t, __nv_bfloat16>(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target);
template void SDDMMCoo<kDGLCUDA, int64_t, __nv_bfloat16>(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target);
#endif  // BF16_ENABLED
template void SDDMMCoo<kDGLCUDA, int32_t, float>(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target);
template void SDDMMCoo<kDGLCUDA, int64_t, float>(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target);
template void SDDMMCoo<kDGLCUDA, int32_t, double>(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target);
template void SDDMMCoo<kDGLCUDA, int64_t, double>(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target);

}  // namespace aten
}  // namespace dgl
