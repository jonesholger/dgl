#include <hip/hip_runtime.h>
#include <hipsparse/hipsparse.h>
/**
 *  Copyright (c) 2020 by Contributors
 * @file array/hip/spmm.cuh
 * @brief SPMM CUDA kernel function header.
 */
#ifndef DGL_ARRAY_HIP_SPMM_H_
#define DGL_ARRAY_HIP_SPMM_H_

#include <dgl/bcast.h>

#include <limits>

#include "../../runtime/hip/hip_common.h"
#include "./utils.h"
#include "atomic.h"
#include "bf16.h"
#include "fp16.h"
#include "macro.h"

namespace dgl {

using namespace hip;

namespace aten {

/**
 * @brief Determine whether hipsparse SpMM function is applicable.
 */
template <typename DType, typename IdType>
inline bool hipsparse_available(bool more_nnz_than_matrix_size) {
//#if CUDART_VERSION < 11000
  if (std::is_same<IdType, int>::value &&
      (std::is_same<DType, float>::value || std::is_same<DType, double>::value))
    return true;
  return false;
/*
#else
  if (std::is_same<DType, __half>::value ||
      std::is_same<DType, __nv_bfloat16>::value)
    return false;  // hipsparse's SpMM on fp16 is slow, temporally disabled.
  // If the CSR matrix has more NNZ than matrix size, we should not use
  // hipsparse 11.1.
  return !more_nnz_than_matrix_size;
#endif
*/
}

namespace {

/** @brief Call cuBLAS geam API for transpose operation for float and double. */
template <typename DType>
hipblasStatus_t Xgeam(
    hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb,
    int m, int n, const DType* alpha, const DType* A, int lda,
    const DType* beta, const DType* B, int ldb, DType* C, int ldc) {
  LOG(FATAL) << "Not supported dtype";
  return HIPBLAS_STATUS_EXECUTION_FAILED;
}

#ifdef DGL_ENABLE_HALF
template <>
hipblasStatus_t Xgeam<__half>(
    hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb,
    int m, int n, const __half* alpha, const __half* A, int lda,
    const __half* beta, const __half* B, int ldb, __half* C, int ldc) {
  // TODO(ndickson): There is no cublasHgeam, so a different
  // implementation would be required.
  LOG(FATAL) << "Xgeam does not support dtype half (FP16)";
  return HIPBLAS_STATUS_EXECUTION_FAILED;
}

#if BF16_ENABLED
template <>
hipblasStatus_t Xgeam<__nv_bfloat16>(
    hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb,
    int m, int n, const __nv_bfloat16* alpha, const __nv_bfloat16* A, int lda,
    const __nv_bfloat16* beta, const __nv_bfloat16* B, int ldb,
    __nv_bfloat16* C, int ldc) {
  // TODO(ndickson): There is no cublasHgeam, so a different
  // implementation would be required.
  LOG(FATAL) << "Xgeam does not support dtype bfloat16 (BF16)";
  return HIPBLAS_STATUS_EXECUTION_FAILED;
}
#endif  // BF16_ENABLED
#endif // DGL_ENABLE_HALF

template <>
hipblasStatus_t Xgeam<float>(
    hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb,
    int m, int n, const float* alpha, const float* A, int lda,
    const float* beta, const float* B, int ldb, float* C, int ldc) {
  return hipblasSgeam(
      handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

template <>
hipblasStatus_t Xgeam<double>(
    hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb,
    int m, int n, const double* alpha, const double* A, int lda,
    const double* beta, const double* B, int ldb, double* C, int ldc) {
  return hipblasDgeam(
      handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

/**
 * @brief Transpose operator kernel implementation.
 * @note not efficient but it's not a bottleneck, used for float16 dtype.
 */
template <typename DType>
__global__ void _TransposeKernel(
    const DType* __restrict__ in, DType* __restrict__ out, int n, int m) {
  int i = blockIdx.x;
  for (int j = threadIdx.x; j < m; j += blockDim.x)
    out[i * m + j] = in[j * n + i];
}

/**
 * @brief Tranpose the input matrix.
 * @param row number of rows of input matrix.
 * @param col number of columns of input matrix.
 */
template <typename DType>
void _Transpose(const DType* in, DType* out, int row, int col) {
  DType alpha = 1., beta = 0.;
  auto* thr_entry = runtime::HIPThreadEntry::ThreadLocal();
  hipStream_t stream = runtime::getCurrentHIPStream();
  if (!thr_entry->hipblas_handle)
    HIPBLAS_CALL(hipblasCreate(&(thr_entry->hipblas_handle)));
  HIPBLAS_CALL(hipblasSetStream(thr_entry->hipblas_handle, stream));
  HIPBLAS_CALL(Xgeam<DType>(
      thr_entry->hipblas_handle, HIPBLAS_OP_T, HIPBLAS_OP_N, row, col, &alpha, in,
      col, &beta, nullptr, row, out, row));
}

/**
 * @brief Tranpose the input matrix for data type half.
 * @note cuBLAS has no geam API for half data type, fallback to our kernel.
 */
#ifdef DGL_ENABLE_HALF
template <>
void _Transpose<__half>(const __half* in, __half* out, int row, int col) {
  hipStream_t stream = runtime::getCurrentHIPStream();
  int nt = FindNumThreads(row);
  int nb = col;
  HIP_KERNEL_CALL(_TransposeKernel, nb, nt, 0, stream, in, out, col, row);
}

#if BF16_ENABLED
/**
 * @brief Tranpose the input matrix for data type half.
 * @note cuBLAS has no geam API for bf16 data type, fallback to our kernel.
 */
template <>
void _Transpose<__nv_bfloat16>(
    const __nv_bfloat16* in, __nv_bfloat16* out, int row, int col) {
  hipStream_t stream = runtime::getCurrentHIPStream();
  int nt = FindNumThreads(row);
  int nb = col;
  HIP_KERNEL_CALL(_TransposeKernel, nb, nt, 0, stream, in, out, col, row);
}
#endif  // BF16_ENABLED
#endif

//#if CUDART_VERSION < 11000
template <typename DType>
hipsparseStatus_t Xcsrmm2(
    hipsparseHandle_t handle, hipsparseOperation_t transA,
    hipsparseOperation_t transB, int m, int n, int k, int nnz,
    const DType* alpha, const hipsparseMatDescr_t descrA, const DType* csrValA,
    const int* csrRowPtrA, const int* csrColIndA, const DType* B, int ldb,
    const DType* beta, DType* C, int ldc) {
  LOG(INFO) << "Not supported dtype";
  return HIPSPARSE_STATUS_EXECUTION_FAILED;
}

template <>
hipsparseStatus_t Xcsrmm2<float>(
    hipsparseHandle_t handle, hipsparseOperation_t transA,
    hipsparseOperation_t transB, int m, int n, int k, int nnz,
    const float* alpha, const hipsparseMatDescr_t descrA, const float* csrValA,
    const int* csrRowPtrA, const int* csrColIndA, const float* B, int ldb,
    const float* beta, float* C, int ldc) {
  return hipsparseScsrmm2(
      handle, transA, transB, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA,
      csrColIndA, B, ldb, beta, C, ldc);
}

template <>
hipsparseStatus_t Xcsrmm2<double>(
    hipsparseHandle_t handle, hipsparseOperation_t transA,
    hipsparseOperation_t transB, int m, int n, int k, int nnz,
    const double* alpha, const hipsparseMatDescr_t descrA, const double* csrValA,
    const int* csrRowPtrA, const int* csrColIndA, const double* B, int ldb,
    const double* beta, double* C, int ldc) {
  return hipsparseDcsrmm2(
      handle, transA, transB, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA,
      csrColIndA, B, ldb, beta, C, ldc);
}

/** hipsparse implementation of SpMM on Csr format. */
template <typename DType, typename IdType>
void hipsparseCsrmm2(
    const DGLContext& ctx, const CSRMatrix& csr, const DType* B_data,
    const DType* A_data, DType* C_data, int x_length) {
  // We use csrmm2 to perform following operation:
  // C = A x B, where A is a sparse matrix in csr format, B is the dense matrix
  // for node feature tensor. However, since hipsparse only supports
  // column-major, while our tensor is stored in row-major, the actual
  // computation is: C = trans(A x trans(B)). Currently, we use cublasXgeam to
  // implement transposition and allocate intermediate workspace memory for
  // this.
  const int m = csr.num_rows;
  const int n = x_length;
  const int k = csr.num_cols;
  const int nnz = csr.indices->shape[0];
  const DType alpha = 1.0;
  const DType beta = 0.0;
  // device
  auto device = runtime::DeviceAPI::Get(ctx);
  auto* thr_entry = runtime::HIPThreadEntry::ThreadLocal();
  hipStream_t stream = runtime::getCurrentHIPStream();
  // allocate hipsparse handle if needed
  if (!thr_entry->hipsparse_handle) {
    HIPSPARSE_CALL(hipsparseCreate(&(thr_entry->hipsparse_handle)));
  }
  HIPSPARSE_CALL(hipsparseSetStream(thr_entry->hipsparse_handle, stream));
  // all one data array
  DType* valptr = nullptr;
  if (!A_data) {
    valptr =
        static_cast<DType*>(device->AllocWorkspace(ctx, nnz * sizeof(DType)));
    _Fill(valptr, nnz, static_cast<DType>(1.));
  }
//#if CUDART_VERSION >= 11000
  hipsparseSpMatDescr_t matA;
  hipsparseDnMatDescr_t matB, matC;
  constexpr auto dtype = HIP_dtype<DType>::value;
  constexpr auto idtype = hipsparse_idtype<IdType>::value;
  HIPSPARSE_CALL(hipsparseCreateCsr(
      &matA, m, k, nnz, static_cast<IdType*>(csr.indptr->data),
      static_cast<IdType*>(csr.indices->data),
      const_cast<DType*>(valptr ? valptr : A_data), idtype, idtype,
      HIPSPARSE_INDEX_BASE_ZERO, dtype));
  HIPSPARSE_CALL(hipsparseCreateDnMat(
      &matB, k, n, n, const_cast<DType*>(B_data), dtype, HIPSPARSE_ORDER_ROW));
  HIPSPARSE_CALL(
      hipsparseCreateDnMat(&matC, m, n, n, C_data, dtype, HIPSPARSE_ORDER_ROW));

  auto transA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
  auto transB = HIPSPARSE_OPERATION_NON_TRANSPOSE;
  size_t workspace_size;
  HIPSPARSE_CALL(hipsparseSpMM_bufferSize(
      thr_entry->hipsparse_handle, transA, transB, &alpha, matA, matB, &beta,
      matC, dtype, HIPSPARSE_SPMM_CSR_ALG2, &workspace_size));
  void* workspace = device->AllocWorkspace(ctx, workspace_size);
  HIPSPARSE_CALL(hipsparseSpMM(
      thr_entry->hipsparse_handle, transA, transB, &alpha, matA, matB, &beta,
      matC, dtype, HIPSPARSE_SPMM_CSR_ALG2, workspace));
  device->FreeWorkspace(ctx, workspace);

  HIPSPARSE_CALL(hipsparseDestroySpMat(matA));
  HIPSPARSE_CALL(hipsparseDestroyDnMat(matB));
  HIPSPARSE_CALL(hipsparseDestroyDnMat(matC));
/*
#else
  // allocate matrix for temporary transposed output
  DType* trans_out =
      static_cast<DType*>(device->AllocWorkspace(ctx, m * n * sizeof(DType)));

  hipsparseMatDescr_t descr;
  HIPSPARSE_CALL(hipsparseCreateMatDescr(&descr));
  HIPSPARSE_CALL(hipsparseSetMatType(descr, HIPSPARSE_MATRIX_TYPE_GENERAL));
  HIPSPARSE_CALL(hipsparseSetMatIndexBase(descr, HIPSPARSE_INDEX_BASE_ZERO));
  HIPSPARSE_CALL(Xcsrmm2<DType>(
      thr_entry->hipsparse_handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
      HIPSPARSE_OPERATION_TRANSPOSE, m, n, k, nnz, &alpha, descr,
      (valptr) ? valptr : A_data, static_cast<int32_t*>(csr.indptr->data),
      static_cast<int32_t*>(csr.indices->data), B_data, n, &beta, trans_out,
      m));
  HIPSPARSE_CALL(hipsparseDestroyMatDescr(descr));
  // transpose the output matrix
  _Transpose(trans_out, C_data, n, m);
  device->FreeWorkspace(ctx, trans_out);
#endif
*/
  if (valptr) device->FreeWorkspace(ctx, valptr);
}

/** hipsparse implementation of SpMM on Csr format. */
template <typename DType, typename IdType>
void hipsparseCsrmm2Hetero(
    const DGLContext& ctx, const CSRMatrix& csr, const DType* B_data,
    const DType* A_data, DType* C_data, int64_t x_length,
    hipStream_t strm_id) {
  // We use csrmm2 to perform following operation:
  // C = A x B, where A is a sparse matrix in csr format, B is the dense matrix
  // for node feature tensor. However, since hipsparse only supports
  // column-major, while our tensor is stored in row-major, the actual
  // computation is: C = trans(A x trans(B)). Currently, we use cublasXgeam to
  // implement transposition and allocate intermediate workspace memory for
  // this.
  int int_maxlimit = std::numeric_limits<int>::max();
  CHECK_GE(int_maxlimit, (csr.num_rows));
  CHECK_GE(int_maxlimit, csr.num_cols);
  CHECK_GE(int_maxlimit, csr.indices->shape[0]);
  const int m = csr.num_rows;
  const int n = x_length;
  const int k = csr.num_cols;
  const int nnz = csr.indices->shape[0];
  const DType alpha = 1.0;
  const DType beta = 1.0;
  // device
  auto device = runtime::DeviceAPI::Get(ctx);
  auto* thr_entry = runtime::HIPThreadEntry::ThreadLocal();
  // allocate hipsparse handle if needed
  if (!thr_entry->hipsparse_handle) {
    HIPSPARSE_CALL(hipsparseCreate(&(thr_entry->hipsparse_handle)));
  }
  HIPSPARSE_CALL(hipsparseSetStream(thr_entry->hipsparse_handle, strm_id));
  // all one data array
  DType* valptr = nullptr;
  if (!A_data) {
    valptr =
        static_cast<DType*>(device->AllocWorkspace(ctx, nnz * sizeof(DType)));
    _Fill(valptr, nnz, static_cast<DType>(1.));
  }
//#if CUDART_VERSION >= 11000
  hipsparseSpMatDescr_t matA;
  hipsparseDnMatDescr_t matB, matC;
  constexpr auto dtype = HIP_dtype<DType>::value;
  constexpr auto idtype = hipsparse_idtype<IdType>::value;
  HIPSPARSE_CALL(hipsparseCreateCsr(
      &matA, m, k, nnz, static_cast<IdType*>(csr.indptr->data),
      static_cast<IdType*>(csr.indices->data),
      const_cast<DType*>(valptr ? valptr : A_data), idtype, idtype,
      HIPSPARSE_INDEX_BASE_ZERO, dtype));
  HIPSPARSE_CALL(hipsparseCreateDnMat(
      &matB, k, n, n, const_cast<DType*>(B_data), dtype, HIPSPARSE_ORDER_ROW));
  HIPSPARSE_CALL(
      hipsparseCreateDnMat(&matC, m, n, n, C_data, dtype, HIPSPARSE_ORDER_ROW));

  auto transA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
  auto transB = HIPSPARSE_OPERATION_NON_TRANSPOSE;
  size_t workspace_size;
  HIPSPARSE_CALL(hipsparseSpMM_bufferSize(
      thr_entry->hipsparse_handle, transA, transB, &alpha, matA, matB, &beta,
      matC, dtype, HIPSPARSE_SPMM_CSR_ALG2, &workspace_size));
  void* workspace = device->AllocWorkspace(ctx, workspace_size);
  HIPSPARSE_CALL(hipsparseSpMM(
      thr_entry->hipsparse_handle, transA, transB, &alpha, matA, matB, &beta,
      matC, dtype, HIPSPARSE_SPMM_CSR_ALG2, workspace));
  device->FreeWorkspace(ctx, workspace);

  HIPSPARSE_CALL(hipsparseDestroySpMat(matA));
  HIPSPARSE_CALL(hipsparseDestroyDnMat(matB));
  HIPSPARSE_CALL(hipsparseDestroyDnMat(matC));
/*
#else
  hipsparseMatDescr_t descr;
  HIPSPARSE_CALL(hipsparseCreateMatDescr(&descr));
  HIPSPARSE_CALL(hipsparseSetMatType(descr, HIPSPARSE_MATRIX_TYPE_GENERAL));
  HIPSPARSE_CALL(hipsparseSetMatIndexBase(descr, HIPSPARSE_INDEX_BASE_ZERO));
  CHECK_EQ(sizeof(IdType), sizeof(int32_t));
  HIPSPARSE_CALL(Xcsrmm2<DType>(
      thr_entry->hipsparse_handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
      HIPSPARSE_OPERATION_TRANSPOSE, m, n, k, nnz, &alpha, descr,
      (valptr) ? valptr : A_data, static_cast<int32_t*>(csr.indptr->data),
      static_cast<int32_t*>(csr.indices->data), B_data, n, &beta, C_data, m));
  HIPSPARSE_CALL(hipsparseDestroyMatDescr(descr));
#endif
*/
  if (valptr) device->FreeWorkspace(ctx, valptr);
}

}  // namespace

#define SWITCH_OP(op, Op, ...)                                  \
  do {                                                          \
    if ((op) == "add") {                                        \
      typedef hip::binary::Add<DType> Op;                      \
      { __VA_ARGS__ }                                           \
    } else if ((op) == "sub") {                                 \
      typedef hip::binary::Sub<DType> Op;                      \
      { __VA_ARGS__ }                                           \
    } else if ((op) == "mul") {                                 \
      typedef hip::binary::Mul<DType> Op;                      \
      { __VA_ARGS__ }                                           \
    } else if ((op) == "div") {                                 \
      typedef hip::binary::Div<DType> Op;                      \
      { __VA_ARGS__ }                                           \
    } else if ((op) == "copy_lhs") {                            \
      typedef hip::binary::CopyLhs<DType> Op;                  \
      { __VA_ARGS__ }                                           \
    } else if ((op) == "copy_rhs") {                            \
      typedef hip::binary::CopyRhs<DType> Op;                  \
      { __VA_ARGS__ }                                           \
    } else {                                                    \
      LOG(FATAL) << "Unsupported SpMM binary operator: " << op; \
    }                                                           \
  } while (0)

namespace hip {

/**
 * @brief CUDA kernel of g-SpMM on Coo format.
 * @note it uses edge parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different edges. Threadblocks
 *       on the x-axis are responsible for the computation on different
 * positions in feature dimension. To avoid possible data hazards, it uses
 * atomic operators for reduction.
 */
template <
    typename Idx, typename DType, typename BinaryOp, typename ReduceOp,
    bool UseBcast = false, bool UseIdx = false>
__global__ void SpMMCooKernel(
    const DType* __restrict__ ufeat, const DType* __restrict__ efeat,
    DType* __restrict__ out, Idx* __restrict__ arg_u, Idx* __restrict__ arg_e,
    const Idx* __restrict__ row, const Idx* __restrict__ col,
    const Idx* __restrict__ edge_map, int64_t N, int64_t M, int64_t E,
    const int64_t* __restrict__ ubcast_off,
    const int64_t* __restrict__ ebcast_off, int64_t ufeat_len,
    int64_t efeat_len, int64_t out_len) {
  // SPMM with COO.
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = _ldg(row + ty);
    const Idx dst = _ldg(col + ty);
    const Idx eid = UseIdx ? _ldg(edge_map + ty) : ty;
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride_x = blockDim.x * gridDim.x;
    const DType* uoff = BinaryOp::use_lhs ? (ufeat + src * ufeat_len) : nullptr;
    const DType* eoff = BinaryOp::use_rhs ? (efeat + eid * efeat_len) : nullptr;
    DType* outoff = out + dst * out_len;
    while (tx < out_len) {
      const int64_t lhs_add = UseBcast ? ubcast_off[tx] : tx;
      const int64_t rhs_add = UseBcast ? ebcast_off[tx] : tx;
      DType val = BinaryOp::Call(uoff + lhs_add, eoff + rhs_add);
      Idx* arguoff = nullptr;  // arguoff is not used in SpMMCoo.
      Idx* argeoff = nullptr;  // argeoff is not used in SpMMCoo.
      ReduceOp::Call(outoff + tx, arguoff, argeoff, val, src, eid);
      tx += stride_x;
    }
    ty += stride_y;
  }
}

/**
 * @brief CUDA kernel to compute argu and arge in g-SpMM on Coo format.
 * @note it uses edge parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different edges. Threadblocks
 *       on the x-axis are responsible for the computation on different
 * positions in feature dimension.
 */
template <
    typename Idx, typename DType, typename BinaryOp, typename ReduceOp,
    bool UseBcast = false, bool UseIdx = false>
__global__ void ArgSpMMCooKernel(
    const DType* __restrict__ ufeat, const DType* __restrict__ efeat,
    DType* __restrict__ out, Idx* __restrict__ arg_u, Idx* __restrict__ arg_e,
    const Idx* __restrict__ row, const Idx* __restrict__ col,
    const Idx* __restrict__ edge_map, int64_t N, int64_t M, int64_t E,
    const int64_t* __restrict__ ubcast_off,
    const int64_t* __restrict__ ebcast_off, int64_t ufeat_len,
    int64_t efeat_len, int64_t out_len) {
  // SPMM with COO arg max/min.
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = _ldg(row + ty);
    const Idx dst = _ldg(col + ty);
    const Idx eid = UseIdx ? _ldg(edge_map + ty) : ty;
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride_x = blockDim.x * gridDim.x;
    const DType* uoff = BinaryOp::use_lhs ? (ufeat + src * ufeat_len) : nullptr;
    const DType* eoff = BinaryOp::use_rhs ? (efeat + eid * efeat_len) : nullptr;
    const DType* outoff = out + dst * out_len;
    Idx* arguoff = BinaryOp::use_lhs ? (arg_u + dst * out_len) : nullptr;
    Idx* argeoff = BinaryOp::use_rhs ? (arg_e + dst * out_len) : nullptr;
    while (tx < out_len) {
      int64_t lhs_add = UseBcast ? ubcast_off[tx] : tx;
      int64_t rhs_add = UseBcast ? ebcast_off[tx] : tx;
      DType val = BinaryOp::Call(uoff + lhs_add, eoff + rhs_add);
      ReduceOp::CallArg(tx, arguoff, argeoff, val, outoff[tx], src, eid);
      tx += stride_x;
    }
    ty += stride_y;
  }
}

/**
 * @brief CUDA kernel of g-SpMM on Csr format.
 * @note it uses node parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different destination nodes.
 *       Threadblocks on the x-axis are responsible for the computation on
 *       different positions in feature dimension.
 */
template <
    typename Idx, typename DType, typename BinaryOp, typename ReduceOp,
    bool UseBcast = false, bool UseIdx = false>
__global__ void SpMMCsrKernel(
    const DType* __restrict__ ufeat, const DType* __restrict__ efeat,
    DType* __restrict__ out, Idx* __restrict__ arg_u, Idx* __restrict__ arg_e,
    const Idx* __restrict__ indptr, const Idx* __restrict__ indices,
    const Idx* __restrict__ edge_map, int64_t num_rows, int64_t num_cols,
    const int64_t* __restrict__ ubcast_off,
    const int64_t* __restrict__ ebcast_off, int64_t ufeat_len,
    int64_t efeat_len, int64_t out_len) {
  // SPMM with CSR.
  int ty = blockIdx.x * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.x;
  const int stride_x = blockDim.x * gridDim.y;
  while (ty < num_rows) {
    int tx = blockIdx.y * blockDim.x + threadIdx.x;
    while (tx < out_len) {
      typename accum_dtype<DType>::type local_accum = ReduceOp::zero();
      Idx local_argu = 0, local_arge = 0;
      const int lhs_add = UseBcast ? ubcast_off[tx] : tx;
      const int rhs_add = UseBcast ? ebcast_off[tx] : tx;
      for (Idx i = indptr[ty]; i < indptr[ty + 1]; ++i) {
        const Idx eid = UseIdx ? _ldg(edge_map + i) : i;
        const Idx cid = _ldg(indices + i);
        const DType* uoff =
            BinaryOp::use_lhs ? (ufeat + cid * ufeat_len) : nullptr;
        const DType* eoff =
            BinaryOp::use_rhs ? (efeat + eid * efeat_len) : nullptr;
        DType out = BinaryOp::Call(uoff + lhs_add, eoff + rhs_add);
        ReduceOp::Call(&local_accum, &local_argu, &local_arge, out, cid, eid);
      }
      // The use of += is to compute cross-type reducing on heterogeneous graph
      // when reduce op is `sum`.
      //     C = SpMM(SpA, B) + C
      // Separate kernel `SpMMCmpCsrHeteroKernel` is used for max- and
      // min-reducer. It does not affect the output on homogeneous graph as
      // `out` is initialized to zero.
      out[ty * out_len + tx] += static_cast<DType>(local_accum);
      if (ReduceOp::require_arg && BinaryOp::use_lhs)
        arg_u[ty * out_len + tx] = local_argu;
      if (ReduceOp::require_arg && BinaryOp::use_rhs)
        arg_e[ty * out_len + tx] = local_arge;
      tx += stride_x;
    }
    ty += stride_y;
  }
}

/**
 * @brief CUDA kernel of SpMM-Min/Max on Csr format.
 * @note it uses node parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different destination nodes.
 *       Threadblocks on the x-axis are responsible for the computation on
 *       different positions in feature dimension.
 */
template <
    typename Idx, typename DType, typename BinaryOp, typename ReduceOp,
    bool UseBcast = false, bool UseIdx = false>
__global__ void SpMMCmpCsrHeteroKernel(
    const DType* __restrict__ ufeat, const DType* __restrict__ efeat,
    DType* __restrict__ out, Idx* __restrict__ arg_u, Idx* __restrict__ arg_e,
    Idx* __restrict__ arg_u_ntype, Idx* __restrict__ arg_e_etype,
    const Idx* __restrict__ indptr, const Idx* __restrict__ indices,
    const Idx* __restrict__ edge_map, int64_t num_rows, int64_t num_cols,
    const int64_t* __restrict__ ubcast_off,
    const int64_t* __restrict__ ebcast_off, int64_t ufeat_len,
    int64_t efeat_len, int64_t out_len, const int src_type, const int etype) {
  // SPMM with CSR.
  int ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  const int stride_x = blockDim.x * gridDim.x;
  while (ty < num_rows) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    while (tx < out_len) {
      using accum_type = typename accum_dtype<DType>::type;
      accum_type local_accum = static_cast<accum_type>(
          out[ty * out_len + tx]);  // ReduceOp::zero();
      Idx local_argu = 0, local_arge = 0;
      const int lhs_add = UseBcast ? ubcast_off[tx] : tx;
      const int rhs_add = UseBcast ? ebcast_off[tx] : tx;
      for (Idx i = indptr[ty]; i < indptr[ty + 1]; ++i) {
        const Idx eid = UseIdx ? _ldg(edge_map + i) : i;
        const Idx cid = _ldg(indices + i);
        const DType* uoff =
            BinaryOp::use_lhs ? (ufeat + cid * ufeat_len) : nullptr;
        const DType* eoff =
            BinaryOp::use_rhs ? (efeat + eid * efeat_len) : nullptr;
        DType tmp_out = BinaryOp::Call(uoff + lhs_add, eoff + rhs_add);
        ReduceOp::Call(
            &local_accum, &local_argu, &local_arge, tmp_out, cid, eid);
      }
      // Update output only when max/min values are different that original
      // output
      DType new_out = static_cast<DType>(local_accum);
      if (out[ty * out_len + tx] != new_out) {
        out[ty * out_len + tx] = new_out;
        if (ReduceOp::require_arg && BinaryOp::use_lhs) {
          arg_u[ty * out_len + tx] = local_argu;
          arg_u_ntype[ty * out_len + tx] = src_type;
        }
        if (ReduceOp::require_arg && BinaryOp::use_rhs) {
          arg_e[ty * out_len + tx] = local_arge;
          arg_e_etype[ty * out_len + tx] = etype;
        }
      }
      tx += stride_x;
    }
    ty += stride_y;
  }
}

/**
 * @brief CUDA implementation of g-SpMM on Coo format.
 * @param bcast Broadcast information.
 * @param coo The Coo matrix.
 * @param ufeat The feature on source nodes.
 * @param efeat The feature on edges.
 * @param out The result feature on destination nodes.
 * @param argu Arg-Min/Max on source nodes, which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 * @param arge Arg-Min/Max on edges. which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 */
template <typename Idx, typename DType, typename BinaryOp, typename ReduceOp>
void SpMMCoo(
    const BcastOff& bcast, const COOMatrix& coo, NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge) {
  /**
   * TODO(Xin): Disable half precision for SpMMCoo due to the round-off error.
   * We should use fp32 for the accumulation but it's hard to modify the 
   * current implementation.
   */
#if BF16_ENABLED
  if (std::is_same<DType, __half>::value ||
      std::is_same<DType, __nv_bfloat16>::value)
#else
  if (std::is_same<DType, __half>::value)
#endif  // BF16_ENABLED
    LOG(FATAL) << "SpMMCoo doesn't support half precision fow now. "
               << "Please use SpMMCsr instead by allowing the graph "
               << "materialize CSR/CSC formats.";
  const Idx *row = coo.row.Ptr<Idx>(), *col = coo.col.Ptr<Idx>(),
            *edge_map = coo.data.Ptr<Idx>();
  const DType *ufeat_data = ufeat.Ptr<DType>(),
              *efeat_data = efeat.Ptr<DType>();
  DType* out_data = out.Ptr<DType>();
  Idx *argu_data = argu.Ptr<Idx>(), *arge_data = arge.Ptr<Idx>();
  hipStream_t stream = runtime::getCurrentHIPStream();
  const int64_t N = coo.num_rows, M = coo.num_cols, E = coo.row->shape[0];

  int64_t *ubcast_off = nullptr, *ebcast_off = nullptr;
  int64_t len = bcast.out_len, lhs_len = bcast.lhs_len, rhs_len = bcast.rhs_len;

  int64_t out_size = out.NumElements();
  const int nt = FindNumThreads(out_size);
  const int nb = (out_size + nt - 1) / nt;
  HIP_KERNEL_CALL(
      _FillKernel, nb, nt, 0, stream, out_data, out_size, ReduceOp::zero());

  const int ntx = FindNumThreads(len);
  const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nbx = (len + ntx - 1) / ntx;
  const int nby = FindNumBlocks<'y'>((E + nty - 1) / nty);
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  const bool use_idx = !IsNullArray(coo.data);

  BCAST_IDX_CTX_SWITCH(bcast, use_idx, ufeat->ctx, ubcast_off, ebcast_off, {
    HIP_KERNEL_CALL(
        (SpMMCooKernel<Idx, DType, BinaryOp, ReduceOp, UseBcast, UseIdx>),
        nblks, nthrs, 0, stream, ufeat_data, efeat_data, out_data, argu_data,
        arge_data, row, col, edge_map, N, M, E, ubcast_off, ebcast_off, lhs_len,
        rhs_len, len);
    if (ReduceOp::require_arg) {
      HIP_KERNEL_CALL(
          (ArgSpMMCooKernel<Idx, DType, BinaryOp, ReduceOp, UseBcast, UseIdx>),
          nblks, nthrs, 0, stream, ufeat_data, efeat_data, out_data, argu_data,
          arge_data, row, col, edge_map, N, M, E, ubcast_off, ebcast_off,
          lhs_len, rhs_len, len);
    }
  });
}

/**
 * @brief CUDA implementation of g-SpMM on Csr format.
 * @param bcast Broadcast information.
 * @param csr The Csr matrix.
 * @param ufeat The feature on source nodes.
 * @param efeat The feature on edges.
 * @param out The result feature on destination nodes.
 * @param argu Arg-Min/Max on source nodes, which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 * @param arge Arg-Min/Max on edges. which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 */
template <typename Idx, typename DType, typename BinaryOp, typename ReduceOp>
void SpMMCsr(
    const BcastOff& bcast, const CSRMatrix& csr, NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge) {
  const Idx* indptr = csr.indptr.Ptr<Idx>();
  const Idx* indices = csr.indices.Ptr<Idx>();
  const Idx* edge_map = csr.data.Ptr<Idx>();
  const DType* ufeat_data = ufeat.Ptr<DType>();
  const DType* efeat_data = efeat.Ptr<DType>();
  DType* out_data = out.Ptr<DType>();
  Idx* argu_data = argu.Ptr<Idx>();
  Idx* arge_data = arge.Ptr<Idx>();

  hipStream_t stream = runtime::getCurrentHIPStream();

  int64_t *ubcast_off = nullptr, *ebcast_off = nullptr;
  int64_t len = bcast.out_len, lhs_len = bcast.lhs_len, rhs_len = bcast.rhs_len;
  const int ntx = FindNumThreads(len);
  const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nby = (len + ntx - 1) / ntx;
  const int nbx = FindNumBlocks<'x'>((csr.num_rows + nty - 1) / nty);
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  const bool use_idx = !IsNullArray(csr.data);

  BCAST_IDX_CTX_SWITCH(
      bcast, use_idx, ufeat->ctx, ubcast_off, ebcast_off,
      {HIP_KERNEL_CALL(
          (SpMMCsrKernel<Idx, DType, BinaryOp, ReduceOp, UseBcast, UseIdx>),
          nblks, nthrs, 0, stream, ufeat_data, efeat_data, out_data, argu_data,
          arge_data, indptr, indices, edge_map, csr.num_rows, csr.num_cols,
          ubcast_off, ebcast_off, lhs_len, rhs_len, len)});
}

/**
 * @brief CUDA kernel of SpMM-Min/Max on Csr format on heterogeneous graph.
 * @param bcast Broadcast information.
 * @param csr The Csr matrix.
 * @param ufeat The feature on source nodes.
 * @param efeat The feature on edges.
 * @param out The result feature on destination nodes.
 * @param argu Arg-Min/Max on source nodes, which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 * @param arge Arg-Min/Max on edges. which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 * @param argu_ntype Node type of the arg-Min/Max on source nodes, which refers
 * the source node types correspond to the minimum/maximum values of reduction
 * result on destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 * @param arge_etype Edge-type of the arg-Min/Max on edges. which refers the
 * source node indices correspond to the minimum/maximum values of reduction
 * result on destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 * @param src_type Node type of the source nodes of an etype
 * @param etype Edge type
 */
template <typename Idx, typename DType, typename BinaryOp, typename ReduceOp>
void SpMMCmpCsrHetero(
    const BcastOff& bcast, const CSRMatrix& csr, NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge, NDArray argu_ntype,
    NDArray arge_etype, const int src_type, const int etype) {
  const Idx* indptr = csr.indptr.Ptr<Idx>();
  const Idx* indices = csr.indices.Ptr<Idx>();
  const Idx* edge_map = csr.data.Ptr<Idx>();
  const DType* ufeat_data = ufeat.Ptr<DType>();
  const DType* efeat_data = efeat.Ptr<DType>();
  DType* out_data = out.Ptr<DType>();
  Idx* argu_data = argu.Ptr<Idx>();
  Idx* arge_data = arge.Ptr<Idx>();

  hipStream_t stream = runtime::getCurrentHIPStream();

  int64_t *ubcast_off = nullptr, *ebcast_off = nullptr;
  int64_t len = bcast.out_len, lhs_len = bcast.lhs_len, rhs_len = bcast.rhs_len;
  const int ntx = FindNumThreads(len);
  const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nbx = (len + ntx - 1) / ntx;
  const int nby = FindNumBlocks<'y'>((csr.num_rows + nty - 1) / nty);
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  const bool use_idx = !IsNullArray(csr.data);

  BCAST_IDX_CTX_SWITCH(
      bcast, use_idx, ufeat->ctx, ubcast_off, ebcast_off,
      {HIP_KERNEL_CALL(
          (SpMMCmpCsrHeteroKernel<
              Idx, DType, BinaryOp, ReduceOp, UseBcast, UseIdx>),
          nblks, nthrs, 0, stream, ufeat_data, efeat_data, out_data, argu_data,
          arge_data, static_cast<Idx*>(argu_ntype->data),
          static_cast<Idx*>(arge_etype->data), indptr, indices, edge_map,
          csr.num_rows, csr.num_cols, ubcast_off, ebcast_off, lhs_len, rhs_len,
          len, src_type, etype)});
}

}  // namespace hip
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_HIP_SPMM_H_