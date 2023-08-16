/**
 *  Copyright (c) 2020 by Contributors
 * @file array/hip/csr_mm.cu
 * @brief SpSpMM/SpGEMM C APIs and definitions.
 */
#include <dgl/array.h>
#include <dgl/runtime/device_api.h>

#include <limits>

#include "../../runtime/hip/hip_common.h"
#include "./hipsparse_dispatcher.h"
#include "./functor.h"
namespace dgl {

using namespace dgl::runtime;

namespace aten {
namespace hipsparse {

#if CUDART_VERSION >= 12000

/** @brief hipsparse implementation of SpGEMM on Csr format for CUDA 12.0+ */
template <typename DType, typename IdType>
std::pair<CSRMatrix, NDArray> hipsparseSpgemm(
    const CSRMatrix& A, const NDArray A_weights_array, const CSRMatrix& B,
    const NDArray B_weights_array) {
  // We use Spgemm (SpSpMM) to perform following operation:
  // C = A x B, where A, B and C are sparse matrices in csr format.
  const int nnzA = A.indices->shape[0];
  const int nnzB = B.indices->shape[0];
  const DType alpha = 1.0;
  const DType beta = 0.0;
  auto transA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
  auto transB = HIPSPARSE_OPERATION_NON_TRANSPOSE;
  // device
  auto ctx = A.indptr->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  auto* thr_entry = runtime::HIPThreadEntry::ThreadLocal();
  hipStream_t stream = runtime::getCurrentHIPStream();
  const DType* A_weights = A_weights_array.Ptr<DType>();
  const DType* B_weights = B_weights_array.Ptr<DType>();
  // allocate hipsparse handle if needed
  if (!thr_entry->hipsparse_handle) {
    HIPSPARSE_CALL(hipsparseCreate(&(thr_entry->hipsparse_handle)));
  }
  HIPSPARSE_CALL(hipsparseSetStream(thr_entry->hipsparse_handle, stream));
  // all one data array
  hipsparseSpMatDescr_t matA, matB, matC;
  IdArray dC_csrOffsets =
      IdArray::Empty({A.num_rows + 1}, A.indptr->dtype, A.indptr->ctx);
  IdType* dC_csrOffsets_data = dC_csrOffsets.Ptr<IdType>();
  constexpr auto idtype = hipsparse_idtype<IdType>::value;
  constexpr auto dtype = cuda_dtype<DType>::value;
  // Create sparse matrix A, B and C in CSR format
  HIPSPARSE_CALL(hipsparseCreateCsr(
      &matA, A.num_rows, A.num_cols, nnzA, A.indptr.Ptr<IdType>(),
      A.indices.Ptr<IdType>(),
      // hipsparseCreateCsr only accepts non-const pointers.
      const_cast<DType*>(A_weights),
      idtype, idtype, HIPSPARSE_INDEX_BASE_ZERO, dtype));
  HIPSPARSE_CALL(hipsparseCreateCsr(
      &matB, B.num_rows, B.num_cols, nnzB, B.indptr.Ptr<IdType>(),
      B.indices.Ptr<IdType>(),
      // hipsparseCreateCsr only accepts non-const pointers.
      const_cast<DType*>(B_weights),
      idtype, idtype, HIPSPARSE_INDEX_BASE_ZERO, dtype));
  HIPSPARSE_CALL(hipsparseCreateCsr(
      &matC, A.num_rows, B.num_cols, 0, nullptr, nullptr, nullptr, idtype,
      idtype, HIPSPARSE_INDEX_BASE_ZERO, dtype));
  // SpGEMM Computation
  hipsparseSpGEMMDescr_t spgemmDesc;
  hipsparseSpGEMMAlg_t alg = HIPSPARSE_SPGEMM_DEFAULT;

  HIPSPARSE_CALL(hipsparseSpGEMM_createDescr(&spgemmDesc));
  size_t workspace_size1 = 0, workspace_size2 = 0, workspace_size3 = 0;
  // ask bufferSize1 bytes for external memory
  HIPSPARSE_CALL(hipsparseSpGEMM_workEstimation(
      thr_entry->hipsparse_handle, transA, transB, &alpha, matA, matB, &beta,
      matC, dtype, alg, spgemmDesc, &workspace_size1,
      NULL));
  void* workspace1 = (device->AllocWorkspace(ctx, workspace_size1));
  // inspect the matrices A and B to understand the memory requiremnent
  hipsparseStatus_t e =
    hipsparseSpGEMM_workEstimation(thr_entry->hipsparse_handle, transA,
                                  transB, &alpha, matA, matB, &beta,
                                  matC, dtype, alg, spgemmDesc,
                                  &workspace_size1, workspace1);
  // HIPSPARSE_SPGEMM_DEFAULT not support getting num_prods > 2^31 -1
  // and throws insufficient memory error within workEstimation call
  if (e == HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES) {
    // fall back to ALG2 to estimate num_prods
    alg = hipsparse_SPGEMM_ALG2;
    device->FreeWorkspace(ctx, workspace1);
    // rerun hipsparseSpGEMM_workEstimation
    HIPSPARSE_CALL(hipsparseSpGEMM_workEstimation(thr_entry->hipsparse_handle,
                                                transA, transB, &alpha, matA,
                                                matB, &beta, matC, dtype, alg,
                                                spgemmDesc, &workspace_size1,
                                                NULL));
    workspace1 = (device->AllocWorkspace(ctx, workspace_size1));
    HIPSPARSE_CALL(hipsparseSpGEMM_workEstimation(thr_entry->hipsparse_handle,
                                                transA, transB, &alpha, matA,
                                                matB, &beta, matC, dtype, alg,
                                                spgemmDesc, &workspace_size1,
                                                workspace1));
  } else {
    CHECK(e == HIPSPARSE_STATUS_SUCCESS) << "hipsparse ERROR in SpGEMM: " << e;
  }

  // get the number of intermediate products required for SpGEMM compute
  // num_prods indicates device memory consumption for SpGEMM if using ALG2/3
  int64_t num_prods;
  HIPSPARSE_CALL(hipsparseSpGEMM_getNumProducts(spgemmDesc, &num_prods));

  // assume free GPU mem at least ~15G for below heuristics to work
  // user-defined medium problem size (below will use DEFAULT)
  int64_t MEDIUM_NUM_PRODUCTS = 400000000;  // 400*1000*1000;
  // user-defined large problem size (above will use ALG3)
  int64_t LARGE_NUM_PRODUCTS  = 800000000;  // 800*1000*1000;

  // switch to ALG2/ALG3 for medium & large problem size
  if (alg == HIPSPARSE_SPGEMM_DEFAULT && num_prods > MEDIUM_NUM_PRODUCTS) {
    // use ALG3 for very large problem
    alg = num_prods > LARGE_NUM_PRODUCTS ? hipsparse_SPGEMM_ALG3 :
      hipsparse_SPGEMM_ALG2;

    device->FreeWorkspace(ctx, workspace1);
    // rerun hipsparseSpGEMM_workEstimation
    HIPSPARSE_CALL(hipsparseSpGEMM_workEstimation(thr_entry->hipsparse_handle,
                                                transA, transB, &alpha, matA,
                                                matB, &beta, matC, dtype, alg,
                                                spgemmDesc, &workspace_size1,
                                                NULL));
    workspace1 = (device->AllocWorkspace(ctx, workspace_size1));
    HIPSPARSE_CALL(hipsparseSpGEMM_workEstimation(thr_entry->hipsparse_handle,
                                                transA, transB, &alpha, matA,
                                                matB, &beta, matC, dtype, alg,
                                                spgemmDesc, &workspace_size1,
                                                workspace1));
  } else if (alg == hipsparse_SPGEMM_ALG2 && num_prods > LARGE_NUM_PRODUCTS) {
    // no need to rerun hipsparseSpGEMM_workEstimation between ALG2 and ALG3
    alg = hipsparse_SPGEMM_ALG3;
  }

  if (alg == hipsparse_SPGEMM_ALG2 || alg == hipsparse_SPGEMM_ALG3) {
    // estimate memory for ALG2/ALG3; note chunk_fraction is only used by ALG3
    // reduce chunk_fraction if crash due to mem., but it trades off speed
    float chunk_fraction = num_prods < 4 * LARGE_NUM_PRODUCTS ? 0.15 : 0.05;
    HIPSPARSE_CALL(hipsparseSpGEMM_estimateMemory(thr_entry->hipsparse_handle,
                                                transA, transB, &alpha, matA,
                                                matB, &beta, matC, dtype, alg,
                                                spgemmDesc, chunk_fraction,
                                                &workspace_size3,
                                                NULL, NULL));
    void* workspace3 = (device->AllocWorkspace(ctx, workspace_size3));
    HIPSPARSE_CALL(hipsparseSpGEMM_estimateMemory(thr_entry->hipsparse_handle,
                                                transA, transB, &alpha, matA,
                                                matB, &beta, matC, dtype, alg,
                                                spgemmDesc, chunk_fraction,
                                                &workspace_size3,
                                                workspace3, &workspace_size2));
    device->FreeWorkspace(ctx, workspace3);
  } else {
    HIPSPARSE_CALL(hipsparseSpGEMM_compute(thr_entry->hipsparse_handle,
                                         transA, transB, &alpha, matA,
                                         matB, &beta, matC, dtype, alg,
                                         spgemmDesc, &workspace_size2,
                                         NULL));
  }
  // ask bufferSize2 bytes for external memory
  void* workspace2 = device->AllocWorkspace(ctx, workspace_size2);
  // compute the intermediate product of A * B
  HIPSPARSE_CALL(hipsparseSpGEMM_compute(
      thr_entry->hipsparse_handle, transA, transB, &alpha, matA, matB, &beta,
      matC, dtype, alg, spgemmDesc, &workspace_size2,
      workspace2));
  // get matrix C non-zero entries C_nnz1
  int64_t C_num_rows1, C_num_cols1, C_nnz1;
  HIPSPARSE_CALL(
      hipsparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_nnz1));
  IdArray dC_columns = IdArray::Empty({C_nnz1}, A.indptr->dtype, A.indptr->ctx);
  NDArray dC_weights = NDArray::Empty(
      {C_nnz1}, A_weights_array->dtype, A.indptr->ctx);
  IdType* dC_columns_data = dC_columns.Ptr<IdType>();
  DType* dC_weights_data = dC_weights.Ptr<DType>();
  // update matC with the new pointers
  HIPSPARSE_CALL(hipsparseCsrSetPointers(
      matC, dC_csrOffsets_data, dC_columns_data, dC_weights_data));
  // copy the final products to the matrix C
  HIPSPARSE_CALL(hipsparseSpGEMM_copy(
      thr_entry->hipsparse_handle, transA, transB, &alpha, matA, matB, &beta,
      matC, dtype, alg, spgemmDesc));

  device->FreeWorkspace(ctx, workspace1);
  device->FreeWorkspace(ctx, workspace2);
  // destroy matrix/vector descriptors
  HIPSPARSE_CALL(hipsparseSpGEMM_destroyDescr(spgemmDesc));
  HIPSPARSE_CALL(hipsparseDestroySpMat(matA));
  HIPSPARSE_CALL(hipsparseDestroySpMat(matB));
  HIPSPARSE_CALL(hipsparseDestroySpMat(matC));
  return {
      CSRMatrix(
          A.num_rows, B.num_cols, dC_csrOffsets, dC_columns,
          NullArray(dC_csrOffsets->dtype, dC_csrOffsets->ctx)),
      dC_weights};
}

#else  // CUDART_VERSION < 12000

/** @brief hipsparse implementation of SpGEMM on Csr format for older CUDA
 * versions */
template <typename DType, typename IdType>
std::pair<CSRMatrix, NDArray> hipsparseSpgemm(
    const CSRMatrix& A, const NDArray A_weights_array, const CSRMatrix& B,
    const NDArray B_weights_array) {
  int nnzC;
  csrgemm2Info_t info = nullptr;
  size_t workspace_size;
  const DType alpha = 1.;
  const int nnzA = A.indices->shape[0];
  const int nnzB = B.indices->shape[0];
  const int m = A.num_rows;
  const int n = A.num_cols;
  const int k = B.num_cols;
  auto ctx = A.indptr->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  auto* thr_entry = runtime::HIPThreadEntry::ThreadLocal();
  hipStream_t stream = runtime::getCurrentHIPStream();
  auto idtype = A.indptr->dtype;
  auto dtype = A_weights_array->dtype;
  const DType* A_weights = A_weights_array.Ptr<DType>();
  const DType* B_weights = B_weights_array.Ptr<DType>();
  if (!thr_entry->hipsparse_handle) {
    HIPSPARSE_CALL(hipsparseCreate(&(thr_entry->hipsparse_handle)));
  }
  HIPSPARSE_CALL(hipsparseSetStream(thr_entry->hipsparse_handle, stream));
  HIPSPARSE_CALL(hipsparseSetPointerMode(
      thr_entry->hipsparse_handle, HIPSPARSE_POINTER_MODE_HOST));

  HIPSPARSE_CALL(hipsparseCreateCsrgemm2Info(&info));

  hipsparseMatDescr_t matA, matB, matC, matD;
  HIPSPARSE_CALL(hipsparseCreateMatDescr(&matA));
  HIPSPARSE_CALL(hipsparseCreateMatDescr(&matB));
  HIPSPARSE_CALL(hipsparseCreateMatDescr(&matC));
  HIPSPARSE_CALL(hipsparseCreateMatDescr(&matD));  // needed even if D is null

  HIPSPARSE_CALL(CSRGEMM<DType>::bufferSizeExt(
      thr_entry->hipsparse_handle, m, n, k, &alpha, matA, nnzA,
      A.indptr.Ptr<IdType>(), A.indices.Ptr<IdType>(), matB, nnzB,
      B.indptr.Ptr<IdType>(), B.indices.Ptr<IdType>(), nullptr, matD, 0,
      nullptr, nullptr, info, &workspace_size));

  void* workspace = device->AllocWorkspace(ctx, workspace_size);
  IdArray C_indptr = IdArray::Empty({m + 1}, idtype, ctx);
  HIPSPARSE_CALL(CSRGEMM<DType>::nnz(
      thr_entry->hipsparse_handle, m, n, k, matA, nnzA, A.indptr.Ptr<IdType>(),
      A.indices.Ptr<IdType>(), matB, nnzB, B.indptr.Ptr<IdType>(),
      B.indices.Ptr<IdType>(), matD, 0, nullptr, nullptr, matC,
      C_indptr.Ptr<IdType>(), &nnzC, info, workspace));

  IdArray C_indices = IdArray::Empty({nnzC}, idtype, ctx);
  NDArray C_weights = NDArray::Empty({nnzC}, dtype, ctx);
  HIPSPARSE_CALL(CSRGEMM<DType>::compute(
      thr_entry->hipsparse_handle, m, n, k, &alpha, matA, nnzA, A_weights,
      A.indptr.Ptr<IdType>(), A.indices.Ptr<IdType>(), matB, nnzB, B_weights,
      B.indptr.Ptr<IdType>(), B.indices.Ptr<IdType>(), nullptr, matD, 0,
      nullptr, nullptr, nullptr, matC, C_weights.Ptr<DType>(),
      C_indptr.Ptr<IdType>(), C_indices.Ptr<IdType>(), info, workspace));

  device->FreeWorkspace(ctx, workspace);
  HIPSPARSE_CALL(hipsparseDestroyCsrgemm2Info(info));
  HIPSPARSE_CALL(hipsparseDestroyMatDescr(matA));
  HIPSPARSE_CALL(hipsparseDestroyMatDescr(matB));
  HIPSPARSE_CALL(hipsparseDestroyMatDescr(matC));
  HIPSPARSE_CALL(hipsparseDestroyMatDescr(matD));

  return {
      CSRMatrix(
          m, k, C_indptr, C_indices, NullArray(C_indptr->dtype, C_indptr->ctx)),
      C_weights};
}

#endif  // CUDART_VERSION >= 12000
}  // namespace hipsparse

template <int XPU, typename IdType, typename DType>
std::pair<CSRMatrix, NDArray> CSRMM(
    const CSRMatrix& A, NDArray A_weights, const CSRMatrix& B,
    NDArray B_weights) {
  auto ctx = A.indptr->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  CSRMatrix newA, newB;
  bool cast = false;

  // Cast 64 bit indices to 32 bit.
  if (A.indptr->dtype.bits == 64) {
    newA = CSRMatrix(
        A.num_rows, A.num_cols, AsNumBits(A.indptr, 32),
        AsNumBits(A.indices, 32), AsNumBits(A.data, 32));
    newB = CSRMatrix(
        B.num_rows, B.num_cols, AsNumBits(B.indptr, 32),
        AsNumBits(B.indices, 32), AsNumBits(B.data, 32));
    cast = true;
  }

  // Reorder weights if A or B has edge IDs
  NDArray newA_weights, newB_weights;
  if (CSRHasData(A)) newA_weights = IndexSelect(A_weights, A.data);
  if (CSRHasData(B)) newB_weights = IndexSelect(B_weights, B.data);

  auto result = hipsparse::hipsparseSpgemm<DType, int32_t>(
      cast ? newA : A, CSRHasData(A) ? newA_weights : A_weights,
      cast ? newB : B, CSRHasData(B) ? newB_weights : B_weights);

  // Cast 32 bit indices back to 64 bit if necessary
  if (cast) {
    CSRMatrix C = result.first;
    return {
        CSRMatrix(
            C.num_rows, C.num_cols, AsNumBits(C.indptr, 64),
            AsNumBits(C.indices, 64), AsNumBits(C.data, 64)),
        result.second};
  } else {
    return result;
  }
}

#ifdef DGL_ENABLE_HALF
template std::pair<CSRMatrix, NDArray> CSRMM<kDGLROCM, int32_t, __half>(
    const CSRMatrix&, NDArray, const CSRMatrix&, NDArray);
template std::pair<CSRMatrix, NDArray> CSRMM<kDGLROCM, int64_t, __half>(
    const CSRMatrix&, NDArray, const CSRMatrix&, NDArray);
#endif
#if BF16_ENABLED
template std::pair<CSRMatrix, NDArray> CSRMM<kDGLROCM, int32_t, __nv_bfloat16>(
    const CSRMatrix&, NDArray, const CSRMatrix&, NDArray);
template std::pair<CSRMatrix, NDArray> CSRMM<kDGLROCM, int64_t, __nv_bfloat16>(
    const CSRMatrix&, NDArray, const CSRMatrix&, NDArray);
#endif  // BF16_ENABLED
template std::pair<CSRMatrix, NDArray> CSRMM<kDGLROCM, int32_t, float>(
    const CSRMatrix&, NDArray, const CSRMatrix&, NDArray);
template std::pair<CSRMatrix, NDArray> CSRMM<kDGLROCM, int64_t, float>(
    const CSRMatrix&, NDArray, const CSRMatrix&, NDArray);
template std::pair<CSRMatrix, NDArray> CSRMM<kDGLROCM, int32_t, double>(
    const CSRMatrix&, NDArray, const CSRMatrix&, NDArray);
template std::pair<CSRMatrix, NDArray> CSRMM<kDGLROCM, int64_t, double>(
    const CSRMatrix&, NDArray, const CSRMatrix&, NDArray);

}  // namespace aten
}  // namespace dgl
