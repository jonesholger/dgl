#include "hip/hip_runtime.h"
/**
 *  Copyright (c) 2020 by Contributors
 * @file array/hip/csr_sort.cc
 * @brief Sort CSR index
 */
#include <dgl/array.h>

#include "../../runtime/hip/hip_common.h"
#include "./dgl_cub.h"
#include "./utils.h"

namespace dgl {

using runtime::NDArray;

namespace aten {
namespace impl {

/**
 * @brief Check whether each row is sorted.
 */
template <typename IdType>
__global__ void _SegmentIsSorted(
    const IdType* indptr, const IdType* indices, int64_t num_rows,
    int8_t* flags) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;
  while (tx < num_rows) {
    bool f = true;
    for (IdType i = indptr[tx] + 1; f && i < indptr[tx + 1]; ++i) {
      f = (indices[i - 1] <= indices[i]);
    }
    flags[tx] = static_cast<int8_t>(f);
    tx += stride_x;
  }
}

template <DGLDeviceType XPU, typename IdType>
bool CSRIsSorted(CSRMatrix csr) {
  const auto& ctx = csr.indptr->ctx;
  hipStream_t stream = runtime::getCurrentHIPStream();
  auto device = runtime::DeviceAPI::Get(ctx);
  // We allocate a workspace of num_rows bytes. It wastes a little bit memory
  // but should be fine.
  int8_t* flags =
      static_cast<int8_t*>(device->AllocWorkspace(ctx, csr.num_rows));
  const int nt = hip::FindNumThreads(csr.num_rows);
  const int nb = (csr.num_rows + nt - 1) / nt;
  HIP_KERNEL_CALL(
      _SegmentIsSorted, nb, nt, 0, stream, csr.indptr.Ptr<IdType>(),
      csr.indices.Ptr<IdType>(), csr.num_rows, flags);
  bool ret = hip::AllTrue(flags, csr.num_rows, ctx);
  device->FreeWorkspace(ctx, flags);
  return ret;
}

template bool CSRIsSorted<kDGLROCM, int32_t>(CSRMatrix csr);
template bool CSRIsSorted<kDGLROCM, int64_t>(CSRMatrix csr);

template <DGLDeviceType XPU, typename IdType>
void CSRSort_(CSRMatrix* csr) {
  LOG(FATAL) << "Unreachable codes";
}

template <>
void CSRSort_<kDGLROCM, int32_t>(CSRMatrix* csr) {
  auto* thr_entry = runtime::HIPThreadEntry::ThreadLocal();
  auto device = runtime::DeviceAPI::Get(csr->indptr->ctx);
  hipStream_t stream = runtime::getCurrentHIPStream();
  // allocate hipsparse handle if needed
  if (!thr_entry->hipsparse_handle) {
    HIPSPARSE_CALL(hipsparseCreate(&(thr_entry->hipsparse_handle)));
  }
  HIPSPARSE_CALL(hipsparseSetStream(thr_entry->hipsparse_handle, stream));

  NDArray indptr = csr->indptr;
  NDArray indices = csr->indices;
  const auto& ctx = indptr->ctx;
  const int64_t nnz = indices->shape[0];
  if (!aten::CSRHasData(*csr))
    csr->data = aten::Range(0, nnz, indices->dtype.bits, ctx);
  NDArray data = csr->data;

  size_t workspace_size = 0;
  HIPSPARSE_CALL(hipsparseXcsrsort_bufferSizeExt(
      thr_entry->hipsparse_handle, csr->num_rows, csr->num_cols, nnz,
      indptr.Ptr<int32_t>(), indices.Ptr<int32_t>(), &workspace_size));
  void* workspace = device->AllocWorkspace(ctx, workspace_size);

  hipsparseMatDescr_t descr;
  HIPSPARSE_CALL(hipsparseCreateMatDescr(&descr));
  HIPSPARSE_CALL(hipsparseSetMatType(descr, HIPSPARSE_MATRIX_TYPE_GENERAL));
  HIPSPARSE_CALL(hipsparseSetMatIndexBase(descr, HIPSPARSE_INDEX_BASE_ZERO));
  HIPSPARSE_CALL(hipsparseXcsrsort(
      thr_entry->hipsparse_handle, csr->num_rows, csr->num_cols, nnz, descr,
      indptr.Ptr<int32_t>(), indices.Ptr<int32_t>(), data.Ptr<int32_t>(),
      workspace));

  csr->sorted = true;

  // free resources
  HIPSPARSE_CALL(hipsparseDestroyMatDescr(descr));
  device->FreeWorkspace(ctx, workspace);
}

template <>
void CSRSort_<kDGLROCM, int64_t>(CSRMatrix* csr) {
  hipStream_t stream = runtime::getCurrentHIPStream();
  auto device = runtime::DeviceAPI::Get(csr->indptr->ctx);

  const auto& ctx = csr->indptr->ctx;
  const int64_t nnz = csr->indices->shape[0];
  const auto nbits = csr->indptr->dtype.bits;
  if (!aten::CSRHasData(*csr)) csr->data = aten::Range(0, nnz, nbits, ctx);

  IdArray new_indices = csr->indices.Clone();
  IdArray new_data = csr->data.Clone();

  const int64_t* offsets = csr->indptr.Ptr<int64_t>();
  const int64_t* key_in = csr->indices.Ptr<int64_t>();
  int64_t* key_out = new_indices.Ptr<int64_t>();
  const int64_t* value_in = csr->data.Ptr<int64_t>();
  int64_t* value_out = new_data.Ptr<int64_t>();

  // Allocate workspace
  size_t workspace_size = 0;
  HIP_CALL(hipcub::DeviceSegmentedRadixSort::SortPairs(
      nullptr, workspace_size, key_in, key_out, value_in, value_out, nnz,
      csr->num_rows, offsets, offsets + 1, 0, sizeof(int64_t) * 8, stream));
  void* workspace = device->AllocWorkspace(ctx, workspace_size);

  // Compute
  HIP_CALL(hipcub::DeviceSegmentedRadixSort::SortPairs(
      workspace, workspace_size, key_in, key_out, value_in, value_out, nnz,
      csr->num_rows, offsets, offsets + 1, 0, sizeof(int64_t) * 8, stream));

  csr->sorted = true;
  csr->indices = new_indices;
  csr->data = new_data;

  // free resources
  device->FreeWorkspace(ctx, workspace);
}

template void CSRSort_<kDGLROCM, int32_t>(CSRMatrix* csr);
template void CSRSort_<kDGLROCM, int64_t>(CSRMatrix* csr);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
