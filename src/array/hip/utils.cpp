/**
 *  Copyright (c) 2020 by Contributors
 * @file array/hip/utils.cu
 * @brief Utilities for CUDA kernels.
 */

#include "../../runtime/hip/hip_common.h"
#include "./dgl_cub.h"
#include "./utils.h"

namespace dgl {
namespace hip {

bool AllTrue(int8_t* flags, int64_t length, const DGLContext& ctx) {
  auto device = runtime::DeviceAPI::Get(ctx);
  int8_t* rst = static_cast<int8_t*>(device->AllocWorkspace(ctx, 1));
  // Call CUB's reduction
  size_t workspace_size = 0;
  hipStream_t stream = runtime::getCurrentHIPStream();
  HIP_CALL(hipcub::DeviceReduce::Min(
      nullptr, workspace_size, flags, rst, length, stream));
  void* workspace = device->AllocWorkspace(ctx, workspace_size);
  HIP_CALL(hipcub::DeviceReduce::Min(
      workspace, workspace_size, flags, rst, length, stream));
  int8_t cpu_rst = GetHIPScalar(device, ctx, rst);
  device->FreeWorkspace(ctx, workspace);
  device->FreeWorkspace(ctx, rst);
  return cpu_rst == 1;
}

}  // namespace hip
}  // namespace dgl
