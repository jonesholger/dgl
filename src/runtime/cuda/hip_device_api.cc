/**
 *  Copyright (c) 2017-2022 by Contributors
 * @file cuda_device_api.cc
 * @brief GPU specific API
 */
#include <hip/hip_runtime.h>
#include <dgl/runtime/device_api.h>
#include <dgl/runtime/registry.h>
#include <dgl/runtime/tensordispatch.h>
#include <dmlc/thread_local.h>

#include "cuda_common.h"

namespace dgl {
namespace runtime {

class CUDADeviceAPI final : public DeviceAPI {
 public:
  CUDADeviceAPI() {
    int count;
    auto err = hipGetDeviceCount(&count);
    switch (err) {
      case hipSuccess:
        break;
      default:
        count = 0;
        hipGetLastError();
    }
    is_available_ = count > 0;
  }

  bool IsAvailable() final { return is_available_; }

  void SetDevice(DGLContext ctx) final {
    CUDA_CALL(hipSetDevice(ctx.device_id));
  }
  void GetAttr(DGLContext ctx, DeviceAttrKind kind, DGLRetValue* rv) final {
    int value = 0;
    switch (kind) {
      case kExist:
        value =
            (hipDeviceGetAttribute(
                 &value, hipDeviceAttributeMaxThreadsPerBlock, ctx.device_id) ==
             hipSuccess);
        break;
      case kMaxThreadsPerBlock: {
        CUDA_CALL(hipDeviceGetAttribute(
            &value, hipDeviceAttributeMaxThreadsPerBlock, ctx.device_id));
        break;
      }
      case kWarpSize: {
        CUDA_CALL(
            hipDeviceGetAttribute(&value, hipDeviceAttributeWarpSize, ctx.device_id));
        break;
      }
      case kMaxSharedMemoryPerBlock: {
        CUDA_CALL(hipDeviceGetAttribute(
            &value, hipDeviceAttributeMaxSharedMemoryPerBlock, ctx.device_id));
        break;
      }
      case kComputeVersion: {
        std::ostringstream os;
        CUDA_CALL(hipDeviceGetAttribute(
            &value, hipDeviceAttributeComputeCapabilityMajor, ctx.device_id));
        os << value << ".";
        CUDA_CALL(hipDeviceGetAttribute(
            &value, hipDeviceAttributeComputeCapabilityMinor, ctx.device_id));
        os << value;
        *rv = os.str();
        return;
      }
      case kDeviceName: {
        hipDeviceProp_t props;
        CUDA_CALL(hipGetDeviceProperties(&props, ctx.device_id));
        *rv = std::string(props.name);
        return;
      }
      case kMaxClockRate: {
        CUDA_CALL(hipDeviceGetAttribute(
            &value, hipDeviceAttributeClockRate, ctx.device_id));
        break;
      }
      case kMultiProcessorCount: {
        CUDA_CALL(hipDeviceGetAttribute(
            &value, hipDeviceAttributeMultiprocessorCount, ctx.device_id));
        break;
      }
      case kMaxThreadDimensions: {
        int dims[3];
        CUDA_CALL(hipDeviceGetAttribute(
            &dims[0], hipDeviceAttributeMaxBlockDimX, ctx.device_id));
        CUDA_CALL(hipDeviceGetAttribute(
            &dims[1], hipDeviceAttributeMaxBlockDimY, ctx.device_id));
        CUDA_CALL(hipDeviceGetAttribute(
            &dims[2], hipDeviceAttributeMaxBlockDimZ, ctx.device_id));

        std::stringstream ss;  // use json string to return multiple int values;
        ss << "[" << dims[0] << ", " << dims[1] << ", " << dims[2] << "]";
        *rv = ss.str();
        return;
      }
    }
    *rv = value;
  }
  void* AllocDataSpace(
      DGLContext ctx, size_t nbytes, size_t alignment,
      DGLDataType type_hint) final {
    SetDevice(ctx);
    // Redirect to PyTorch's allocator when available.
    TensorDispatcher* tensor_dispatcher = TensorDispatcher::Global();
    if (tensor_dispatcher->IsAvailable()) {
      return tensor_dispatcher->CUDAAllocWorkspace(
          nbytes, getCurrentCUDAStream());
    }
    CHECK_EQ(256 % alignment, 0U) << "CUDA space is aligned at 256 bytes";
    void* ret;
    CUDA_CALL(hipMalloc(&ret, nbytes));
    return ret;
  }

  void FreeDataSpace(DGLContext ctx, void* ptr) final {
    SetDevice(ctx);
    TensorDispatcher* tensor_dispatcher = TensorDispatcher::Global();
    if (tensor_dispatcher->IsAvailable()) {
      return tensor_dispatcher->CUDAFreeWorkspace(ptr);
    }
    CUDA_CALL(hipFree(ptr));
  }

  void CopyDataFromTo(
      const void* from, size_t from_offset, void* to, size_t to_offset,
      size_t size, DGLContext ctx_from, DGLContext ctx_to,
      DGLDataType type_hint, DGLStreamHandle stream) {
    hipStream_t cu_stream = static_cast<hipStream_t>(stream);
    from = static_cast<const char*>(from) + from_offset;
    to = static_cast<char*>(to) + to_offset;
    if (ctx_from.device_type == kDGLCUDA && ctx_to.device_type == kDGLCUDA) {
      CUDA_CALL(hipSetDevice(ctx_from.device_id));
      if (ctx_from.device_id == ctx_to.device_id) {
        GPUCopy(from, to, size, hipMemcpyDeviceToDevice, cu_stream);
      } else {
        CUDA_CALL(hipMemcpyPeerAsync(
            to, ctx_to.device_id, from, ctx_from.device_id, size, cu_stream));
      }
    } else if (
        ctx_from.device_type == kDGLCUDA && ctx_to.device_type == kDGLCPU) {
      CUDA_CALL(hipSetDevice(ctx_from.device_id));
      GPUCopy(from, to, size, hipMemcpyDeviceToHost, cu_stream);
    } else if (
        ctx_from.device_type == kDGLCPU && ctx_to.device_type == kDGLCUDA) {
      CUDA_CALL(hipSetDevice(ctx_to.device_id));
      GPUCopy(from, to, size, hipMemcpyHostToDevice, cu_stream);
    } else {
      LOG(FATAL) << "expect copy from/to GPU or between GPU";
    }
  }

  void CopyDataFromTo(
      const void* from, size_t from_offset, void* to, size_t to_offset,
      size_t size, DGLContext ctx_from, DGLContext ctx_to,
      DGLDataType type_hint) final {
    auto stream = GetStream();
    CopyDataFromTo(
        from, from_offset, to, to_offset, size, ctx_from, ctx_to, type_hint,
        stream);
  }

  // To ensure correct behavior, `record_event` must be invoked anytime a
  // pointer from PyTorch CachingHostAllocator is used in a hipMemcpyAsync
  // call. It provides a way to re-use freed pinned (page-locked) memory
  // allocations and avoid device sync due to hipHostFree calls.
  void RecordedCopyDataFromTo(
      void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
      DGLContext ctx_from, DGLContext ctx_to, DGLDataType type_hint,
      void* pytorch_ctx) final {
    auto stream = GetStream();
    CopyDataFromTo(
        from, from_offset, to, to_offset, size, ctx_from, ctx_to, type_hint,
        stream);
    auto tensor_dispatcher = TensorDispatcher::Global();
    if (tensor_dispatcher->IsAvailable()) {
      auto custream = static_cast<hipStream_t>(stream);
      void* ptr = ctx_to.device_type == kDGLCPU ? to : from;
      int id =
          ctx_to.device_type == kDGLCPU ? ctx_from.device_id : ctx_to.device_id;
      tensor_dispatcher->CUDARecordHostAlloc(ptr, pytorch_ctx, custream, id);
    }
  }

  DGLStreamHandle CreateStream(DGLContext ctx) {
    CUDA_CALL(hipSetDevice(ctx.device_id));
    hipStream_t retval;
    // make sure the legacy default stream won't block on this stream
    CUDA_CALL(hipStreamCreateWithFlags(&retval, hipStreamNonBlocking));
    return static_cast<DGLStreamHandle>(retval);
  }

  void FreeStream(DGLContext ctx, DGLStreamHandle stream) {
    CUDA_CALL(hipSetDevice(ctx.device_id));
    hipStream_t cu_stream = static_cast<hipStream_t>(stream);
    CUDA_CALL(hipStreamDestroy(cu_stream));
  }

  void SyncStreamFromTo(
      DGLContext ctx, DGLStreamHandle event_src, DGLStreamHandle event_dst) {
    CUDA_CALL(hipSetDevice(ctx.device_id));
    hipStream_t src_stream = static_cast<hipStream_t>(event_src);
    hipStream_t dst_stream = static_cast<hipStream_t>(event_dst);
    hipEvent_t evt;
    CUDA_CALL(hipEventCreate(&evt));
    CUDA_CALL(hipEventRecord(evt, src_stream));
    CUDA_CALL(hipStreamWaitEvent(dst_stream, evt, 0));
    CUDA_CALL(hipEventDestroy(evt));
  }

  void StreamSync(DGLContext ctx, DGLStreamHandle stream) final {
    CUDA_CALL(hipSetDevice(ctx.device_id));
    CUDA_CALL(hipStreamSynchronize(static_cast<hipStream_t>(stream)));
  }

  /** NOTE: If the backend is PyTorch, we will use PyTorch's stream management,
   *        so just avoid calling our SetStream/CreateStream unless
   *        you really need advanced stream control.
   * TODO(Xin): Redirect this to PyTorch or remove it.
   * PyTorch allows external CUDA streams to be set as current since v1.11.
   */
  void SetStream(DGLContext ctx, DGLStreamHandle stream) final {}

  DGLStreamHandle GetStream() const final {
    return static_cast<DGLStreamHandle>(getCurrentCUDAStream());
  }

  /** NOTE: hipHostRegister can be called from an arbitrary GPU device,
   *        so we don't need to specify a ctx.
   *        The pinned memory can be seen by all CUDA contexts,
   *        not just the one that performed the allocation
   */
  bool PinData(void* ptr, size_t nbytes) override {
    // prevent users from pinning empty tensors or graphs
    if (ptr == nullptr || nbytes == 0) return false;
    TensorDispatcher* tensor_dispatcher = TensorDispatcher::Global();
    // Minimize the pinned memory pool allocated by backend (via tensoradapter)
    // to preserve enough memory for DGL inherited in-place pin-memory operation
    if (tensor_dispatcher->IsAvailable()) {
      tensor_dispatcher->CUDAHostAllocatorEmptyCache();
    }
    CUDA_CALL(hipHostRegister(ptr, nbytes, hipHostRegisterDefault));
    return true;
  }

  void UnpinData(void* ptr) {
    if (ptr == nullptr) return;
    CUDA_CALL(hipHostUnregister(ptr));
  }

  void* AllocPinnedDataSpace(
      size_t nbytes, void** ctx, void** deleter) override {
    // prevent pinning empty tensors or graphs
    if (nbytes == 0) return nullptr;
    TensorDispatcher* tensor_dispatcher = TensorDispatcher::Global();
    CHECK(tensor_dispatcher->IsAvailable())
        << "CachingHostAllocator is not available in the current backend "
           "PyTorch. Please update the PyTorch version to 1.11+";
    return tensor_dispatcher->CUDAAllocHostWorkspace(nbytes, ctx, deleter);
  }

  void FreePinnedDataSpace(void** deleter) override {
    TensorDispatcher* tensor_dispatcher = TensorDispatcher::Global();
    CHECK(tensor_dispatcher->IsAvailable())
        << "CachingHostAllocator is not available in the current backend "
           "PyTorch. Please update the PyTorch version to 1.11+";
    tensor_dispatcher->CUDAFreeHostWorkspace(deleter);
  }

  bool IsPinned(const void* ptr) override {
    // can't be a pinned tensor if CUDA context is unavailable.
    if (!is_available_) return false;

    hipPointerAttribute_t attr;
    hipError_t status = hipPointerGetAttributes(&attr, ptr);
    bool result = false;

    switch (status) {
      case hipErrorInvalidValue:
        // might be a normal CPU tensor in CUDA 10.2-
        hipGetLastError();  // clear error
        break;
      case hipSuccess:
        result = (attr.type == hipMemoryTypeHost);
        break;
      case hipErrorNotInitialized:
      case hipErrorNoDevice:
      case hipErrorInsufficientDriver:
      case hipErrorInvalidDevice:
        // We don't want to fail in these particular cases since this function
        // can be called when users only want to run on CPU even if CUDA API is
        // enabled, or in a forked subprocess where CUDA context cannot be
        // initialized.  So we just mark the CUDA context to unavailable and
        // return.
        is_available_ = false;
        hipGetLastError();  // clear error
        break;
      default:
        LOG(FATAL) << "error while determining memory status: "
                   << hipGetErrorString(status);
        break;
    }

    return result;
  }

  void* AllocWorkspace(
      DGLContext ctx, size_t size, DGLDataType type_hint) final {
    SetDevice(ctx);
    // Redirect to PyTorch's allocator when available.
    TensorDispatcher* tensor_dispatcher = TensorDispatcher::Global();
    if (tensor_dispatcher->IsAvailable())
      return tensor_dispatcher->CUDAAllocWorkspace(
          size, getCurrentCUDAStream());

    return CUDAThreadEntry::ThreadLocal()->pool.AllocWorkspace(ctx, size);
  }

  void FreeWorkspace(DGLContext ctx, void* data) final {
    SetDevice(ctx);
    TensorDispatcher* tensor_dispatcher = TensorDispatcher::Global();
    if (tensor_dispatcher->IsAvailable())
      return tensor_dispatcher->CUDAFreeWorkspace(data);

    CUDAThreadEntry::ThreadLocal()->pool.FreeWorkspace(ctx, data);
  }

  static const std::shared_ptr<CUDADeviceAPI>& Global() {
    static std::shared_ptr<CUDADeviceAPI> inst =
        std::make_shared<CUDADeviceAPI>();
    return inst;
  }

 private:
  static void GPUCopy(
      const void* from, void* to, size_t size, hipMemcpyKind kind,
      hipStream_t stream) {
    CUDA_CALL(hipMemcpyAsync(to, from, size, kind, stream));
    if (stream == 0 && kind == hipMemcpyDeviceToHost) {
      // only wait for the copy, when it's on the default stream, and it's to
      // host memory
      CUDA_CALL(hipStreamSynchronize(stream));
    }
  }

  bool is_available_ = true;
};

typedef dmlc::ThreadLocalStore<CUDAThreadEntry> CUDAThreadStore;

CUDAThreadEntry::CUDAThreadEntry() : pool(kDGLCUDA, CUDADeviceAPI::Global()) {}

CUDAThreadEntry* CUDAThreadEntry::ThreadLocal() {
  return CUDAThreadStore::Get();
}

hipStream_t getCurrentCUDAStream() {
  TensorDispatcher* tensor_dispatcher = TensorDispatcher::Global();
  if (tensor_dispatcher->IsAvailable())
    return tensor_dispatcher->CUDAGetCurrentStream();
  else  // return the default stream when TA is not available
    return nullptr;
}

DGL_REGISTER_GLOBAL("device_api.cuda")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      DeviceAPI* ptr = CUDADeviceAPI::Global().get();
      *rv = static_cast<void*>(ptr);
    });

}  // namespace runtime
}  // namespace dgl
