#pragma once

#include <hipsolver.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

// -----------------------------------------------------
// Definitions
// -----------------------------------------------------
namespace dgl
{

namespace detail
{

extern hipblasHandle_t hipblas_handle;
extern hipsolverDnHandle_t hipsolver_handle;
static constexpr hipDataType hip_double = HIP_R_64F;

// -----------------------------------------------------
// Debugging
// -----------------------------------------------------

#define hip_assert(a) hip_assert_(a, #a, __LINE__, __FUNCTION__, __FILE__);

inline void hip_assert_(
  hipError_t error,
  std::string call,
  int line,
  std::string function,
  std::string file )
{
  if(error == hipSuccess)
  {
    return;
  }

  std::cout << "Error in " << function << ", " << file << ":" << line
            << std::endl;
  std::cout << call << std::endl;

  std::cout << hipGetErrorName(error) << " " << hipGetErrorString(error) << std::endl;
  exit(1);
}

// -----------------------------------------------------
// Memory utilities
// -----------------------------------------------------
template < typename T >
inline void hip_memset(T* ptr, int val, size_t nelems)
{
  if (nelems > 0u) {
    hipStream_t stream = 0;
    hip_assert(hipMemsetAsync(ptr, val, sizeof(T)*nelems, stream));
  }
}

template < typename T >
inline void hip_memcpy(T* dst, const T* src, size_t nelems)
{
  boba_assert( nelems > 0u , "can't copy nonpositve nelems" );
  if (nelems > 0u) {
    hipStream_t stream = 0;
    hipMemcpyKind kind = hipMemcpyDefault;
    hip_assert(hipMemcpyAsync(dst, src, sizeof(T)*nelems, kind, stream));
  }
}

template < typename T >
inline T* hip_malloc(size_t nelems)
{
  void* ptr = nullptr;
  if (nelems > 0u) {
    hip_assert(hipMalloc(&ptr, sizeof(T)*nelems));
  }
  return static_cast<T*>(ptr);
}

template < typename T >
inline T* hip_malloc_memcpy(const T* src, size_t nelems)
{
  T* ptr = hip_malloc<T>(nelems);
  hip_memcpy(ptr, src, nelems);
  return static_cast<T*>(ptr);
}

template < typename T >
inline void hip_free(T*& ptr)
{
  if (ptr != nullptr) {
    hip_assert(hipFree(ptr));
    ptr = nullptr;
  }
}

template < typename T >
inline void hip_memcpy_free(T* dst, T*& ptr, size_t nelems)
{
  hip_memcpy(dst, ptr, nelems);
  hip_free(ptr);
}

const size_t block_size = 256; // Hip thread block size

// -----------------------------------------------------
// Lib Debugging
// -----------------------------------------------------

#define hipblas_assert(a) hipblas_assert_(a, #a, __LINE__, __FUNCTION__, __FILE__);

inline void hipblas_assert_(
  hipblasStatus_t error,
  std::string call,
  int line,
  std::string function,
  std::string file )
{
  if(error == HIPBLAS_STATUS_SUCCESS)
  {
    return;
  }

  std::cout << "Error in " << function << ", " << file << ":" << line
            << std::endl;
  std::cout << call << std::endl;
  exit(1);
}

#if 0
inline void hipsolver_error_codes_(hipsolverStatus_t error)
{
  dgl_always_assert(error != HIPSOLVER_STATUS_NOT_INITIALIZED , "HIPSOLVER_STATUS_NOT_INITIALIZED ");
  dgl_always_assert(error != HIPSOLVER_STATUS_ALLOC_FAILED, "HIPSOLVER_STATUS_ALLOC_FAILED");
  dgl_always_assert(error != HIPSOLVER_STATUS_INVALID_VALUE , "HIPSOLVER_STATUS_INVALID_VALUE ");
  dgl_always_assert(error != HIPSOLVER_STATUS_ARCH_MISMATCH , "HIPSOLVER_STATUS_ARCH_MISMATCH ");
  dgl_always_assert(error != HIPSOLVER_STATUS_EXECUTION_FAILED , "HIPSOLVER_STATUS_EXECUTION_FAILED ");
  dgl_always_assert(error != HIPSOLVER_STATUS_INTERNAL_ERROR, "HIPSOLVER_STATUS_INTERNAL_ERROR");
  //dgl_always_assert(error != HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED, "HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED ");
  dgl_always_assert_equal(error, HIPSOLVER_STATUS_SUCCESS, "hipsolver error");
}

#define hipsolver_assert(a) hipsolver_assert_(a, #a, __LINE__, __FUNCTION__, __FILE__);

inline void hipsolver_assert_(
  hipsolverStatus_t error,
  std::string call,
  int line,
  std::string function,
  std::string file )
{
  if(error == HIPSOLVER_STATUS_SUCCESS)
  {
    return;
  }
  std::cout << "Error in " << function << ", " << file << ":" << line
            << std::endl;
  std::cout << call << std::endl;

  hipsolver_error_codes_(error);
  exit(1);
}

#define hipsolver_assert_info(a, b) hipsolver_assert_info_(a, #a, b, __LINE__, __FUNCTION__, __FILE__);

inline void hipsolver_assert_info_(
  hipsolverStatus_t error,
  std::string call,
  int* info,
  int line,
  std::string function,
  std::string file )
{
  if(error == HIPSOLVER_STATUS_SUCCESS)
  {
    return;
  }

  int host_info = 0;
  hip_memcpy(&host_info, info, 1);

  std::cout << "Error in " << function << ", " << file << ":" << line
            << std::endl;
  std::cout << call << std::endl;
  std::cout << "Info flag: " << host_info << std::endl;

  hipsolver_error_codes_(error);
  exit(1);
}

#endif
// -----------------------------------------------------
// Loop details
// -----------------------------------------------------
template < int block_size, typename Lambda >
__launch_bounds__(block_size)
__global__ void boba_lambda_kernel(int begin, int end, Lambda lambda)
{
  int i = begin + threadIdx.x + block_size*blockIdx.x;
  if (i < end) {
    lambda(i);
  }
}

template < int block_size0, int block_size1, typename Lambda >
__launch_bounds__(block_size0 * block_size1)
__global__ void boba_lambda_kernel_2d(
  int begin0, int end0,
  int begin1, int end1,
  Lambda lambda)
{
  int i0 = begin0 + threadIdx.x + block_size0*blockIdx.x;
  int i1 = begin1 + threadIdx.y + block_size1*blockIdx.y;
  if (i0 < end0 && i1 < end1) {
    lambda(i0, i1);
  }
}

template < int block_size0, int block_size1, int block_size2, typename Lambda >
__launch_bounds__(block_size0 * block_size1 * block_size2)
__global__ void boba_lambda_kernel_3d(
  int begin0, int end0,
  int begin1, int end1,
  int begin2, int end2,
  Lambda lambda)
{
  int i0 = begin0 + threadIdx.x + block_size0*blockIdx.x;
  int i1 = begin1 + threadIdx.y + block_size1*blockIdx.y;
  int i2 = begin2 + threadIdx.z + block_size2*blockIdx.z;
  if (i0 < end0 && i1 < end1 && i2 < end2) {
    lambda(i0, i1, i2);
  }
}

template < int block_size0, int block_size1, int block_size2, typename Lambda >
__launch_bounds__(block_size0 * block_size1 * block_size2)
__global__ void boba_lambda_kernel_4d(
  int begin0, int end0,
  int begin1, int end1,
  int begin2, int end2,
  int begin3, int end3,
  Lambda lambda)
{
  int i0 = begin0 + threadIdx.x + block_size0*blockIdx.x;
  int i1 = begin1 + threadIdx.y + block_size1*blockIdx.y;
  int i2 = begin2 + threadIdx.z + block_size2*blockIdx.z;
  if (i0 < end0 && i1 < end1 && i2 < end2) {
    for (int i3 = begin3; i3 < end3; ++i3) {
      lambda(i0, i1, i2, i3);
    }
  }
}

template < typename Lambda >
inline void hip_launch(int begin, int end, Lambda&& lambda)
{
  int shmem = 0;
  hipStream_t stream = 0;
  static constexpr int block_size = 256;
  int grid_size = (end-begin + block_size-1) / block_size;
  void* args[] = {(void*)&begin, (void*)&end, (void*)&lambda};
  hip_assert(hipLaunchKernel(
      (const void*)boba_lambda_kernel<block_size, std::decay_t<Lambda>>,
      grid_size, block_size, args, shmem, stream));
}

template < typename Lambda >
inline void hip_launch_2d(
  int begin0, int end0,
  int begin1, int end1,
  Lambda&& lambda)
{
  int shmem = 0;
  hipStream_t stream = 0;
  static constexpr int block_size0 = 32;
  static constexpr int block_size1 = 8;
  int grid_size0 = (end0-begin0 + block_size0-1) / block_size0;
  int grid_size1 = (end1-begin1 + block_size1-1) / block_size1;
  void* args[] = {(void*)&begin0, (void*)&end0,
                  (void*)&begin1, (void*)&end1,
                  (void*)&lambda};
  hip_assert(hipLaunchKernel(
      (const void*)boba_lambda_kernel_2d<block_size0, block_size1, std::decay_t<Lambda>>,
      dim3(grid_size0, grid_size1), dim3(block_size0, block_size1), args, shmem, stream));
}

template < typename Lambda >
inline void hip_launch_3d(
  int begin0, int end0,
  int begin1, int end1,
  int begin2, int end2,
  Lambda&& lambda)
{
  int shmem = 0;
  hipStream_t stream = 0;
  static constexpr int block_size0 = 32;
  static constexpr int block_size1 = 4;
  static constexpr int block_size2 = 2;
  int grid_size0 = (end0-begin0 + block_size0-1) / block_size0;
  int grid_size1 = (end1-begin1 + block_size1-1) / block_size1;
  int grid_size2 = (end2-begin2 + block_size2-1) / block_size2;
  void* args[] = {(void*)&begin0, (void*)&end0,
                  (void*)&begin1, (void*)&end1,
                  (void*)&begin2, (void*)&end2,
                  (void*)&lambda};
  hip_assert(hipLaunchKernel(
      (const void*)boba_lambda_kernel_3d<block_size0, block_size1, block_size2, std::decay_t<Lambda>>,
      dim3(grid_size0, grid_size1, grid_size2), dim3(block_size0, block_size1, block_size2), args, shmem, stream));
}

template < typename Lambda >
inline void hip_launch_4d(
  int begin0, int end0,
  int begin1, int end1,
  int begin2, int end2,
  int begin3, int end3,
  Lambda&& lambda)
{
  int shmem = 0;
  hipStream_t stream = 0;
  static constexpr int block_size0 = 32;
  static constexpr int block_size1 = 4;
  static constexpr int block_size2 = 2;
  int grid_size0 = (end0-begin0 + block_size0-1) / block_size0;
  int grid_size1 = (end1-begin1 + block_size1-1) / block_size1;
  int grid_size2 = (end2-begin2 + block_size2-1) / block_size2;
  void* args[] = {(void*)&begin0, (void*)&end0,
                  (void*)&begin1, (void*)&end1,
                  (void*)&begin2, (void*)&end2,
                  (void*)&begin3, (void*)&end3,
                  (void*)&lambda};
  hip_assert(hipLaunchKernel(
      (const void*)boba_lambda_kernel_4d<block_size0, block_size1, block_size2, std::decay_t<Lambda>>,
      dim3(grid_size0, grid_size1, grid_size2), dim3(block_size0, block_size1, block_size2), args, shmem, stream));
}

inline void hip_syncronize()
{
  hipStream_t stream = 0;
  hip_assert(hipStreamSynchronize(stream));
}

}; // namespace detail
}; // namespace dgl


