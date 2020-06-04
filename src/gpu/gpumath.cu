#include "gpumath.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>

#include "math.hpp"

using namespace std::string_literals;

namespace {
/// Compute the logarithm of the diagonal entries and store it in matrix[0] - matrix[n-1].
__global__ void logOfDiagonal(thrust::complex<double> *matrix,
                              int n)
{
    int const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        auto const val = matrix[i * n + i];
        matrix[i] = thrust::complex<double>(log(abs(val)), arg(val));
    }
}

/// Store 1 (true) for each row that has been swapped in the pivot array.
__global__ void findSwaps(int *ipiv,
                          int n)
{
    int const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        ipiv[i] = (ipiv[i] - 1 != i);
    }
}

/// Perform a PLU decomposition of mat in place.
void luDecomp(cuDoubleComplex *mat,
              int const n,
              int *ipiv,
              cublasHandle_t handle)
{
    constexpr int batchsize = 1;

    // Create double pointer to mat on the device.
    cuDoubleComplex **A;
    auto result = cudaMalloc(&A, batchsize * sizeof(A));
    if (result != cudaSuccess) {
        throw std::runtime_error("Failed to allocate double pointer: "s
                                 + cudaGetErrorString(result));
    }
    result = cudaMemcpy(A, &mat, batchsize * sizeof(A), cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        cudaFree(A);
        throw std::runtime_error("Failed to copy double pointer: "s
                                 + cudaGetErrorString(result));
    }

    // info array
    thrust::device_vector<int> info(n);

    // perform factorization
    auto getrfResult = cublasZgetrfBatched(handle,
                                           n,
                                           A,
                                           n,
                                           ipiv,
                                           thrust::raw_pointer_cast(info.data()),
                                           batchsize);
    cudaFree(A);  // This is just the double pointer, not the actual matrix.
    if (getrfResult != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("LU decomposition failed: " + std::to_string(getrfResult));
    }
    thrust::host_vector<int> hinfo = info;
    if (hinfo[0] != 0) {
        throw std::runtime_error("LU decomposition failed, info: " + std::to_string(info[0]));
    }
}
}  // namespace

thrust::complex<double>
  ilogdet(thrust::device_vector<thrust::complex<double>> mat,
          int const n,
          cublasHandle_t handle)
{
    thrust::device_vector<int> ipiv(n);
    luDecomp(reinterpret_cast<cuDoubleComplex *>(thrust::raw_pointer_cast(mat.data())),
             n,
             thrust::raw_pointer_cast(ipiv.data()),
             handle);

    // determinant of permutation matrix P
    findSwaps<<<(n + 255) / 256, 256>>>(thrust::raw_pointer_cast(ipiv.data()), n);
    auto swapResult = cudaGetLastError();
    if (swapResult != cudaSuccess) {
        throw std::runtime_error(std::string("Failed to call kernel findSwaps: ")
                                 + cudaGetErrorString(swapResult));
    }
    bool const negDetP = thrust::reduce(
      ipiv.begin(), ipiv.end(), 0,
      [] __device__(int a, int b) { return a ^ b; }) == 1;

    // determinant of U
    logOfDiagonal<<<(n + 255) / 256, 256>>>(thrust::raw_pointer_cast(mat.data()), n);
    auto logResult = cudaGetLastError();
    if (logResult != cudaSuccess) {
        throw std::runtime_error(std::string("Failed to call kernel logOfDiagonal: ")
                                 + cudaGetErrorString(logResult));
    }
    thrust::complex<double> logdetU = thrust::reduce(mat.begin(), mat.begin() + n);

    // combine log dets and project to (-pi, pi]
    return toFirstLogBranch(logdetU + (negDetP ? thrust::complex<double>{ 0, pi<double> } : 0));
}
