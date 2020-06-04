#include "gpumath.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

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

/// Perform an PLU decomposition of mat in place.
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

    thrust::host_vector<int> hipiv = ipiv;
    bool negDetP = false;  // if true det(P) == -1, else det(P) == +1
    for (std::size_t i = 0; i < n; ++i) {
        // determinant of pivot matrix P
        if (hipiv[i] - 1 != i) {
            negDetP = !negDetP;
        }
    }

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
