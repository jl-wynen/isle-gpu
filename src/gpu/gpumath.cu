#include "gpumath.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

#include "math.hpp"

using namespace std::string_literals;

namespace {
// Compute the logarithm of the diagonal entries and store it in
// matrix[0] - matrix[n-1].
__global__ void logOfDiagonal(thrust::complex<double> *matrix,
                              int n)
{
    int const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        auto const val = matrix[i * n + i];
        matrix[i] = thrust::complex<double>(log(abs(val)), arg(val));
    }
}

void luDecomp(cuDoubleComplex *mat,
              int const n,
              int *ipiv,
              cublasHandle_t handle)
{
    // create double pointer to mat on the device
    cuDoubleComplex **A;
    auto result = cudaMalloc(&A, 1 * sizeof(A));
    if (result != cudaSuccess) {
        throw std::runtime_error("Failed to allocate double pointer: "s
                                 + cudaGetErrorString(result));
    }
    result = cudaMemcpy(A, &mat, 1 * sizeof(A), cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        throw std::runtime_error("Failed to copy double pointer: "s
                                 + cudaGetErrorString(result));
    }

    // info array
    thrust::device_vector<int> info(n);

    auto getrfResult = cublasZgetrfBatched(handle,
                                           n,
                                           A,
                                           n,
                                           ipiv,
                                           thrust::raw_pointer_cast(info.data()),
                                           1);
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
    luDecomp(reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(mat.data())),
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

    logOfDiagonal<<<(n+255)/256, 256>>>(thrust::raw_pointer_cast(mat.data()), n);
    auto logResult = cudaGetLastError();
    if (logResult != cudaSuccess) {
        throw std::runtime_error(std::string("Failed to call kernel logOfDiagonal: ")
                                 + cudaGetErrorString(logResult));
    }
    thrust::complex<double> logdetU = thrust::reduce(mat.begin(), mat.begin() + n);

    // combine log dets and project to (-pi, pi]
    return toFirstLogBranch(logdetU + (negDetP ? thrust::complex<double>{ 0, pi<double> } : 0));


    //        // pivot indices
    //        thrust::device_vector<int> ipiv(n);
    //        // info array
    //        thrust::device_vector<int> info(n);
    //        // perform LU-decomposition, afterwards matrix = PLU
    //        thrust::device_ptr<cuDoubleComplex*> rawData;
    //        *rawData = reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(mat.data()));
    //        //cuDoubleComplex *rawData = reinterpret_cast<cuDoubleComplex *>(thrust::raw_pointer_cast(mat.data()));
    //        auto result = cublasZgetrfBatched(handle,
    //                                          n,
    //                                          thrust::raw_pointer_cast(rawData),
    //                                          n,
    //                                          thrust::raw_pointer_cast(ipiv.data()),
    //                                          thrust::raw_pointer_cast(info.data()),
    //                                          1);
    //        if (result != CUBLAS_STATUS_SUCCESS) {
    //            throw std::runtime_error("LU decomposition failed: " + std::to_string(result));
    //        }
    //        if (info[0] != 0) {
    //            throw std::runtime_error("LU decomposition failed, info: " + std::to_string(info[0]));
    //        }
//
//    thrust::host_vector<int> hipiv = ipiv;
//    bool negDetP = false;  // if true det(P) == -1, else det(P) == +1
//    for (std::size_t i = 0; i < n; ++i) {
//        // determinant of pivot matrix P
//        if (hipiv[i] - 1 != i) {
//            negDetP = !negDetP;
//        }
//    }
//
//    logOfDiagonal<<<n, 1>>>(thrust::raw_pointer_cast(mat.data()), n);
//    auto logResult = cudaGetLastError();
//    if (logResult != cudaSuccess) {
//        throw std::runtime_error(std::string("Failed to call kernel logOfDiagonal: ")
//                                 + cudaGetErrorString(logResult));
//    }
//    // thrust::complex<double> res = thrust::reduce(mat.begin(), mat.end() + n);
//    thrust::host_vector<thrust::complex<double>> hm = mat;
//    thrust::complex<double> res = 0;
//    for (int i = 0; i < hm.size(); ++i) {
//        res += hm[i];
//    }
//
//    // combine log dets and project to (-pi, pi]
//    return toFirstLogBranch(res + (negDetP ? thrust::complex<double>{ 0, pi<double> } : 0));
}
