add_library(isle_gpu STATIC
            hubbard_fermi_matrix_gpu.cu
            hubbard_fermi_matrix_gpu.cuh
            gpu.cu
            gpu.cuh
            math.cpp
            math.hpp gpumath.cu gpumath.cuh)

# set standard
set_target_properties(isle_gpu PROPERTIES
                      CXX_STANDARD 14
                      CXX_STANDARD_REQUIRED ON)

set_target_properties(isle_gpu
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(isle_gpu -lcublas)
