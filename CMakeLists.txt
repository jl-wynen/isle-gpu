cmake_minimum_required(VERSION 3.15.0)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# do not allow any compiler specific c++ extensions
set(CMAKE_CXX_EXTENSIONS OFF)

# define the project
project(isle CUDA CXX)

# look for 3rd party packages
find_package(OpenMP REQUIRED)
find_package(blaze 3.6 REQUIRED)

get_filename_component(SOURCE_DIR "src"
  REALPATH BASE_DIR "${CMAKE_SOURCE_DIR}")
add_subdirectory(${SOURCE_DIR})

enable_testing()
add_subdirectory(test)
