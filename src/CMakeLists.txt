add_executable(gisle
               main.cpp
               lattice.cpp
               lattice.hpp)

# set standard
set_target_properties(gisle PROPERTIES
                      CXX_STANDARD 14
                      CXX_STANDARD_REQUIRED ON)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "RELEASE")
  include(CheckIPOSupported)
  check_ipo_supported(RESULT result)
  if (${result})
    set_target_properties(gisle PROPERTIES
                          INTERPROCEDURAL_OPTIMIZATION True)
    message(STATUS "Enabling link time optimization")
  else ()
    message(STATUS "Cannot enable link time optimization")
  endif ()
  unset(result)
endif ()

# warnings
set(GCC_CLANG_WARNINGS -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wpedantic
    -Wold-style-cast -Wcast-align -Wunused -Woverloaded-virtual
    -Wconversion -Wnull-dereference
    -Wdouble-promotion -Wformat=2)

set(GCC_EXTRA_WARNINGS -Wmisleading-indentation -Wduplicated-cond -Wduplicated-branches
    -Wlogical-op -Wuseless-cast)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(gisle PUBLIC ${GCC_CLANG_WARNINGS} -Wno-c++98-compat)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_compile_options(gisle PUBLIC ${GCC_CLANG_WARNINGS} ${GCC_EXTRA_WARNINGS})
endif ()

add_subdirectory(gpu)
set_target_properties(gisle PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(gisle PRIVATE isle_gpu)

add_subdirectory(cpu)
target_link_libraries(gisle PRIVATE isle_cpu)
