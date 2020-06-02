add_library(isle_cpu STATIC
            core.hpp
            tmp.hpp
            math.hpp
            hubbardFermiMatrixExp.hpp
            hubbardFermiMatrixExp.cpp
            cpu.cpp
            cpu.hpp)

# set standard
set_target_properties(isle_cpu PROPERTIES
                      CXX_STANDARD 14
                      CXX_STANDARD_REQUIRED ON)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "RELEASE")
  include(CheckIPOSupported)
  check_ipo_supported(RESULT result)
  if (${result})
    set_target_properties(isle_cpu PROPERTIES
                          INTERPROCEDURAL_OPTIMIZATION True)
    message(STATUS "Enabling link time optimization")
  else ()
    message(STATUS "Cannot enable link time optimization")
  endif ()
  unset(result)
endif ()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(isle_cpu PUBLIC ${GCC_CLANG_WARNINGS} -Wno-c++98-compat)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_compile_options(isle_cpu PUBLIC ${GCC_CLANG_WARNINGS} ${GCC_EXTRA_WARNINGS})
endif ()

# blaze
target_include_directories(isle_cpu SYSTEM PUBLIC ${blaze_INCLUDE_DIRS})
target_compile_options(isle_cpu PUBLIC "${blaze_CXX_FLAGS}")
target_link_libraries(isle_cpu PUBLIC ${blaze_LIBRARIES})
target_link_options(isle_cpu PUBLIC "${blaze_LINKER_FLAGS}")

# OpenMP if blaze does not use it
if (NOT "${BLAZE_PARALLELISM}" STREQUAL "OMP")
  target_compile_options(isle_cpu PRIVATE "${OpenMP_CXX_FLAGS}")
  target_link_libraries(isle_cpu PRIVATE "${OpenMP_CXX_LIBRARIES}")
  target_link_options(isle_cpu PRIVATE "${OpenMP_LINKER_FLAGS}")
endif ()
