add_executable(tests
               test_main.cpp
               logdetM.cpp
               ../src/lattice.cpp
               ../src/lattice.hpp
               ../src/util.hpp)

target_include_directories(tests PRIVATE ../src)
target_link_libraries(tests PRIVATE isle_cpu isle_gpu)
