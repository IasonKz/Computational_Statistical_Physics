cmake_minimum_required(VERSION 3.15)
project(heisenberg_dmi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(heisenberg_dmi
    src/main.cpp
    src/spin_lattice.cpp
    src/metropolis.cpp
    src/observables.cpp
)

target_include_directories(heisenberg_dmi PRIVATE src)

target_compile_options(heisenberg_dmi PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3 -march=native -Wall -Wextra -pedantic>
)
