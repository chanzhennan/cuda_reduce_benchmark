cmake_minimum_required(VERSION 3.14)

project(
  cuda_reduce_benchmark
  DESCRIPTION "reduce benchmark"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  message("CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT")
  set(CMAKE_INSTALL_PREFIX
      "install"
      CACHE PATH "CMAKE_INSTALL_PREFIX" FORCE)
endif()

# Options
option(USE_SYSTEM_LIBS "use system libs or not" OFF)

# Find cuda SDK
find_package(CUDA REQUIRED)

if(USE_SYSTEM_LIBS)
  find_package(GTest REQUIRED)
  find_package(benchmark REQUIRED)
else()

  include(FetchContent)
  # googletest
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.11.0
    GIT_SHALLOW TRUE)
  # https://cmake.org/cmake/help/latest/policy/CMP0077.html
  set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
  set(INSTALL_GTEST OFF)
  FetchContent_MakeAvailable(googletest)
  # googlebenchmark

  FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.6.2
    GIT_SHALLOW TRUE)
  set(BENCHMARK_ENABLE_TESTING OFF)
  set(BENCHMARK_ENABLE_INSTALL OFF)
  set(HAVE_GNU_POSIX_REGEX OFF)
  set(BENCHMARK_INSTALL_DOCS OFF)
  FetchContent_MakeAvailable(googlebenchmark)
  FetchContent_GetProperties(googlebenchmark)

endif()

enable_testing()

include_directories(${googlebenchmark_BINARY_DIR}/include)
include_directories(${googlebenchmark_SOURCE_DIR}/include)

include_directories(benchmarks)

add_subdirectory(3rdparty/bmlib)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/)

add_subdirectory(benchmarks)
