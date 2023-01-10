add_subdirectory(bm_lib)

set(SOURCE_FILES cuda_benchmark.cc)

list(
  APPEND
  SOURCE_FILES
  reduce_baseline/reduce_baseline.cc
  reduce_baseline/reduce_baseline.cu
  no_divergence_branch/no_divergence_branch.cc
  no_divergence_branch/no_divergence_branch.cu
  # reduce3/reduce3.cc reduce3/reduce3.cu reduce4/reduce4.cc reduce4/reduce4.cu
  # reduce5/reduce5.cc reduce5/reduce5.cu reduce6/reduce6.cc reduce6/reduce6.cu
)

cuda_add_executable(cuda_benchmark ${SOURCE_FILES})
target_link_libraries(cuda_benchmark bm_lib)

install(TARGETS cuda_benchmark RUNTIME)
