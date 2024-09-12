add_subdirectory(bm_base)

set(SOURCE_FILES cuda_benchmark.cc)

list(
  APPEND
  SOURCE_FILES
  ReduceArray0/baseline.cc
  ReduceArray0/baseline.cu
  ReduceArray1/no_divergence_branch.cc
  ReduceArray1/no_divergence_branch.cu
  ReduceArray2/no_bank_conflict.cc
  ReduceArray2/no_bank_conflict.cu
  ReduceArray3/add_during_load.cc
  ReduceArray3/add_during_load.cu
  ReduceArray4/unroll_last_warp.cc
  ReduceArray4/unroll_last_warp.cu
  ReduceArray5/completely_unroll.cc
  ReduceArray5/completely_unroll.cu
  ReduceArray6/shuffle.cc
  ReduceArray6/shuffle.cu)

cuda_add_executable(cuda_benchmark ${SOURCE_FILES})
target_link_libraries(cuda_benchmark bm_base bmlib)

install(TARGETS cuda_benchmark RUNTIME)
