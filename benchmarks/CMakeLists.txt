add_subdirectory(ipubm_lib)

set(SOURCE_FILES ipu_benchmark.cc)

list(
  APPEND
  SOURCE_FILES
  )

add_executable(cuda_benchmark ${SOURCE_FILES})
target_link_libraries(
  cuda_benchmark
  gcl
  ipubm_lib)

install(TARGETS cuda_benchmark RUNTIME)
