cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

# project name
project(polynomial_expansion LANGUAGES CXX CUDA)

# set(CMAKE_CXX_STANDARD 17)
# set(CXX_STANDARD 17)
# set(CUDA_STANDARD 17)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set your target GPU architecture (adjust if needed)
set(CMAKE_CUDA_ARCHITECTURES 80)

enable_testing()

add_library(inference_server_cutlass
    src/inference_server_cutlass/utils.cu
    src/inference_server_cutlass/layers.cu
)

target_include_directories(inference_server_cutlass PUBLIC
    ${PROJECT_SOURCE_DIR}/src/inference_server_cutlass
)

# Add GoogleTest (assumes you have it in third_party/googletest)
add_subdirectory(external/googletest)

# CUTLASS root directory (update this path!)
set(CUTLASS_DIR "external/cutlass")

# Include CUTLASS headers
include_directories(
  ${CUTLASS_DIR}/include
  ${CUTLASS_DIR}/tools/util/include
)

# Add your test suite
add_subdirectory(tests/inference_server_cutlass)
