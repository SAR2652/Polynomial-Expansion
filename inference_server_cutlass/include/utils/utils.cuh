#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>


std::tuple<int, int> get_threads_and_blocks(int total_tokens,
                                            int threads = 256);

template <typename T>
__device__ inline float to_float(T x) {
    return static_cast<float>(x);
}

template <>
__device__ inline float to_float<__half>(__half x) {
    return __half2float(x);
}

template <>
__device__ inline float to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <typename T>
__global__ void quantize_to_int8(const T* input, int8_t* output,
                                 int N, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = to_float(input[idx]);
        int q = static_cast<int>(roundf(val / scale));
        q = max(-128, min(127, q));
        output[idx] = static_cast<int8_t>(q);
    }
}
