#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>

#pragma once

struct TensorInfo
{
    std::vector<int> shape;
    std::string dtype;
    int offset;
    int size;
};

class WeightsMetadata {

    private:
        TensorInfo get_kernel_bias_data(nlohmann::json& layer_metadata);

    public:
        WeightsMetadata(const std::string json_path,
                const std::string weights_bin_path);
        std::vector<char> get_data(std::string dtype,
                            int offset, int size) const;
                
        nlohmann::json metadata;

        // allowing modification of internal state inside const methods when
        // the state is logically not part of the object’s “constness”.
        mutable std::ifstream weights_bin;

};

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

// template __global__ void quantize_to_int8<__half>(
//     const __half* input, int8_t* output, int N, float scale);

// template __global__ void quantize_to_int8<__nv_bfloat16>(
//     const __nv_bfloat16* input, int8_t* output, int N, float scale);

