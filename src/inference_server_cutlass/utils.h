#include <cuda_fp16.h>
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
                            int offset, int size);
                
        nlohmann::json metadata;
        std::ifstream weights_bin;

};

std::tuple<int, int> get_threads_and_blocks(int total_tokens,
                                            int threads = 256);

__global__ void quantize_half_to_int8(__half* input, int8_t* output,
                                      int N, float scale);
