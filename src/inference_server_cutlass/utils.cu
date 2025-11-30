#include <iostream>
#include <cuda_fp16.h>
#include "utils.h"

WeightsMetadata::WeightsMetadata(const std::string json_path,
                                 const std::string weights_bin_path)
{
    std::cout << "JSON PATH = " << json_path << std::endl;
    std::cout << "BINARY PATH = " << weights_bin_path << std::endl;
    std::ifstream jf(json_path);
    if (!jf) {
        throw std::invalid_argument("Failed to open metadata.json\n");
    }
    jf >> this->metadata;

    this->weights_bin.open(weights_bin_path, std::ios::binary);
    if (!weights_bin) {
        throw std::invalid_argument("Failed to open weights.bin\n");
    }
}


std::vector<char> WeightsMetadata::get_data(
    std::string dtype, int offset, int size
)
{
    size_t nbytes;
    if(dtype == "int8")
    {
        nbytes = size * sizeof(int8_t);
    }
    else if(dtype == "int32")
    {
        nbytes = size * sizeof(int32_t);
    }
    else if(dtype == "float16")
    {
        nbytes = size * sizeof(uint16_t);
    }

    std::vector<char> buffer(nbytes);
    this->weights_bin.seekg(offset, std::ios::beg);
    this->weights_bin.read(buffer.data(), nbytes);

    return buffer;
}


std::tuple<int, int> get_threads_and_blocks(int total_tokens,
                                            int threads)
{
    int blocks = (total_tokens + threads - 1) / threads;
    return {threads, blocks};
}


__global__ void quantize_half_to_int8(__half* input, int8_t* output, 
                                      int N, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = __half2float(input[idx]);
        int q = static_cast<int>(roundf(val / scale));
        q = max(-128, min(127, q));
        output[idx] = static_cast<int8_t>(q);
    }
}
