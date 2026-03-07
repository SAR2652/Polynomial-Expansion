#include <iostream>
#include <string>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <typeinfo> // Required for typeid
#include "../../../src/inference_server_cutlass/utils.h"
#include "../../../src/inference_server_cutlass/layers.h"


enum class DTypeTag { Int8, Int32, Float32 };

inline DTypeTag dtype_to_tag(const std::string& s) {
    if (s == "int8")  return DTypeTag::Int8;
    if (s == "int32") return DTypeTag::Int32;
    if (s == "float32") return DTypeTag::Float32;
    throw std::runtime_error("unknown dtype");
}

template <DTypeTag> struct dtype_map;

template <> struct dtype_map<DTypeTag::Int8>  { using type = int8_t; };
template <> struct dtype_map<DTypeTag::Int32> { using type = int32_t; };
template <> struct dtype_map<DTypeTag::Float32> { using type = float; };

template <DTypeTag tag>
using dtype_t = typename dtype_map<tag>::type;


int main()
{
    std::string json_path = std::filesystem::absolute(
        "../../../output/metadata.json")
        .string();
    std::string bin_path = std::filesystem::absolute(
        "../../../output/weights.bin")
        .string();

    std::cout << json_path << std::endl;

    WeightsMetadata* wmd = new WeightsMetadata(json_path, bin_path);

    auto embedding_wmd = wmd->metadata["encoder"]["embedding"]["embedding"];
    std::cout << "Embedding shape JSON = " << embedding_wmd["shape"] << std::endl;
    const std::vector<int> embedding_shape =
        embedding_wmd["shape"].get<std::vector<int>>();
    const std::string embedding_dtype = embedding_wmd["dtype"];
    const int embedding_offset = embedding_wmd["offset"];
    const int embedding_size = embedding_wmd["size"];
    const float embedding_scale = embedding_wmd["scale"];

    std::cout << "Reading embedding params" << std::endl;

    Embedding* embedding = new Embedding(embedding_shape, embedding_dtype,
                                         embedding_scale, embedding_offset,
                                         embedding_size, *wmd);
    
    std::cout << "Read embedding params" << std::endl;

    // Prepare synthetic input indices
    const int batch_size = 2;
    const int sequence_length = 3;
    int total_tokens = batch_size * sequence_length;

    std::cout << "Read total tokens" << std::endl;

    std::vector<int> h_input_indices = {0, 1, 2, 1, 0, 3};  // Example indices

    std::cout << "Input indices" << std::endl;

    // Allocate device memory
    int* d_input_indices;
    cudaMalloc(&d_input_indices, total_tokens * sizeof(int));

    void* embedding_output = nullptr;
    int mul_factor = 0;

    if(embedding_dtype == "float16")
    {
        mul_factor = sizeof(__half);
    }
    else if(embedding_dtype == "bfloat16")
    {
        mul_factor = sizeof(__nv_bfloat16);
    }

    int total_embedding_size = total_tokens * embedding_shape[1];

    cudaMalloc(&embedding_output, total_embedding_size * mul_factor);
                                  // 2 x 3 x 64 x dtype_cost

    // Copy input indices to device
    cudaMemcpy(d_input_indices, h_input_indices.data(),
               total_tokens * sizeof(int), cudaMemcpyHostToDevice);

    int8_t* quantized_embedding_int8;
    cudaMalloc(&quantized_embedding_int8,
               total_embedding_size * sizeof(int8_t));
               // 2 x 3 x 64 x dtype_cost for int8

    // Run forward pass
    embedding->forward(d_input_indices, batch_size, sequence_length,
                       embedding_output, quantized_embedding_int8);

    // Cleanup
    cudaFree(d_input_indices);
    cudaFree(embedding_output);
    cudaFree(quantized_embedding_int8);

}