#include <iostream>
#include <string>
#include <cuda_fp16.h>
#include "../../../src/inference_server_cutlass/utils.h"
#include "../../../src/inference_server_cutlass/layers.h"

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
    const std::vector<int> embedding_shape =
        embedding_wmd["shape"].get<std::vector<int>>();
    const std::string dtype = embedding_wmd["dtype"];
    const int offset = embedding_wmd["offset"];
    const int size = embedding_wmd["size"];
    const float scale = embedding_wmd["scale"];

    Embedding* embedding = new Embedding(embedding_shape, dtype, scale, offset,
                                         size, *wmd);
    
    // Prepare synthetic input indices
    const int batch_size = 2;
    const int sequence_length = 3;
    const int total_tokens = batch_size * sequence_length;
    std::vector<int> h_input_indices = {0, 1, 2, 1, 0, 3};  // Example indices

    // Allocate device memory
    int* d_input_indices;
    __half* embedding_output;
    cudaMalloc(&d_input_indices, total_tokens * sizeof(int));
    cudaMalloc(&embedding_output,
               total_tokens * embedding_shape[1] * sizeof(__half));

    // Copy input indices to device
    cudaMemcpy(d_input_indices, h_input_indices.data(),
               total_tokens * sizeof(int), cudaMemcpyHostToDevice);

    // Run forward pass
    embedding->forward(d_input_indices, batch_size, sequence_length,
                       embedding_output);

    // Cleanup
    cudaFree(d_input_indices);

    float embedding_scale = embedding->get_embedding_scale();
    embedding_scale /= 255.0;   // for int8

    int8_t* quantized_embedding_int8;
    
    cudaMalloc(&quantized_embedding_int8,
               total_tokens * embedding_shape[1] * sizeof(int8_t));

    auto [blocks, threads] = get_threads_and_blocks(total_tokens);
    
    quantize_half_to_int8<<<blocks, threads>>>(
        embedding_output, quantized_embedding_int8, total_tokens,
        embedding_scale
    );

    cudaFree(embedding_output);

    cutlass::int8_t* quantized_embedding_int8_cutlass =
        reinterpret_cast<cutlass::int8_t*>(quantized_embedding_int8);

    cudaFree(quantized_embedding);
    
    delete embedding;
    delete wmd;

    return 0;
}