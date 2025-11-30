#include <string>
#include <iostream>
#include <cuda_fp16.h>
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
    const std::vector<int> shape = 
        embedding_wmd["shape"].get<std::vector<int>>();
    const float scale = embedding_wmd["scale"];
    const std::string dtype = embedding_wmd["dtype"];
    const int offset = embedding_wmd["offset"];
    const int size = embedding_wmd["size"];

    Embedding* embedding = new Embedding(shape, dtype, scale, offset, size,
                                         *wmd);
    
    // Prepare synthetic input indices
    const int batch_size = 2;
    const int sequence_length = 3;
    const int total_tokens = batch_size * sequence_length;
    std::vector<int> h_input_indices = {0, 1, 2, 1, 0, 3};  // Example indices

    // Allocate device memory
    int* d_input_indices;
    __half* d_output;
    cudaMalloc(&d_input_indices, total_tokens * sizeof(int));
    cudaMalloc(&d_output, total_tokens * shape[1] * sizeof(__half));

    // Copy input indices to device
    cudaMemcpy(d_input_indices, h_input_indices.data(),
               total_tokens * sizeof(int), cudaMemcpyHostToDevice);

    // Run forward pass
    embedding->forward(d_input_indices, batch_size, sequence_length, d_output);

    // Copy output back to host
    std::vector<__half> h_output(total_tokens * shape[1]);
    cudaMemcpy(h_output.data(), d_output,
               total_tokens * shape[1] * sizeof(__half),
               cudaMemcpyDeviceToHost);

    // Print output
    std::cout << "\nEmbedding output:\n";
    for (int i = 0; i < total_tokens; ++i) {
        std::cout << "Token " << i << ": ";
        for (int j = 0; j < shape[1]; ++j) {
            // convert for CPU side inspection
            float val = __half2float(h_output[i * shape[1] + j]);
            std::cout << val << " ";
        }
        std::cout << "\n";
    }

    // Cleanup
    cudaFree(d_input_indices);
    cudaFree(d_output);
    delete embedding;
    delete wmd;

    return 0;
}