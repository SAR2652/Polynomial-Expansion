#include <iostream>
#include <string>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <typeinfo>
#include "utils/utils.h"
#include "utils/utils.cuh"
#include "layers/embedding.h"
#include "layers/lstmcell.h"


enum class DTypeTag { Int8, Int32, Float32 };

inline DTypeTag dtype_to_tag(const std::string& s) {
    if (s == "int8")    return DTypeTag::Int8;
    if (s == "int32")   return DTypeTag::Int32;
    if (s == "float32") return DTypeTag::Float32;
    throw std::runtime_error("unknown dtype");
}

template <DTypeTag> struct dtype_map;
template <> struct dtype_map<DTypeTag::Int8>    { using type = int8_t; };
template <> struct dtype_map<DTypeTag::Int32>   { using type = int32_t; };
template <> struct dtype_map<DTypeTag::Float32> { using type = float; };

template <DTypeTag tag>
using dtype_t = typename dtype_map<tag>::type;


int main()
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::string json_path = std::filesystem::absolute(
        "../../../output/metadata.json").string();
    std::string bin_path = std::filesystem::absolute(
        "../../../output/weights.bin").string();

    WeightsMetadata* wmd = new WeightsMetadata(json_path, bin_path);

    auto embedding_wmd = wmd->metadata["encoder"]["embedding"]["embedding"];
    const std::vector<int> embedding_shape =
        embedding_wmd["shape"].get<std::vector<int>>();
    const std::string embedding_dtype = embedding_wmd["dtype"];
    const int embedding_offset = embedding_wmd["offset"];
    const int embedding_size   = embedding_wmd["size"];
    const int embedding_dim    = embedding_shape[1];

    const float scale_x = wmd->metadata["calibration"]["scale_x"];

    Embedding* embedding = new Embedding(
        embedding_shape, embedding_dtype,
        embedding_wmd["scale"], embedding_offset,
        embedding_size, *wmd
    );

    // -------------------------
    // Prepare input
    // Input indices are laid out [seq, batch] so that at timestep t the
    // contiguous slice [t*batch_size .. (t+1)*batch_size) holds all batch
    // items for that step — matching embeddings[:, t, :] in PyTorch.
    // -------------------------
    const int batch_size = 6;
    const int seq_len    = 4;
    const int total_tokens = batch_size * seq_len;   // 24

    // Layout: row 0 = t=0 tokens, row 1 = t=1 tokens, …
    std::vector<int> h_input_indices = {
        1, 6, 3, 2, 5, 4,   // t=0
        2, 5, 1, 4, 3, 6,   // t=1
        3, 4, 2, 6, 1, 5,   // t=2
        4, 3, 6, 1, 2, 5,   // t=3
    };

    int* d_input_indices;
    cudaMallocAsync(&d_input_indices, total_tokens * sizeof(int), stream);
    cudaMemcpyAsync(
        d_input_indices,
        h_input_indices.data(),
        total_tokens * sizeof(int),
        cudaMemcpyHostToDevice,
        stream
    );

    

    // -------------------------
    // Cleanup
    // -------------------------
    cudaFreeAsync(d_input_indices, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return 0;
}
