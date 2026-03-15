#include <iostream>
#include <string>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <typeinfo> // Required for typeid
#include "utils/utils.h"
#include "utils/utils.cuh"
#include "layers/embedding.h"
#include "layers/lstmcell.h"


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
    const float embedding_scale = embedding_wmd["scale"];

    Embedding* embedding = new Embedding(
        embedding_shape, embedding_dtype,
        embedding_scale, embedding_offset,
        embedding_size, *wmd
    );

    // -------------------------
    // Prepare input
    // -------------------------
    const int batch_size = 2;
    const int sequence_length = 3;
    int total_tokens = batch_size * sequence_length;

    std::vector<int> h_input_indices = {1, 0, 3, 2, 5, 4};

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
    // Allocate embedding output    // -------------------------
    void* embedding_output = nullptr;
    int mul_factor = (embedding_dtype == "float16")
                        ? sizeof(__half)
                        : sizeof(__nv_bfloat16);

    int total_embedding_size = total_tokens * embedding_shape[1];

    cudaMallocAsync(
        &embedding_output,
        total_embedding_size * mul_factor,
        stream
    );

    // -------------------------
    // Allocate quantized int8 buffer
    // -------------------------
    int8_t* quantized_embedding_int8;
    cudaMallocAsync(
        &quantized_embedding_int8,
        total_embedding_size * sizeof(int8_t),
        stream
    );

    // -------------------------
    // Run embedding forward (async)
    // -------------------------
    embedding->forward(
        d_input_indices,
        batch_size,
        sequence_length,
        embedding_output,
        quantized_embedding_int8,
        stream
    );

    // -------------------------
    // LSTMCell
    // -------------------------
    auto encoder_fwd_lstm_wmd = wmd->metadata["encoder"]["forward_lstm"];

    auto kernel_tag = dtype_to_tag(
        encoder_fwd_lstm_wmd["hf"]["kernel"]["dtype"]
    );
    auto bias_tag = dtype_to_tag(
        encoder_fwd_lstm_wmd["hf"]["bias"]["dtype"]
    );

    if (kernel_tag == DTypeTag::Int8 && bias_tag == DTypeTag::Int32)
    {
        using KernelType = int8_t;
        using BiasType   = int32_t;

        float* fwd_hidden;
        float* fwd_cell;

        cudaMallocAsync(
            &fwd_hidden,
            batch_size * embedding_shape[1] * sizeof(float),
            stream
        );
        cudaMallocAsync(
            &fwd_cell,
            batch_size * embedding_shape[1] * sizeof(float),
            stream
        );

        cudaMemsetAsync(
            fwd_hidden, 0,
            batch_size * embedding_shape[1] * sizeof(float),
            stream
        );
        cudaMemsetAsync(
            fwd_cell, 0,
            batch_size * embedding_shape[1] * sizeof(float),
            stream
        );

        auto* lstmcell = new LSTMCell<KernelType, BiasType>(
            encoder_fwd_lstm_wmd, *wmd
        );

        // lstmcell->forward(..., stream);

        delete lstmcell;

        cudaFreeAsync(fwd_hidden, stream);
        cudaFreeAsync(fwd_cell, stream);
    }
    else
    {
        throw std::runtime_error("Unsupported kernel/bias dtype combination");
    }

    // -------------------------
    // Cleanup
    // -------------------------
    cudaFreeAsync(d_input_indices, stream);
    cudaFreeAsync(embedding_output, stream);
    cudaFreeAsync(quantized_embedding_int8, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    delete embedding;
    delete wmd;

    return 0;
}
