#include <vector>
#include "layers.h"
#include "utils.h"
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>


template <typename T>
T* Layer::load_to_device(const std::vector<char>& buffer) const {
    size_t num_elements = buffer.size() / sizeof(T);
    T* device_ptr = nullptr;

    cudaError_t err = cudaMalloc(&device_ptr,
        num_elements * sizeof(T));
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed: "
        + std::string(cudaGetErrorString(err)));
    }

    err = cudaMemcpy(device_ptr, buffer.data(),
    num_elements * sizeof(T),
                    cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(device_ptr);
        throw std::runtime_error("cudaMemcpy failed: " +
        std::string(cudaGetErrorString(err)));
    }

    return device_ptr;
}

template <typename T>
T* Layer::extract_details_and_load_parameters(
    nlohmann::json& param_md, const std::string& param_type,
    WeightsMetadata& wmd) const
{
    int offset = param_md[param_type]["offset"];
    int size = param_md[param_type]["size"];
    std::string dtype = param_md[param_type]["dtype"];
    std::vector<char> buffer = wmd.get_data(dtype, offset, size);
    T* device_ptr = load_to_device<T>(buffer);
    return device_ptr;
}


// Embedding methods
Embedding::Embedding(const std::vector<int> shape, const std::string dtype,
                     const float scale, const int offset, const int size,
                     WeightsMetadata& metadata)
{
    this->shape = shape;
    this->dtype = dtype; 
    this->scale = scale;
    std::vector<char> buffer = metadata.get_data(dtype, offset, size);

    if (dtype == "float16")
    {
        embedding = load_to_device<__half>(buffer);
    }

}


__global__ void embedding_lookup_kernel(
    const __half* __restrict__ embedding_table,
    const int* __restrict__ input_indices,
    __half* __restrict__ output,
    int embedding_dim,
    int sequence_length,
    int batch_size
)
{
    int token_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_tokens = batch_size * sequence_length;

    if (token_id < total_tokens) {
        int index = input_indices[token_id];
        for (int i = 0; i < embedding_dim; ++i) {
            output[token_id * embedding_dim + i] = 
            embedding_table[index * embedding_dim + i];
        }
    }
}

float Embedding::get_embedding_scale()
{
    return scale;
}

void Embedding::forward(int* input_indices, int batch_size,
    int sequence_length, __half* output)
{

    int total_tokens = batch_size * sequence_length;
    int threads = 256;
    int blocks = (total_tokens + threads - 1) / threads;
    int embedding_dim = shape[1];

    embedding_lookup_kernel<<<blocks, threads>>>(
        embedding,
        input_indices,
        output,
        embedding_dim,
        sequence_length,
        batch_size
    );

    cudaDeviceSynchronize(); // Optional for debugging
}

Embedding::~Embedding()
{
    cudaFree(embedding);
}


// LSTMCell methods
template <typename KernelType, typename BiasType>
LSTMCell<KernelType, BiasType>::LSTMCell(
    nlohmann::json lstm_metadata, WeightsMetadata& metadata)
{
    // HF
    auto hf_md = lstm_metadata["hf"];
    BiasType* hf_bias = 
        extract_details_and_load_parameters<BiasType>(
            hf_md, "bias", metadata);
    KernelType* hf_kernel = 
        extract_details_and_load_parameters<KernelType>(
            hf_md, "kernel", metadata);

    // HG
    auto hg_md = lstm_metadata["hg"];
    BiasType* hg_bias = 
        extract_details_and_load_parameters<BiasType>(
            hg_md, "bias", metadata);
    KernelType* hg_kernel = 
        extract_details_and_load_parameters<KernelType>(
            hg_md, "kernel", metadata);

    // HI
    auto hi_md = lstm_metadata["hi"];

    BiasType* hi_bias = 
        extract_details_and_load_parameters<BiasType>(
            hi_md, "bias", metadata);
    KernelType* hi_kernel = 
        extract_details_and_load_parameters<KernelType>(
            hi_md, "kernel", metadata);

    // HO
    auto ho_md = lstm_metadata["ho"];

    BiasType* ho_bias = 
        extract_details_and_load_parameters<BiasType>(
            ho_md, "bias", metadata);
    KernelType* ho_kernel = 
        extract_details_and_load_parameters<KernelType>(
            ho_md, "kernel", metadata);

    // IF
    auto if_md = lstm_metadata["if"];
    BiasType* if_bias = 
        extract_details_and_load_parameters<BiasType>(
            if_md, "bias", metadata);
    KernelType* if_kernel = 
        extract_details_and_load_parameters<KernelType>(
            if_md, "kernel", metadata);

    // IG
    auto ig_md = lstm_metadata["ig"];
    BiasType* ig_bias = 
        extract_details_and_load_parameters<BiasType>(
            ig_md, "bias", metadata);
    KernelType* ig_kernel = 
        extract_details_and_load_parameters<KernelType>(
            ig_md, "kernel", metadata);

    // II
    auto ii_md = lstm_metadata["ii"];

    BiasType* ii_bias = 
        extract_details_and_load_parameters<BiasType>(
            ii_md, "bias", metadata);
    KernelType* ii_kernel = 
        extract_details_and_load_parameters<KernelType>(
            ii_md, "kernel", metadata);

    // IO
    auto io_md = lstm_metadata["io"];

    BiasType* io_bias = 
        extract_details_and_load_parameters<BiasType>(
            io_md, "bias", metadata);
    KernelType* io_kernel = 
        extract_details_and_load_parameters<KernelType>(
            io_md, "kernel", metadata);

}


LSTMCell::~LSTMCell()
{
    cudaFree(hf_bias);
    cudaFree(hf_kernel);

    cudaFree(hg_bias);
    cudaFree(hg_kernel);

    cudaFree(hi_bias);
    cudaFree(hi_kernel);

    cudaFree(ho_bias);
    cudaFree(ho_kernel);

    cudaFree(if_bias);
    cudaFree(if_kernel);

    cudaFree(ig_bias);
    cudaFree(ig_kernel);

    cudaFree(ii_bias);
    cudaFree(ii_kernel);

    cudaFree(io_bias);
    cudaFree(io_kernel);
}
