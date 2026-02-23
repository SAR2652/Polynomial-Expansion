#include <vector>
#include "layers.h"
#include "utils.h"
#include <cstdint>

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm.h"


#pragma once

using LayoutRM = cutlass::layout::RowMajor;
using LayoutCM = cutlass::layout::ColumnMajor;


// Specifically for int8 on RTX 5070
// using ArchTag       = cutlass::arch::Sm80;   // RTX 50 series
// using OperatorClass = cutlass::arch::OpClassTensorOp;
// const int Alignment = 16;
// using TileShape = cutlass::gemm::GemmShape<128, 128, 32>;
// using ClusterShape = cutlass::gemm::GemmShape<1, 1, 1>;
// using StageCountType      = cutlass::gemm::collective::StageCountAuto;
// using KernelScheduleType  = cutlass::gemm::collective::KernelScheduleAuto;
// using EpilogueScheduleType = cutlass::epilogue::collective::EpilogueScheduleAuto;
// using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;  // IMMA


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
    nlohmann::json param_md, const std::string& param_type,
    const WeightsMetadata& wmd) const
{
    int offset = param_md[param_type]["offset"];
    int size = param_md[param_type]["size"];
    std::string dtype = param_md[param_type]["dtype"];
    std::vector<char> buffer = wmd.get_data(dtype, offset, size);
    T* device_ptr = load_to_device<T>(buffer);
    return device_ptr;
}


// Embedding methods
Embedding::Embedding(const std::vector<int> shape,
                     const std::string dtype,
                     const float scale,
                     const int offset,
                     const int size,
                     WeightsMetadata& metadata)
{
    this->shape = shape;
    this->dtype = dtype;
    this->scale = scale;

    std::vector<char> buffer = metadata.get_data(dtype, offset, size);

    if (dtype == "bfloat16")
    {
        embedding = load_to_device<__nv_bfloat16>(buffer);
    }
    else if (dtype == "int8")
    {
        embedding = load_to_device<int8_t>(buffer);
    }
    else if (dtype == "float32")
    {
        embedding = load_to_device<float>(buffer);
    }
    else
    {
        throw std::runtime_error("Unsupported dtype: " + dtype);
    }
}


// __restrict__ is a guarantee that the pointer points to only one specific
// memory location during its lifetime
template <typename T>
__global__ void embedding_lookup_kernel_t(
    const T* __restrict__ embedding_table,
    const int* __restrict__ input_indices,
    T* __restrict__ output,
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

template <typename T>
void forward_impl(void* embedding_raw,
                  int* input_indices,
                  int B, int S,
                  void* output_raw,
                  int embedding_dim)
{
    T* embedding = static_cast<T*>(embedding_raw);
    T* output = static_cast<T*>(output_raw);

    int total_tokens = B * S;
    int threads = 256;
    int blocks = (total_tokens + threads - 1) / threads;

    embedding_lookup_kernel_t<T><<<blocks, threads>>>(
        embedding,
        input_indices,
        output,
        embedding_dim,
        S,
        B
    );
}



float Embedding::get_embedding_scale()
{
    return scale;
}

void Embedding::forward(int* input_indices,
                        int batch_size,
                        int sequence_length,
                        void* output,
                        int8_t* quantized_embedding_int8,
                        float embedding_scale)
{
    int embedding_dim = shape[1];
    float new_embedding_scale = embedding_scale / 255.0f;   // for int8
    int total_tokens = batch_size * sequence_length;
    auto [blocks, threads] = get_threads_and_blocks(total_tokens);

    if (dtype == "float16")
    {
        forward_impl<__half>(
            embedding,
            input_indices,
            batch_size,
            sequence_length,
            output,
            embedding_dim
        );

        quantize_to_int8<<<blocks, threads>>>(
            static_cast<__half*>(output),
            quantized_embedding_int8,
            total_tokens, new_embedding_scale
        );
    }
    else if (dtype == "bfloat16")
    {
        forward_impl<__nv_bfloat16>(
            embedding,
            input_indices,
            batch_size,
            sequence_length,
            output,
            embedding_dim
        );
        quantize_to_int8<<<blocks, threads>>>(
            static_cast<__nv_bfloat16*>(output),
            quantized_embedding_int8,
            total_tokens, new_embedding_scale
        );
    }
    else
    {
        throw std::runtime_error("Unsupported dtype: " + dtype);
    }

}


Embedding::~Embedding()
{
    cudaFree(embedding);
}


// Linear layer methods
template <typename KernelType, typename BiasType>
Linear<KernelType, BiasType>::Linear(
    const nlohmann::json linear_md, const WeightsMetadata& metadata)
{
    bias = 
        extract_details_and_load_parameters<BiasType>(
            linear_md, "bias", metadata);
    kernel = 
        extract_details_and_load_parameters<KernelType>(
            linear_md, "kernel", metadata);
}


template <typename KernelType, typename BiasType>
Linear<KernelType, BiasType>::~Linear()
{
    cudaFree(bias);
    cudaFree(kernel);
}


template <typename KernelType, typename BiasType>
void Linear<KernelType, BiasType>::forward(
    KernelType* input,
    BiasType*   output,
    int         total_tokens,
    int         input_size,
    int         output_size
)
{
    // ----------------------
    // CUTLASS device GEMM
    // ----------------------
    using Gemm = cutlass::gemm::device::Gemm<
        KernelType, LayoutRM,    // A matrix
        KernelType, LayoutCM,    // B matrix
        BiasType,  LayoutRM,     // C matrix
        BiasType,                // accumulator/output type
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80
    >;

    // using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    //     cutlass::gemm::GemmShape<128,128,32>,       // Tile shape (M,N,K)
    //     KernelType, LayoutRM,                        // A matrix
    //     KernelType, LayoutCM,                        // B matrix
    //     BiasType,  LayoutRM,                         // C matrix
    //     BiasType,                                    // Accumulator
    //     OperatorClass,
    //     ArchTag,
    //     cutlass::gemm::GemmShape<16,8,32>           // Instruction shape
    // >;

    typename Gemm::Arguments args{
        {total_tokens, output_size, input_size},  // Gemm dimensions M,N,K
        {input, input_size},                       // A matrix (RowMajor)
        {kernel, input_size},                      // B matrix (ColumnMajor)
        {bias, output_size},                       // C matrix (RowMajor)
        {output, output_size},                     // D matrix (RowMajor)
        {1, 0}                                     // alpha=1, beta=1
    };

    Gemm gemm_op;

    // Allocate workspace if needed
    size_t workspace_size = gemm_op.get_workspace_size(args);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    // Initialize GEMM
    cutlass::Status status = gemm_op.initialize(args, workspace);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM initialize failed: " << int(status) << "\n";
        if (workspace) cudaFree(workspace);
        return;
    }

    // Launch GEMM
    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM launch failed: " << int(status) << "\n";
    }

    if (workspace) cudaFree(workspace);
}


// template <typename KernelType, typename BiasType>
// void Linear<KernelType, BiasType>::forward(
//     KernelType* input,
//     BiasType*   output,
//     int         total_tokens,
//     int         input_size,
//     int         output_size
// )
// {
//     // Problem / instruction shapes
//     using ProblemShape     = cutlass::gemm::GemmShape<128, 128, 32>;   // threadblock (M,N,K)
//     using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;      // mma instruction shape

//     // Build collective mainloop / epilogue types via CollectiveBuilder
//     using CollectiveBuilder = cutlass::gemm::collective::CollectiveBuilder<
//         ArchTag,                    // architecture tag (e.g., Sm80)
//         OperatorClass,              // operator class (TensorOp)
//         KernelType, LayoutRM,       // ElementA, LayoutA (A is RowMajor)
//         KernelType, LayoutCM,       // ElementB, LayoutB (B is ColumnMajor)
//         BiasType,  LayoutRM,        // ElementC/Accumulator layout (accumulator type/layout)
//         ProblemShape,               // Problem shape (threadblock)
//         InstructionShape            // Instruction shape
//     >;

//     using CollectiveMainloop = typename CollectiveBuilder::CollectiveMainloop;
//     using CollectiveEpilogue = typename CollectiveBuilder::CollectiveEpilogue;

//     // Gemm kernel and device adapter
//     using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
//         ProblemShape,
//         CollectiveMainloop,
//         CollectiveEpilogue
//     >;

//     using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
//     Gemm gemm_op;

//     // Arguments: (M,N,K), A, B, C, D, alpha/beta
//     typename Gemm::Arguments args{
//         { total_tokens, output_size, input_size },   // Gemm dimensions M,N,K
//         { input, input_size },                       // A matrix (RowMajor)  -> [M x K], lda = K
//         { kernel, input_size },                      // B matrix (ColumnMajor) -> [K x N], ldb = K for ColumnMajor
//         { bias, output_size },                       // C matrix (RowMajor) -> bias (broadcast along M)
//         { output, output_size },                     // D matrix (RowMajor) -> output
//         { 1, 1 }                                     // alpha=1, beta=1 (C contributes)
//     };

//     // Allocate workspace if required by the kernel
//     size_t workspace_size = gemm_op.get_workspace_size(args);
//     void* workspace = nullptr;
//     if (workspace_size > 0) {
//         cudaError_t cuerr = cudaMalloc(&workspace, workspace_size);
//         if (cuerr != cudaSuccess) {
//             std::cerr << "cudaMalloc workspace failed: " << cudaGetErrorString(cuerr) << "\n";
//             return;
//         }
//     }

//     // Initialize and launch
//     cutlass::Status status = gemm_op.initialize(args, workspace);
//     if (status != cutlass::Status::kSuccess) {
//         std::cerr << "GEMM initialize failed: " << int(status) << "\n";
//         if (workspace) cudaFree(workspace);
//         return;
//     }

//     status = gemm_op();
//     if (status != cutlass::Status::kSuccess) {
//         std::cerr << "GEMM launch failed: " << int(status) << "\n";
//     }

//     if (workspace) cudaFree(workspace);
// }


template class Linear<int8_t, int32_t>;



// LSTMCell methods
// template <typename KernelType, typename BiasType>
// LSTMCell<KernelType, BiasType>::LSTMCell(
//     const nlohmann::json lstm_metadata, const WeightsMetadata& metadata)
// {
//     // HF
//     auto hf_md = lstm_metadata["hf"];
//     BiasType* hf_bias = 
//         extract_details_and_load_parameters<BiasType>(
//             hf_md, "bias", metadata);
//     KernelType* hf_kernel = 
//         extract_details_and_load_parameters<KernelType>(
//             hf_md, "kernel", metadata);

//     // HG
//     auto hg_md = lstm_metadata["hg"];
//     BiasType* hg_bias = 
//         extract_details_and_load_parameters<BiasType>(
//             hg_md, "bias", metadata);
//     KernelType* hg_kernel = 
//         extract_details_and_load_parameters<KernelType>(
//             hg_md, "kernel", metadata);

//     // HI
//     auto hi_md = lstm_metadata["hi"];

//     BiasType* hi_bias = 
//         extract_details_and_load_parameters<BiasType>(
//             hi_md, "bias", metadata);
//     KernelType* hi_kernel = 
//         extract_details_and_load_parameters<KernelType>(
//             hi_md, "kernel", metadata);

//     // HO
//     auto ho_md = lstm_metadata["ho"];

//     BiasType* ho_bias = 
//         extract_details_and_load_parameters<BiasType>(
//             ho_md, "bias", metadata);
//     KernelType* ho_kernel = 
//         extract_details_and_load_parameters<KernelType>(
//             ho_md, "kernel", metadata);

//     // IF
//     auto if_md = lstm_metadata["if"];
//     BiasType* if_bias = 
//         extract_details_and_load_parameters<BiasType>(
//             if_md, "bias", metadata);
//     KernelType* if_kernel = 
//         extract_details_and_load_parameters<KernelType>(
//             if_md, "kernel", metadata);

//     // IG
//     auto ig_md = lstm_metadata["ig"];
//     BiasType* ig_bias = 
//         extract_details_and_load_parameters<BiasType>(
//             ig_md, "bias", metadata);
//     KernelType* ig_kernel = 
//         extract_details_and_load_parameters<KernelType>(
//             ig_md, "kernel", metadata);

//     // II
//     auto ii_md = lstm_metadata["ii"];

//     BiasType* ii_bias = 
//         extract_details_and_load_parameters<BiasType>(
//             ii_md, "bias", metadata);
//     KernelType* ii_kernel = 
//         extract_details_and_load_parameters<KernelType>(
//             ii_md, "kernel", metadata);

//     // IO
//     auto io_md = lstm_metadata["io"];

//     BiasType* io_bias = 
//         extract_details_and_load_parameters<BiasType>(
//             io_md, "bias", metadata);
//     KernelType* io_kernel = 
//         extract_details_and_load_parameters<KernelType>(
//             io_md, "kernel", metadata);

// }


// template <typename KernelType, typename BiasType>
// LSTMCell<KernelType, BiasType>::~LSTMCell()
// {
//     cudaFree(hf_bias);
//     cudaFree(hf_kernel);

//     cudaFree(hg_bias);
//     cudaFree(hg_kernel);

//     cudaFree(hi_bias);
//     cudaFree(hi_kernel);

//     cudaFree(ho_bias);
//     cudaFree(ho_kernel);

//     cudaFree(if_bias);
//     cudaFree(if_kernel);

//     cudaFree(ig_bias);
//     cudaFree(ig_kernel);

//     cudaFree(ii_bias);
//     cudaFree(ii_kernel);

//     cudaFree(io_bias);
//     cudaFree(io_kernel);
// }
