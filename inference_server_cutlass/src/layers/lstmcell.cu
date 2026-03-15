#include "layers/layer.cuh"
#include "layers/lstmcell.h"


// LSTMCell methods
template <typename KernelType, typename BiasType>
LSTMCell<KernelType, BiasType>::LSTMCell(
    const nlohmann::json lstm_metadata, const WeightsMetadata& metadata)
{
    // HF
    auto hf_md = lstm_metadata["hf"];
    hf_bias = extract_details_and_load_parameters<BiasType>(
        hf_md, "bias", metadata
    );
    hf_kernel = extract_details_and_load_parameters<KernelType>(
        hf_md, "kernel", metadata
    );

    // HG
    auto hg_md = lstm_metadata["hg"];
    hg_bias = extract_details_and_load_parameters<BiasType>(
        hg_md, "bias", metadata
    );
    hg_kernel = extract_details_and_load_parameters<KernelType>(
        hg_md, "kernel", metadata
    );

    // HI
    auto hi_md = lstm_metadata["hi"];

    hi_bias = extract_details_and_load_parameters<BiasType>(
        hi_md, "bias", metadata
    );
    hi_kernel = extract_details_and_load_parameters<KernelType>(
        hi_md, "kernel", metadata
    );

    // HO
    auto ho_md = lstm_metadata["ho"];

    ho_bias = extract_details_and_load_parameters<BiasType>(
        ho_md, "bias", metadata
    );
    ho_kernel = extract_details_and_load_parameters<KernelType>(
        ho_md, "kernel", metadata
    );

    // IF
    auto if_md = lstm_metadata["if"];
    if_kernel = extract_details_and_load_parameters<KernelType>(
        if_md, "kernel", metadata
    );

    // IG
    auto ig_md = lstm_metadata["ig"];
    ig_kernel = extract_details_and_load_parameters<KernelType>(
        ig_md, "kernel", metadata
    );

    // II
    auto ii_md = lstm_metadata["ii"];
    ii_kernel = extract_details_and_load_parameters<KernelType>(
        ii_md, "kernel", metadata
    );

    // IO
    auto io_md = lstm_metadata["io"];
    io_kernel = extract_details_and_load_parameters<KernelType>(
        io_md, "kernel", metadata
    );

}

template <typename KernelType, typename BiasType>
LSTMCell<KernelType, BiasType>::~LSTMCell()
{
    cudaFreeAsync(hf_bias, stream_);
    cudaFreeAsync(hf_kernel, stream_);

    cudaFreeAsync(hg_bias, stream_);
    cudaFreeAsync(hg_kernel, stream_);

    cudaFreeAsync(hi_bias, stream_);
    cudaFreeAsync(hi_kernel, stream_);

    cudaFreeAsync(ho_bias, stream_);
    cudaFreeAsync(ho_kernel, stream_);

    cudaFreeAsync(if_kernel, stream_);
    cudaFreeAsync(ig_kernel, stream_);
    cudaFreeAsync(ii_kernel, stream_);
    cudaFreeAsync(io_kernel, stream_);
}


// __device__ inline float sigmoidf(float x) {
//     return 1.f / (1.f + expf(-x));
// }

// __device__ inline float tanhf_approx(float x) {
//     return tanhf(x);
// }

// template <typename KernelType, typename BiasType>
// void LSTMCell<KernelType, BiasType>::run_gate(
//     KernelType* x, KernelType* h, BiasType* gate_out, int batch_size,
//     int input_dim, int hidden_dim, bool use_sigmoid
// )
// {
//     // ----------------------
//     // CUTLASS device GEMM
//     // ----------------------
//     using Gemm = typename GemmConfigSm80<KernelType, BiasType>::Gemm;

//     // workspace: tmp = x @ Wx
//     BiasType *tmp;
//     cudaMalloc(&tmp, sizeof(BiasType) * M * N);

    
// }

// template <typename KernelType, typename BiasType>
// void LSTMCell<KernelType, BiasType>::forward(KernelType* quantized_embeddings,
//     float* hidden, float* cell)
// {
//     // ----------------------
//     // CUTLASS device GEMM
//     // ----------------------
//     using Gemm = typename GemmConfigSm80<KernelType, BiasType>::Gemm;

//     // Within {kernel, input_size} , the input_size is the stride

//     BiasType* 

//     typename Gemm::Arguments args{
//         {total_tokens, output_size, input_size},  // Gemm dimensions M,N,K
//         {input, input_size},                       // A matrix (RowMajor)
//         {kernel, input_size},                      // B matrix (ColumnMajor)
//         // bias broadcast across rows
//         {bias, 0},                                 // C matrix (RowMajor)
//         {output, output_size},                     // D matrix (RowMajor)
//         {1, 1}                                     // alpha=1, beta=1
//     };

//     Gemm gemm_op;

//     // Allocate workspace if needed
//     size_t workspace_size = gemm_op.get_workspace_size(args);
//     void* workspace = nullptr;
//     if (workspace_size > 0) {
//         cudaMalloc(&workspace, workspace_size);
//     }

//     // Initialize GEMM
//     cutlass::Status status = gemm_op.initialize(args, workspace);
//     if (status != cutlass::Status::kSuccess) {
//         std::cerr << "GEMM initialize failed: " << int(status) << "\n";
//         if (workspace) cudaFree(workspace);
//         return;
//     }

//     // Launch GEMM
//     status = gemm_op();
//     if (status != cutlass::Status::kSuccess) {
//         std::cerr << "GEMM launch failed: " << int(status) << "\n";
//     }

//     if (workspace) cudaFree(workspace);
// }

template class LSTMCell<int8_t, int32_t>;
