#include "layers/linear.h"


// Linear layer methods
template <typename KernelType, typename BiasType>
Linear<KernelType, BiasType>::Linear(
    const nlohmann::json linear_md, const WeightsMetadata& metadata)
{
    std::cout << linear_md["bias"]["shape"] << std::endl;
    bias = 
        extract_details_and_load_parameters<BiasType>(
            linear_md, "bias", metadata);
    std::cout << linear_md["kernel"]["shape"] << std::endl;
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
     using Gemm = typename GemmConfigSm80<KernelType, BiasType>::Gemm;

    // Within {kernel, input_size} , the input_size is the stride

    typename Gemm::Arguments args{
        {total_tokens, output_size, input_size},  // Gemm dimensions M,N,K
        {input, input_size},                       // A matrix (RowMajor)
        {kernel, input_size},                      // B matrix (ColumnMajor)
        // bias broadcast across rows
        {bias, 0},                                 // C matrix (RowMajor)
        {output, output_size},                     // D matrix (RowMajor)
        {1, 1}                                     // alpha=1, beta=1
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

template class Linear<int8_t, int32_t>;
