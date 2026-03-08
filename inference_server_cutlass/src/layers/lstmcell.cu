#include "layers/layer.cuh"
#include "layers/lstmcell.h"


// LSTMCell methods
template <typename KernelType, typename BiasType>
LSTMCell<KernelType, BiasType>::LSTMCell(
    const nlohmann::json lstm_metadata, const WeightsMetadata& metadata)
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
    KernelType* if_kernel = 
        extract_details_and_load_parameters<KernelType>(
            if_md, "kernel", metadata);

    // IG
    auto ig_md = lstm_metadata["ig"];
    KernelType* ig_kernel = 
        extract_details_and_load_parameters<KernelType>(
            ig_md, "kernel", metadata);

    // II
    auto ii_md = lstm_metadata["ii"];
    KernelType* ii_kernel = 
        extract_details_and_load_parameters<KernelType>(
            ii_md, "kernel", metadata);

    // IO
    auto io_md = lstm_metadata["io"];
    KernelType* io_kernel = 
        extract_details_and_load_parameters<KernelType>(
            io_md, "kernel", metadata);

}

template <typename KernelType, typename BiasType>
LSTMCell<KernelType, BiasType>::~LSTMCell()
{
    cudaFree(hf_bias);
    cudaFree(hf_kernel);

    cudaFree(hg_bias);
    cudaFree(hg_kernel);

    cudaFree(hi_bias);
    cudaFree(hi_kernel);

    cudaFree(ho_bias);
    cudaFree(ho_kernel);

    cudaFree(if_kernel);
    cudaFree(ig_kernel);
    cudaFree(ii_kernel);
    cudaFree(io_kernel);
}

template class LSTMCell<int8_t, int32_t>;
