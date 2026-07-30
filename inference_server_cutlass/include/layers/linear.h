#pragma once
#include "layer.h"
#include "layers/gemm_config.h"


template<typename KernelType, typename BiasType>
class Linear: public Layer
{
    private:
        KernelType* kernel;
        BiasType* bias;
        typename GemmConfigSm80<KernelType, BiasType>::Gemm gemm_op;

    public:
        Linear(const nlohmann::json linear_metadata,
               const WeightsMetadata& metadata);

        ~Linear();

        void forward(KernelType* input, BiasType* output,
            const int total_tokens, const int input_shape,
            const int output_shape, cudaStream_t stream);

};