#include <string>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "device_launch_parameters.h"
#include "utils.h"

#pragma once


class Layer {
    protected:
        template <typename T>
        T* load_to_device(const std::vector<char>& buffer) const;

        template <typename T>
        T* extract_details_and_load_parameters(nlohmann::json param_md,
            const std::string& param_type, const WeightsMetadata& wmd) const;
};


class Embedding : public Layer
{
    private:
        void* embedding = nullptr;          // type-erased pointer
        std::vector<int> shape;
        std::string dtype;
        float scale;

    public:
        Embedding(const std::vector<int> shape, const std::string dtype,
                const float scale, const int offset, const int size,
                WeightsMetadata& metadata);

        ~Embedding();

        // forward now takes void*
        void forward(int* input_indices, int batch_size,
                    int sequence_length, void* output,
                    int8_t* quantized_embedding_int8,
                    float embedding_scale);

        float get_embedding_scale();
};



template<typename KernelType, typename BiasType>
class Linear: public Layer
{
    private:
        KernelType* kernel;
        BiasType* bias;

    public:
        Linear(const nlohmann::json linear_metadata,
               const WeightsMetadata& metadata);

        ~Linear();

        void forward(KernelType* input, BiasType* output,
            const int total_tokens, const int input_shape,
            const int output_shape);

};


// template<typename KernelType, typename BiasType>
// class LSTMCell: public Layer
// {
//     private:
//         KernelType* hf_kernel;
//         BiasType* hf_bias;

//         KernelType* hg_kernel;
//         BiasType* hg_bias;
        
//         KernelType* hi_kernel;
//         BiasType* hi_bias;
        
//         KernelType* ho_kernel;
//         BiasType* ho_bias;

//         KernelType* if_kernel;
//         BiasType* if_bias;

//         KernelType* ig_kernel;
//         BiasType* ig_bias;

//         KernelType* ii_kernel;
//         BiasType* ii_bias;

//         KernelType* io_kernel;
//         BiasType* io_bias;  

//     public: 
//         LSTMCell(const nlohmann::json lstm_metadata,
//             const WeightsMetadata& metadata);

//         ~LSTMCell();

//         void forward(KernelType* input, BiasType* output);
// };