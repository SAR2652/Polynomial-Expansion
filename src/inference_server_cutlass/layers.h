#include <string>
#include <cuda_fp16.h>
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
        T* extract_details_and_load_parameters(nlohmann::json& param_md,
            const std::string& param_type, WeightsMetadata& wmd) const;
};


class Embedding: public Layer {
    private:
        __half* embedding = nullptr;
        std::vector<int> shape;
        std::string dtype;
        float scale;

    public:
        Embedding(const std::vector<int> shape, const std::string dtype,
                  const float scale, const int offset, const int size,
                  WeightsMetadata& metadata);
        ~Embedding();
        void forward(int* input_indices, int batch_size,
                     int sequence_length, __half* output);
        float get_embedding_scale();

};


template<typename KernelType, typename BiasType>
class LSTMCell: public Layer
{
    private:
        KernelType* hf_kernel;
        BiasType* hf_bias;

        KernelType* hg_kernel;
        BiasType* hg_bias;
        
        KernelType* hi_kernel;
        BiasType* hi_bias;
        
        KernelType* ho_kernel;
        BiasType* ho_bias;

        KernelType* if_kernel;
        BiasType* if_bias;

        KernelType* ig_kernel;
        BiasType* ig_bias;

        KernelType* ii_kernel;
        BiasType* ii_bias;

        KernelType* io_kernel;
        BiasType* io_bias;  

    public:
        template<typename KernelType, typename BiasType> 
            LSTMCell(nlohmann::json lstm_metadata, WeightsMetadata& metadata);

        ~LSTMCell();

        template<typename KernelType, typename BiasType>
            void forward(KernelType* input, BiasType* output);
};