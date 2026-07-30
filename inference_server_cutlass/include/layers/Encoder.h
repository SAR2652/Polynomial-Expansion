#pragma once
#include "embedding.h"
#include "lstmcell.h"


template <typename KernelType, typename BiasType>
class Encoder
{
    private:
        Embedding* embedding;
        std::string embedding_dtype;
        int embedding_dim;

        LSTMCell<KernelType, BiasType>* forward_lstmcell;
        LSTMCell<KernelType, BiasType>* backward_lstmcell = nullptr;
        int hidden_dim;

    public:
        int output_hidden_dim() const {
            return backward_lstmcell ? 2 * hidden_dim : hidden_dim;
        }
        
        
        Encoder(const nlohmann::json encoder_metadata,
            WeightsMetadata* wmd);

        ~Encoder();

        void forward(int* d_input_indices, float* encoder_outputs,
            float* decoder_initial_hidden, float* decoder_initial_cell,
            int batch_size, int seq_len, float scale_x,
            cudaStream_t stream);

};
