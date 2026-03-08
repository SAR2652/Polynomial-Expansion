#include "layer.h"


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
        KernelType* ig_kernel;
        KernelType* ii_kernel;
        KernelType* io_kernel;

    public: 
        LSTMCell(const nlohmann::json lstm_metadata,
            const WeightsMetadata& metadata);

        ~LSTMCell();

        // void forward(KernelType* input, BiasType* output);
};
