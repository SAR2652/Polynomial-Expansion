#include "layer.h"


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