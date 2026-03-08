#include "layer.h"


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
                    int8_t* quantized_embedding_int8);

        float get_embedding_scale();
};