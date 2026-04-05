#include "embedding.h"
#include "lstmcell.h"

class Encoder
{
    private:
        Embedding* embedding;
        LSTMCell* forward_lstmcell;
        LSTMCell* backward_lstmcell = nullptr;
        int hidden_dim;

    public:
        Encoder(const nlohmann::json encoder_metadata,
            const WeightsMetadata& metadata);

        ~Encoder();

        // void forward()


}