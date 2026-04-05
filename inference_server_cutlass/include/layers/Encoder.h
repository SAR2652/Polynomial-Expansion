#include "embedding.h"
#include "lstmcell.h"

class Encoder
{
    private:
        Embedding* embedding;
        LSTMCell* forward_lstmcell;
        LSTMCell* backward_lstmcell = nullptr;
        int hidden_dim;
        bool bkwd_lstm = false;

    public:
        bool check_for_bkwd_lstm();
        
        Encoder(const nlohmann::json encoder_metadata,
            const WeightsMetadata& metadata);

        ~Encoder();

        // void forward()


}