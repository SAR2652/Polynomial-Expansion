#pragma once
#include "layers/embedding.h"
#include "layers/multiheadattention.h"
#include "layers/lstmcell.h"
#include "layers/linear.h"


template <typename EmbeddingType, typename KernelType, typename BiasType>
class Decoder
{
    private:
        int num_heads;
        bool use_cache;

        std::string embedding_dtype;
        int embedding_dim;
        int hidden_dim;
        int vocab_size;

        // Calibrated scale for transiently quantizing the LSTM hidden state
        // (float32 -> int8) before it is fed into fc_out.
        float h_quant_scale;
        float fc_out_kernel_scale;

        Embedding* embedding;
        // Self-attention queries/keys/values are the token embedding (EmbeddingType,
        // e.g. bf16). Cross-attention's query is the float32 self-attention output
        // and its keys/values are the float32 encoder outputs, so it is always
        // specialized on float regardless of EmbeddingType.
        MultiHeadAttention<EmbeddingType, KernelType, BiasType>* self_attention;
        MultiHeadAttention<float, KernelType, BiasType>* cross_attention;
        LSTMCell<KernelType, BiasType>* lstmcell;
        Linear<KernelType, BiasType>* fc_out;

    public:
        Decoder(const nlohmann::json decoder_metadata,
            WeightsMetadata* wmd, bool use_cache_, int num_heads_);

        ~Decoder();

        // Runs one decode step (mirrors DecoderSACAFLAX.__call__):
        //   d_target_token   [B]      last generated token id (or <SOS>)
        //   d_decoder_tokens [B, T]   previously generated tokens; only read
        //                             when use_cache is false and decoder_step > 0
        //   encoder_outputs  [B, S, embed_dim]  float32, from Encoder::forward
        //   hidden_state, cell_state             [B, hidden_dim] previous LSTM state
        //   new_hidden_state, new_cell_state      [B, hidden_dim] updated LSTM state (out)
        //   next_token_logits                     [B, vocab_size] (out)
        void forward(
            int*          d_target_token,
            int*          d_decoder_tokens,
            float*        encoder_outputs,
            float*        hidden_state,
            float*        cell_state,
            float*        new_hidden_state,
            float*        new_cell_state,
            float*        next_token_logits,
            int           batch_size,
            int           encoder_seq_len,
            int           decoder_step,
            float         x_quant_scale,
            KVCache*      self_attn_kv_cache,
            KVCache*      cross_attn_kv_cache,
            cudaStream_t  stream);
};