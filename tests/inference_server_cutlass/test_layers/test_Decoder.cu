#include <iostream>
#include <string>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "utils/utils.h"
#include "utils/utils.cuh"
#include "layers/Encoder.h"
#include "layers/Decoder.h"


int main()
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::string common_path = "/home/sarvesh26/ML/seq2seq-cutlass-inference/";
    common_path += "output/";
    const std::string json_path = common_path + "metadata.json";
    const std::string bin_path  = common_path + "weights.bin";

    WeightsMetadata* wmd = new WeightsMetadata(json_path, bin_path);
    const float scale_x = wmd->metadata["calibration"]["scale_x"];

    // -------------------------
    // 1. Run the Encoder to get real encoder_outputs.
    //    Shape: (batch_size, seq_len, lstm_hidden_dim)
    // -------------------------
    const int batch_size = 6;
    const int seq_len    = 4;
    const int total_tokens = batch_size * seq_len;

    // Layout: row 0 = t=0 tokens, row 1 = t=1 tokens, …
    std::vector<int> h_encoder_input_indices = {
        1, 6, 3, 2, 5, 4,   // t=0
        2, 5, 1, 4, 3, 6,   // t=1
        3, 4, 2, 6, 1, 5,   // t=2
        4, 3, 6, 1, 2, 5,   // t=3
    };

    int* d_encoder_input_indices;
    cudaMallocAsync(&d_encoder_input_indices, total_tokens * sizeof(int), stream);
    cudaMemcpyAsync(
        d_encoder_input_indices,
        h_encoder_input_indices.data(),
        total_tokens * sizeof(int),
        cudaMemcpyHostToDevice,
        stream
    );

    auto encoder_wmd = wmd->metadata["encoder"];
    Encoder<int8_t, int32_t>* encoder =
        new Encoder<int8_t, int32_t>(encoder_wmd, wmd);

    // Bidirectional encoder -> 2 * forward_lstm hidden_dim; this is also the
    // decoder's embed_dim/hidden_dim (DecoderSACAFLAX is sized off of it).
    const int lstm_hidden_dim = encoder->output_hidden_dim();

    // encoder_outputs: (batch_size, seq_len, lstm_hidden_dim)
    float* encoder_outputs;
    cudaMallocAsync(
        &encoder_outputs,
        batch_size * seq_len * lstm_hidden_dim * sizeof(float),
        stream
    );

    // encoder_hidden/encoder_cell: (batch_size, lstm_hidden_dim) — the
    // encoder's final states, used below to seed the decoder's initial
    // LSTM state.
    float* encoder_hidden;
    float* encoder_cell;
    cudaMallocAsync(&encoder_hidden, batch_size * lstm_hidden_dim * sizeof(float), stream);
    cudaMallocAsync(&encoder_cell,   batch_size * lstm_hidden_dim * sizeof(float), stream);

    encoder->forward(d_encoder_input_indices, encoder_outputs, encoder_hidden,
        encoder_cell, batch_size, seq_len, scale_x, stream);

    // -------------------------
    // 2. Build the Decoder with the KV cache enabled.
    // -------------------------
    const int num_heads = 4;
    auto decoder_wmd = wmd->metadata["decoder"];
    Decoder<__nv_bfloat16, int8_t, int32_t>* decoder =
        new Decoder<__nv_bfloat16, int8_t, int32_t>(
            decoder_wmd, wmd, /*use_cache_=*/true, num_heads
        );

    const int hidden_dim = lstm_hidden_dim;   // decoder LSTM hidden size == encoder output dim
    const int head_dim   = hidden_dim / num_heads;
    const int vocab_size = decoder_wmd["fc_out"]["kernel"]["shape"][0].get<int>();
    const int target_len = 5;   // number of greedy decode steps to run

    // -------------------------
    // 3. Decoder input token, shape (batch_size, 1) — one token per batch item.
    // -------------------------
    const int sos_token_id = 0;
    std::vector<int> h_target_token(batch_size, sos_token_id);

    int* d_target_token;
    cudaMallocAsync(&d_target_token, batch_size * sizeof(int), stream);
    cudaMemcpyAsync(
        d_target_token,
        h_target_token.data(),
        batch_size * sizeof(int),
        cudaMemcpyHostToDevice,
        stream
    );

    // -------------------------
    // 4. LSTM hidden/cell state, double-buffered across decode steps.
    // -------------------------
    float *hidden_state, *cell_state, *new_hidden_state, *new_cell_state;
    cudaMallocAsync(&hidden_state,     batch_size * hidden_dim * sizeof(float), stream);
    cudaMallocAsync(&cell_state,       batch_size * hidden_dim * sizeof(float), stream);
    cudaMallocAsync(&new_hidden_state, batch_size * hidden_dim * sizeof(float), stream);
    cudaMallocAsync(&new_cell_state,   batch_size * hidden_dim * sizeof(float), stream);

    // Seed the decoder's initial LSTM state from the encoder's final state
    // (mirrors CrossAttentionModelFLAX.__call__: decoder_hidden_state and
    // decoder_cell_state come straight out of the encoder, not zero).
    cudaMemcpyAsync(hidden_state, encoder_hidden, batch_size * hidden_dim * sizeof(float),
        cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(cell_state, encoder_cell, batch_size * hidden_dim * sizeof(float),
        cudaMemcpyDeviceToDevice, stream);

    float* next_token_logits;   // (batch_size, vocab_size)
    cudaMallocAsync(&next_token_logits, batch_size * vocab_size * sizeof(float), stream);

    // -------------------------
    // 5. KV caches — required because use_cache is enabled.
    //    self_attn_kv_cache: pre-allocated for the whole decode horizon and
    //    written into incrementally (one decoder_step at a time).
    //    cross_attn_kv_cache: starts empty; MultiHeadAttention::forward
    //    populates it once (from encoder_outputs) on the first decode step
    //    and reuses it for every step after that.
    // -------------------------
    KVCache self_attn_kv_cache;
    self_attn_kv_cache.max_seq_len = target_len;
    cudaMallocAsync(
        &self_attn_kv_cache.key,
        batch_size * num_heads * target_len * head_dim * sizeof(__nv_bfloat16),
        stream
    );
    cudaMallocAsync(
        &self_attn_kv_cache.value,
        batch_size * num_heads * target_len * head_dim * sizeof(__nv_bfloat16),
        stream
    );
    cudaMemsetAsync(self_attn_kv_cache.key, 0,
        batch_size * num_heads * target_len * head_dim * sizeof(__nv_bfloat16), stream);
    cudaMemsetAsync(self_attn_kv_cache.value, 0,
        batch_size * num_heads * target_len * head_dim * sizeof(__nv_bfloat16), stream);

    KVCache cross_attn_kv_cache;
    cross_attn_kv_cache.key        = nullptr;
    cross_attn_kv_cache.value      = nullptr;
    cross_attn_kv_cache.max_seq_len = seq_len;

    // -------------------------
    // 6. Greedy decode loop.
    // -------------------------
    std::vector<int> h_next_token(batch_size);
    std::vector<float> h_logits(batch_size * vocab_size);

    for (int t = 0; t < target_len; ++t) {
        decoder->forward(
            d_target_token,
            /*d_decoder_tokens=*/nullptr,  // unused: use_cache=true always takes
                                           // the incremental self-attention path
            encoder_outputs,
            hidden_state, cell_state,
            new_hidden_state, new_cell_state,
            next_token_logits,
            batch_size, seq_len, /*decoder_step=*/t,
            scale_x,
            &self_attn_kv_cache, &cross_attn_kv_cache,
            stream
        );

        cudaMemcpyAsync(
            h_logits.data(), next_token_logits,
            batch_size * vocab_size * sizeof(float),
            cudaMemcpyDeviceToHost, stream
        );
        cudaStreamSynchronize(stream);

        std::cout << "Step " << t << " argmax tokens: ";
        for (int b = 0; b < batch_size; ++b) {
            int best_idx = 0;
            float best_val = h_logits[b * vocab_size];
            for (int v = 1; v < vocab_size; ++v) {
                float val = h_logits[b * vocab_size + v];
                if (val > best_val) { best_val = val; best_idx = v; }
            }
            h_next_token[b] = best_idx;
            std::cout << best_idx << " ";
        }
        std::cout << "\n";

        cudaMemcpyAsync(
            d_target_token, h_next_token.data(),
            batch_size * sizeof(int),
            cudaMemcpyHostToDevice, stream
        );

        // Advance state: (hidden_state, cell_state) <- (new_hidden_state, new_cell_state)
        std::swap(hidden_state, new_hidden_state);
        std::swap(cell_state,   new_cell_state);
    }

    // -------------------------
    // Cleanup
    // -------------------------
    delete decoder;
    delete encoder;

    cudaFreeAsync(d_encoder_input_indices, stream);
    cudaFreeAsync(encoder_outputs, stream);
    cudaFreeAsync(encoder_hidden, stream);
    cudaFreeAsync(encoder_cell, stream);
    cudaFreeAsync(d_target_token, stream);
    cudaFreeAsync(hidden_state, stream);
    cudaFreeAsync(cell_state, stream);
    cudaFreeAsync(new_hidden_state, stream);
    cudaFreeAsync(new_cell_state, stream);
    cudaFreeAsync(next_token_logits, stream);
    cudaFreeAsync(self_attn_kv_cache.key, stream);
    cudaFreeAsync(self_attn_kv_cache.value, stream);
    if (cross_attn_kv_cache.key)   cudaFreeAsync(cross_attn_kv_cache.key, stream);
    if (cross_attn_kv_cache.value) cudaFreeAsync(cross_attn_kv_cache.value, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    delete wmd;

    return 0;
}
