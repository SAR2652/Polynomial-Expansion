#include "layers/Decoder.h"
#include "utils/utils.cuh"


template <typename EmbeddingType, typename KernelType, typename BiasType>
Decoder<EmbeddingType, KernelType, BiasType>::Decoder(
    const nlohmann::json decoder_metadata,
    WeightsMetadata* wmd, bool use_cache_, int num_heads_)
{
    use_cache = use_cache_;
    num_heads = num_heads_;

    auto embedding_wmd = decoder_metadata["embedding"]["embedding"];
    const std::vector<int> embedding_shape =
        embedding_wmd["shape"].get<std::vector<int>>();
    embedding_dtype = embedding_wmd["dtype"];
    const int embedding_offset = embedding_wmd["offset"];
    const int embedding_size   = embedding_wmd["size"];
    embedding_dim = embedding_shape[1];

    // const float scale_x = wmd->metadata["calibration"]["scale_x"];

    embedding = new Embedding(
        embedding_shape, embedding_dtype,
        embedding_wmd["scale"], embedding_offset,
        embedding_size, *wmd
    );

    auto self_attention_wmd = decoder_metadata["self_attention"];
    auto cross_attention_wmd = decoder_metadata["cross_attention"];

    // Self- and cross-attention share the same kernel/bias dtypes.
    auto embedding_tag = dtype_to_tag(embedding_dtype);
    auto multihead_attention_kernel_tag = dtype_to_tag(
        self_attention_wmd["key_proj"]["kernel"]["dtype"]
    );
    auto multihead_attention_bias_tag = dtype_to_tag(
        self_attention_wmd["key_proj"]["bias"]["dtype"]
    );

    if (embedding_tag == DTypeTag::BFloat16 &&
        multihead_attention_kernel_tag == DTypeTag::Int8 &&
        multihead_attention_bias_tag == DTypeTag::Int32)
    {
        // Mirrors DecoderSACAFLAX.setup: with use_cache, self-attention runs
        // incrementally (SELF) and cross-attention caches encoder K/V (CROSS);
        // without it, both attend over the full context every step (NONE).
        using SelfAttention  = MultiHeadAttention<EmbeddingType, KernelType, BiasType>;
        using CrossAttention = MultiHeadAttention<float, KernelType, BiasType>;

        auto self_attn_mode  = use_cache ? SelfAttention::Mode::SELF
                                          : SelfAttention::Mode::NONE;
        auto cross_attn_mode = use_cache ? CrossAttention::Mode::CROSS
                                          : CrossAttention::Mode::NONE;

        self_attention = new SelfAttention(
            self_attention_wmd, *wmd, num_heads, use_cache, self_attn_mode
        );

        cross_attention = new CrossAttention(
            cross_attention_wmd, *wmd, num_heads, use_cache, cross_attn_mode
        );
    }

    auto decoder_lstm_wmd = decoder_metadata["lstm"];

    auto decoder_lstm_kernel_tag = dtype_to_tag(
        decoder_lstm_wmd["hf"]["kernel"]["dtype"]
    );
    auto decoder_lstm_bias_tag = dtype_to_tag(
        decoder_lstm_wmd["hf"]["bias"]["dtype"]
    );

    // "if" kernel projects the concatenated [embedded, cross_attn_output]
    // input, but its hidden_dim (shape[0]) matches the LSTM's hidden state.
    hidden_dim = decoder_lstm_wmd["if"]["kernel"]["shape"][0].get<int>();

    if (decoder_lstm_kernel_tag == DTypeTag::Int8 &&
        decoder_lstm_bias_tag == DTypeTag::Int32)
    {
        lstmcell = new LSTMCell<KernelType, BiasType>(
            decoder_lstm_wmd, *wmd
        );
    }

    // Final projection to vocabulary logits (fc_out in DecoderSACAFLAX).
    auto fc_out_wmd = decoder_metadata["fc_out"];
    vocab_size = fc_out_wmd["kernel"]["shape"][0].get<int>();
    fc_out_kernel_scale = fc_out_wmd["kernel"]["scale"];
    fc_out = new Linear<KernelType, BiasType>(fc_out_wmd, *wmd);

    // Calibrated scale for transiently quantizing the LSTM hidden state before
    // it is fed into fc_out (mirrors LSTMCell's own h_quant_scale usage).
    h_quant_scale = wmd->metadata["calibration"]["h_scale"];
}


template <typename EmbeddingType, typename KernelType, typename BiasType>
Decoder<EmbeddingType, KernelType, BiasType>::~Decoder()
{
    delete embedding;
    delete self_attention;
    delete cross_attention;
    delete lstmcell;
    delete fc_out;
}


// Concatenates the current token's embedding with the cross-attention output
// into the LSTM's input row: [embedded | cross_attn_output]. The embedding is
// upcast to float32 to match jnp.concatenate's dtype promotion in
// DecoderSACAFLAX.__call__ step 4 (embedded is EmbeddingType, cross_attn_output
// is already float32).
template <typename T>
__global__ void concat_embedded_cross_kernel(
    float* out, const T* embedded, const float* cross_attn, int B, int E)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * E;
    if (idx >= total) return;
    int b = idx / E;
    int e = idx % E;
    out[b * 2 * E + e]     = to_float<T>(embedded[idx]);
    out[b * 2 * E + E + e] = cross_attn[idx];
}

__global__ void dequant_fc_out_int32_to_fp32(
    float* out, const int32_t* in, float scale, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    out[idx] = (float)in[idx] * scale;
}


// Runs one decode step, mirroring DecoderSACAFLAX.__call__:
//   1. embed target_token
//   2. self-attention (full context if !use_cache, incremental cache otherwise)
//   3. cross-attention (queries = self-attn output, K/V = encoder_outputs)
//   4. concat(embedded, cross_attn_output) -> quantize -> LSTM step
//   5. fc_out(new_hidden_state) -> next_token_logits
template <typename EmbeddingType, typename KernelType, typename BiasType>
void Decoder<EmbeddingType, KernelType, BiasType>::forward(
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
    cudaStream_t  stream)
{
    const int B = batch_size;
    const int E = embedding_dim;
    const int threads = 256;
    int blocks;

    // 1. Embed the target token -> [B, E]
    EmbeddingType* embedded;
    cudaMallocAsync(&embedded, sizeof(EmbeddingType) * B * E, stream);
    embedding->forward(d_target_token, B, 1, embedded, stream);

    // 2. Self-attention.
    float* self_attn_output;
    cudaMallocAsync(&self_attn_output, sizeof(float) * B * E, stream);

    if (decoder_step > 0 && !use_cache) {
        // No cache: attend over embeddings of every token generated so far.
        EmbeddingType* context_embeds;
        cudaMallocAsync(&context_embeds,
            sizeof(EmbeddingType) * B * decoder_step * E, stream);
        embedding->forward(d_decoder_tokens, B, decoder_step, context_embeds, stream);

        self_attention->forward(
            embedded, context_embeds, context_embeds, self_attn_output,
            B, /*query_seq_len=*/1, /*key_seq_len=*/decoder_step,
            x_quant_scale, decoder_step, self_attn_kv_cache, stream
        );

        cudaFreeAsync(context_embeds, stream);
    } else {
        // Cached (incremental) self-attention, or the very first decode step.
        self_attention->forward(
            embedded, embedded, embedded, self_attn_output,
            B, /*query_seq_len=*/1, /*key_seq_len=*/1,
            x_quant_scale, decoder_step, self_attn_kv_cache, stream
        );
    }

    // 3. Cross-attention: queries from self-attention output, K/V from encoder.
    float* cross_attn_output;
    cudaMallocAsync(&cross_attn_output, sizeof(float) * B * E, stream);
    cross_attention->forward(
        self_attn_output, encoder_outputs, encoder_outputs, cross_attn_output,
        B, /*query_seq_len=*/1, /*key_seq_len=*/encoder_seq_len,
        x_quant_scale, /*decoder_step=*/0, cross_attn_kv_cache, stream
    );
    cudaFreeAsync(self_attn_output, stream);

    // 4. Concatenate [embedded, cross_attn_output] -> quantize -> LSTM input.
    float* lstm_input_f32;
    cudaMallocAsync(&lstm_input_f32, sizeof(float) * B * 2 * E, stream);
    {
        blocks = (B * E + threads - 1) / threads;
        concat_embedded_cross_kernel<EmbeddingType><<<blocks, threads, 0, stream>>>(
            lstm_input_f32, embedded, cross_attn_output, B, E);
    }
    cudaFreeAsync(embedded, stream);
    cudaFreeAsync(cross_attn_output, stream);

    KernelType* lstm_input_int8;
    cudaMallocAsync(&lstm_input_int8, sizeof(KernelType) * B * 2 * E, stream);
    {
        blocks = (B * 2 * E + threads - 1) / threads;
        quantize_to_int8<float><<<blocks, threads, 0, stream>>>(
            lstm_input_f32, lstm_input_int8, B * 2 * E, 1.0f / x_quant_scale);
    }
    cudaFreeAsync(lstm_input_f32, stream);

    // 5. LSTM step.
    lstmcell->forward(
        cell_state, hidden_state, lstm_input_int8,
        new_cell_state, new_hidden_state, B, x_quant_scale, stream
    );
    cudaFreeAsync(lstm_input_int8, stream);

    // 6. Predict next token logits from the updated hidden state.
    KernelType* hidden_int8;
    cudaMallocAsync(&hidden_int8, sizeof(KernelType) * B * hidden_dim, stream);
    {
        blocks = (B * hidden_dim + threads - 1) / threads;
        quantize_to_int8<float><<<blocks, threads, 0, stream>>>(
            new_hidden_state, hidden_int8, B * hidden_dim, 1.0f / h_quant_scale);
    }

    BiasType* logits_int32;
    cudaMallocAsync(&logits_int32, sizeof(BiasType) * B * vocab_size, stream);
    fc_out->forward(hidden_int8, logits_int32, B, hidden_dim, vocab_size, stream);
    cudaFreeAsync(hidden_int8, stream);

    {
        blocks = (B * vocab_size + threads - 1) / threads;
        dequant_fc_out_int32_to_fp32<<<blocks, threads, 0, stream>>>(
            next_token_logits, logits_int32,
            fc_out_kernel_scale * h_quant_scale, B * vocab_size);
    }
    cudaFreeAsync(logits_int32, stream);
}


template class Decoder<__nv_bfloat16, int8_t, int32_t>;