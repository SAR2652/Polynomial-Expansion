#pragma once
#include "layer.h"
#include "layers/linear.h"
#include <cuda_bf16.h>


struct KVCache {
    __nv_bfloat16* key   = nullptr;  // [B, H, max_seq_len, D]
    __nv_bfloat16* value = nullptr;  // [B, H, max_seq_len, D]
    int            max_seq_len = 0;
};


template<typename EmbeddingType, typename KernelType, typename BiasType>
class MultiHeadAttention : public Layer
{
    public:
        enum class Mode {
            NONE,
            SELF,
            CROSS
        };

    private:
        Linear<KernelType, BiasType> *query_proj, *key_proj, *value_proj, *out_proj;

        // Per-projection kernel quantization scales (loaded from metadata.json).
        // Effective dequantization scale for a projection = kernel_scale * x_quant_scale.
        float query_proj_kernel_scale;
        float key_proj_kernel_scale;
        float value_proj_kernel_scale;
        float out_proj_kernel_scale;

        // Dimensions derived from metadata / constructor arguments.
        int embed_dim;  // = query_proj kernel output dim
        int num_heads;
        int head_dim;   // = embed_dim / num_heads
        bool use_cache;
        Mode mode;

    public:
        MultiHeadAttention(const nlohmann::json mha_metadata,
                           const WeightsMetadata& metadata,
                           int num_heads,
                           bool use_cache = false,
                           Mode mode = Mode::NONE);

        ~MultiHeadAttention();

        // Forward pass:
        //   query  [batch_size, query_seq_len, embed_dim]  EmbeddingType
        //   key    [batch_size, key_seq_len,   embed_dim]  EmbeddingType
        //   value  [batch_size, key_seq_len,   embed_dim]  EmbeddingType
        //   output [batch_size, query_seq_len, embed_dim]  float32
        //
        //   x_quant_scale : used to quantise activations -> int8 before every INT8 GEMM.
        //   decoder_step  : current decode position (0-indexed); used by self-attention
        //                   to write the new K/V slot into kv_cache.
        //   kv_cache      : nullptr -> no caching.
        //                   Self-attention: pre-allocated [B,H,max_seq_len,D] buffers.
        //                   Cross-attention: populated on first call, reused thereafter.
        void forward(
            EmbeddingType* query,
            EmbeddingType* key,
            EmbeddingType* value,
            float*         output,
            int            batch_size,
            int            query_seq_len,
            int            key_seq_len,
            float          x_quant_scale,
            int            decoder_step  = 0,
            KVCache*       kv_cache      = nullptr,
            cudaStream_t   stream        = 0
        );
};
