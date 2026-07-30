#pragma once
#include "layer.h"
#include "linear.h"
#include <cuda_bf16.h>

template<typename KernelType, typename BiasType>
class MultiHeadAttention : public Layer
{
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

    public:
        // num_heads must satisfy:
        //   • embed_dim % num_heads == 0
        //   • head_dim (= embed_dim / num_heads) ∈ {32, 64, 96, 128, 192, 256}
        //     (those are the head dims compiled into the FA2 kernel binaries)
        MultiHeadAttention(const nlohmann::json mha_metadata,
                           const WeightsMetadata& metadata,
                           int num_heads);

        ~MultiHeadAttention();

        // Forward pass:
        //   query  [batch_size, query_seq_len, embed_dim]  bfloat16
        //   key    [batch_size, key_seq_len,   embed_dim]  bfloat16
        //          (pass `query` pointer for self-attention)
        //   value  [batch_size, key_seq_len,   embed_dim]  bfloat16
        //          (pass `key`   pointer for self-attention)
        //   output [batch_size, query_seq_len, embed_dim]  float32
        //
        //   x_quant_scale : calibration["scale_x"] from metadata.json —
        //                   used to quantise bfloat16 activations → int8
        //                   before every INT8 GEMM.
        //   is_causal     : true for self-attention (masks future positions),
        //                   false for cross-attention.
        void forward(
            __nv_bfloat16* query,
            __nv_bfloat16* key,
            __nv_bfloat16* value,
            float*         output,
            int            batch_size,
            int            query_seq_len,
            int            key_seq_len,
            float          x_quant_scale,
            bool           is_causal,
            cudaStream_t   stream
        );
};
