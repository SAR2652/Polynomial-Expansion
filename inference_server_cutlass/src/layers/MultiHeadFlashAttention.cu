// flash_attn_shim.h must be included FIRST: it provides at::PhiloxCudaState and
// stubs out c10 macros before the FA2 headers pull in <ATen/...>.
#include "layers/flash_attn_shim.h"
// FA2 public API: declares Flash_fwd_params and run_mha_fwd_<T, HeadDim, Is_causal>.
// Definitions live in flash_fwd_hdim{N}_{bf16,fp16}[_causal]_sm80.cu; link those
// objects when building.
#include "flash_attn/src/flash.h"

#include "layers/MultiHeadAttention.h"
#include "layers/gemm_config.h"
#include <cuda_bf16.h>
#include <cmath>    // sqrtf, M_LOG2E
#include <cstdio>   // printf (error path only)


// ── Helper CUDA kernels ────────────────────────────────────────────────────────

// Quantise bfloat16 activations → int8 using a calibrated fixed scale.
//   int8_val = clamp( round( bf16_val / scale ), -128, 127 )
//   inv_scale = 1 / scale is precomputed by the caller.
__global__ void quantize_bf16_to_int8_kernel(
    int8_t*              out,
    const __nv_bfloat16* in,
    float                inv_scale,
    int                  total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float v = rintf(__bfloat162float(in[idx]) * inv_scale);
    out[idx] = (int8_t)fmaxf(-128.f, fminf(127.f, v));
}

// Dequantise an INT8 GEMM accumulator (int32) → bfloat16.
//   dequant_scale = kernel_scale * x_quant_scale  (precomputed by the caller)
__global__ void dequant_int32_to_bf16_kernel(
    __nv_bfloat16* out,
    const int32_t* in,
    float          dequant_scale,
    int            total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    out[idx] = __float2bfloat16((float)in[idx] * dequant_scale);
}

// Dequantise an INT8 GEMM accumulator (int32) → float32 (used for the final
// output projection whose result feeds directly into the LSTM).
__global__ void dequant_int32_to_fp32_kernel(
    float*         out,
    const int32_t* in,
    float          dequant_scale,
    int            total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    out[idx] = (float)in[idx] * dequant_scale;
}


// ── Small host helpers ─────────────────────────────────────────────────────────

// Round x up to the next multiple of m.
static inline int round_multiple(int x, int m) { return (x + m - 1) / m * m; }

// Dispatch flash::run_mha_fwd_<bf16, kHeadDim, Is_causal> at runtime given a
// head_dim value.  Template parameters must be compile-time constants; the
// switch statement instantiates all head dims compiled into the FA2 binaries.
template<bool Is_causal>
static void dispatch_flash_fwd_bf16(flash::Flash_fwd_params& params,
                                    cudaStream_t stream,
                                    int head_dim)
{
    switch (head_dim) {
        case  32: flash::run_mha_fwd_<__nv_bfloat16,  32, Is_causal>(params, stream); break;
        case  64: flash::run_mha_fwd_<__nv_bfloat16,  64, Is_causal>(params, stream); break;
        case  96: flash::run_mha_fwd_<__nv_bfloat16,  96, Is_causal>(params, stream); break;
        case 128: flash::run_mha_fwd_<__nv_bfloat16, 128, Is_causal>(params, stream); break;
        case 192: flash::run_mha_fwd_<__nv_bfloat16, 192, Is_causal>(params, stream); break;
        case 256: flash::run_mha_fwd_<__nv_bfloat16, 256, Is_causal>(params, stream); break;
        default:
            printf("MultiHeadAttention: unsupported head_dim=%d "
                   "(FA2 supports 32, 64, 96, 128, 192, 256)\n", head_dim);
    }
}


// ── Constructor ────────────────────────────────────────────────────────────────

template <typename KernelType, typename BiasType>
MultiHeadAttention<KernelType, BiasType>::MultiHeadAttention(
    const nlohmann::json  mha_metadata,
    const WeightsMetadata& metadata,
    int                    num_heads_)
{
    auto q_md = mha_metadata["query_proj"];
    query_proj_kernel_scale = q_md["kernel"]["scale"];
    query_proj = new Linear<KernelType, BiasType>(q_md, metadata);

    auto k_md = mha_metadata["key_proj"];
    key_proj_kernel_scale = k_md["kernel"]["scale"];
    key_proj = new Linear<KernelType, BiasType>(k_md, metadata);

    auto v_md = mha_metadata["value_proj"];
    value_proj_kernel_scale = v_md["kernel"]["scale"];
    value_proj = new Linear<KernelType, BiasType>(v_md, metadata);

    auto o_md = mha_metadata["out_proj"];
    out_proj_kernel_scale = o_md["kernel"]["scale"];
    out_proj = new Linear<KernelType, BiasType>(o_md, metadata);

    // query_proj kernel shape is [input_dim, output_dim] = [embed_dim, embed_dim].
    // Both dims are embed_dim for a square projection; use index 0 (input dim).
    embed_dim = mha_metadata["query_proj"]["kernel"]["shape"][0];
    num_heads = num_heads_;
    head_dim  = embed_dim / num_heads;
}


// ── Destructor ─────────────────────────────────────────────────────────────────

template <typename KernelType, typename BiasType>
MultiHeadAttention<KernelType, BiasType>::~MultiHeadAttention()
{
    delete query_proj;
    delete key_proj;
    delete value_proj;
    delete out_proj;
}


// ── Forward pass ───────────────────────────────────────────────────────────────
//
// Pipeline (mirrors DecoderSACAFLAX.__call__ → MultiHeadAttentionFLAX.__call__):
//
//   bf16 input
//     → quantise bf16 → int8                          (calibration scale_x)
//     → INT8 GEMM Q/K/V projections → int32           (CUTLASS Sm80)
//     → dequantise int32 → bf16                       (kernel_scale * scale_x)
//     → Flash Attention forward                        (FA2, bf16 I/O)
//     → quantise attention-output bf16 → int8          (calibration scale_x)
//     → INT8 GEMM output projection → int32            (CUTLASS Sm80)
//     → dequantise int32 → float32                    (kernel_scale * scale_x)
//
// All intermediate buffers are allocated/freed with cudaMallocAsync/cudaFreeAsync
// on the caller-supplied stream so they stay in the same async memory pool.

template <typename KernelType, typename BiasType>
void MultiHeadAttention<KernelType, BiasType>::forward(
    __nv_bfloat16* query,
    __nv_bfloat16* key,
    __nv_bfloat16* value,
    float*         output,
    int            batch_size,
    int            query_seq_len,
    int            key_seq_len,
    float          x_quant_scale,
    bool           is_causal,
    cudaStream_t   stream)
{
    const int B        = batch_size;
    const int Tq       = query_seq_len;
    const int Tk       = key_seq_len;
    const int E        = embed_dim;        // total head channels
    const int H        = num_heads;
    const int D        = head_dim;         // channels per head  (= E / H)
    const int total_q  = B * Tq;
    const int total_k  = B * Tk;
    const float inv_scale = 1.0f / x_quant_scale;

    // ── 1. Quantise bfloat16 activations → int8 ──────────────────────────────
    // Each of Q / K / V may point to the same tensor in self-attention; we
    // quantise them independently so every downstream INT8 GEMM gets its own
    // non-aliased buffer.
    KernelType *q_int8, *k_int8, *v_int8;
    cudaMallocAsync(&q_int8, sizeof(KernelType) * total_q * E, stream);
    cudaMallocAsync(&k_int8, sizeof(KernelType) * total_k * E, stream);
    cudaMallocAsync(&v_int8, sizeof(KernelType) * total_k * E, stream);

    {
        const int threads = 256;
        quantize_bf16_to_int8_kernel<<<(total_q * E + threads - 1) / threads,
                                       threads, 0, stream>>>(
            q_int8, query, inv_scale, total_q * E);
        quantize_bf16_to_int8_kernel<<<(total_k * E + threads - 1) / threads,
                                       threads, 0, stream>>>(
            k_int8, key,   inv_scale, total_k * E);
        quantize_bf16_to_int8_kernel<<<(total_k * E + threads - 1) / threads,
                                       threads, 0, stream>>>(
            v_int8, value, inv_scale, total_k * E);
    }

    // ── 2. Q / K / V projections  (INT8 × INT8 → int32, bias fused) ──────────
    // Output layout: [total_{q,k}, E]  which we will reinterpret as
    //                [B, T_{q,k}, H, D]  (Flash Attention's expected layout).
    BiasType *q_int32, *k_int32, *v_int32;
    cudaMallocAsync(&q_int32, sizeof(BiasType) * total_q * E, stream);
    cudaMallocAsync(&k_int32, sizeof(BiasType) * total_k * E, stream);
    cudaMallocAsync(&v_int32, sizeof(BiasType) * total_k * E, stream);

    query_proj->forward(q_int8, q_int32, total_q, E, E, stream);
    key_proj  ->forward(k_int8, k_int32, total_k, E, E, stream);
    value_proj->forward(v_int8, v_int32, total_k, E, E, stream);

    cudaFreeAsync(q_int8, stream);
    cudaFreeAsync(k_int8, stream);
    cudaFreeAsync(v_int8, stream);

    // ── 3. Dequantise int32 → bfloat16  ──────────────────────────────────────
    // Effective scale = kernel_scale * x_quant_scale.
    // The bias stored in the weight file was pre-quantised with this combined
    // scale (same convention as LSTMCell), so no separate bias rescaling is
    // needed — the int32 accumulator already accounts for the bias in the same
    // numeric range.
    __nv_bfloat16 *q_bf16, *k_bf16, *v_bf16, *attn_out_bf16;
    cudaMallocAsync(&q_bf16,        sizeof(__nv_bfloat16) * total_q * E, stream);
    cudaMallocAsync(&k_bf16,        sizeof(__nv_bfloat16) * total_k * E, stream);
    cudaMallocAsync(&v_bf16,        sizeof(__nv_bfloat16) * total_k * E, stream);
    cudaMallocAsync(&attn_out_bf16, sizeof(__nv_bfloat16) * total_q * E, stream);

    {
        const int threads = 256;
        dequant_int32_to_bf16_kernel<<<(total_q * E + threads - 1) / threads,
                                       threads, 0, stream>>>(
            q_bf16, q_int32, query_proj_kernel_scale * x_quant_scale, total_q * E);
        dequant_int32_to_bf16_kernel<<<(total_k * E + threads - 1) / threads,
                                       threads, 0, stream>>>(
            k_bf16, k_int32, key_proj_kernel_scale   * x_quant_scale, total_k * E);
        dequant_int32_to_bf16_kernel<<<(total_k * E + threads - 1) / threads,
                                       threads, 0, stream>>>(
            v_bf16, v_int32, value_proj_kernel_scale * x_quant_scale, total_k * E);
    }

    cudaFreeAsync(q_int32, stream);
    cudaFreeAsync(k_int32, stream);
    cudaFreeAsync(v_int32, stream);

    // ── 4. Flash Attention forward  ───────────────────────────────────────────
    // Tensor layout for FA2: (batch, seqlen, num_heads, head_dim)
    //   • last dim (head_dim) contiguous → head_stride = D
    //   • next dim (num_heads)           → row_stride  = H * D  = E
    //   • next dim (seqlen)              → batch_stride = T * H * D = T * E
    // The memory produced by the INT8 GEMMs is already in [B*T, E] order, which
    // maps directly to [B, T, H, D] without any extra transpose.

    const int seqlen_q_rounded = round_multiple(Tq, 128);
    const int seqlen_k_rounded = round_multiple(Tk, 128);
    const int d_rounded        = round_multiple(D,   32);
    const float softmax_scale  = 1.0f / sqrtf((float)D);

    // softmax_lse: log-sum-exp written by FA2, shape [B, H, seqlen_q_rounded].
    float *softmax_lse;
    cudaMallocAsync(&softmax_lse, sizeof(float) * B * H * seqlen_q_rounded, stream);

    // Zero-initialise the entire struct so every field not explicitly set below
    // starts at a defined, benign value (nullptr / 0 / false).
    flash::Flash_fwd_params fa{};

    // ── Dtype / mode flags ────────────────────────────────────────────────────

    // is_bf16: tells the kernel which fp16 variant to use for the Q/K/V/O
    // pointers.  true → __nv_bfloat16 (BF16_SWITCH path in the kernel).
    // false → __half (FP16).  Must match the actual pointer type above.
    fa.is_bf16 = true;

    // is_causal: when true the kernel applies a causal (lower-triangular) mask
    // so each query position can only attend to key positions ≤ its own index.
    // Used for self-attention.  Cross-attention sets this false so every query
    // sees the full encoder sequence.
    fa.is_causal = is_causal;

    // is_seqlens_k_cumulative: governs how cu_seqlens_k is interpreted in the
    // variable-length path.  true → cu_seqlens_k holds exclusive prefix-sums
    // (standard CSR format, e.g. [0, s0, s0+s1, ...]).  false → it holds the
    // individual lengths directly.  We pass nullptr for cu_seqlens_k (uniform
    // batch), so the value doesn't matter functionally; true is the safe default
    // consistent with PyTorch's flash_attn bindings.
    fa.is_seqlens_k_cumulative = true;

    // ── Data pointers ─────────────────────────────────────────────────────────

    // q_ptr / k_ptr / v_ptr: device pointers to the Query, Key, Value tensors
    // in bfloat16.  Expected layout: (batch, seqlen, num_heads, head_dim) with
    // the last dimension contiguous.  The kernel indexes into these using the
    // stride fields below.
    fa.q_ptr = q_bf16;
    fa.k_ptr = k_bf16;
    fa.v_ptr = v_bf16;

    // o_ptr: output tensor, same layout as q_ptr: (B, Tq, H, D) bf16.
    // FA2 writes the weighted sum of Values here.
    fa.o_ptr = attn_out_bf16;

    // softmax_lse_ptr: device buffer for the log-sum-exp of each query row,
    // shape [B, H, seqlen_q_rounded], dtype float32.  FA2 writes this as a
    // by-product of the online softmax computation.  It is not used downstream
    // in inference (only needed for the backward pass), but FA2 always writes
    // it, so we must provide a valid allocation.
    fa.softmax_lse_ptr = softmax_lse;

    // ── Strides (in elements, not bytes) ──────────────────────────────────────
    // FA2 uses three strides per tensor to navigate the (B, T, H, D) layout:
    //
    //   batch_stride : elements to advance to reach the next batch item
    //                  = T * H * D   (one full sequence worth of heads)
    //   row_stride   : elements to advance to reach the next sequence position
    //                  = H * D       (one token's worth of head vectors)
    //   head_stride  : elements to advance to reach the next head at the same
    //                  sequence position = D  (one head vector)
    //
    // Because the INT8 GEMMs produced [B*T, E] row-major output and E == H*D,
    // these three strides describe the layout without any extra transpose.

    fa.q_batch_stride = (int64_t)Tq * H * D;
    fa.k_batch_stride = (int64_t)Tk * H * D;
    fa.v_batch_stride = (int64_t)Tk * H * D;
    fa.o_batch_stride = (int64_t)Tq * H * D;

    fa.q_row_stride = (int64_t)H * D;
    fa.k_row_stride = (int64_t)H * D;
    fa.v_row_stride = (int64_t)H * D;
    fa.o_row_stride = (int64_t)H * D;

    fa.q_head_stride = (int64_t)D;
    fa.k_head_stride = (int64_t)D;
    fa.v_head_stride = (int64_t)D;
    fa.o_head_stride = (int64_t)D;

    // ── Batch and head counts ─────────────────────────────────────────────────

    // b: batch size — the outermost dimension of Q/K/V/O.
    fa.b = B;

    // h: number of query heads.
    fa.h = H;

    // h_k: number of key/value heads.  Equal to h for standard MHA (every
    // query head has its own K/V head).  Would be < h for GQA or MQA.
    fa.h_k = H;

    // h_h_k_ratio: precomputed h / h_k.  The kernel uses this to map a query
    // head index to its corresponding K/V head in GQA/MQA.  Always 1 for MHA.
    fa.h_h_k_ratio = 1;

    // ── Sequence lengths ──────────────────────────────────────────────────────

    // seqlen_q / seqlen_k: actual (unpadded) number of tokens in queries and
    // keys/values.  For cross-attention Tq == 1 (current decoder token) and
    // Tk == encoder sequence length.  For self-attention both equal Tq.
    fa.seqlen_q = Tq;
    fa.seqlen_k = Tk;

    // seqlen_q_rounded / seqlen_k_rounded: sequence lengths rounded up to the
    // next multiple of 128, which is the tile size (kBlockM / kBlockN) used by
    // the FA2 kernel.  The kernel internally pads or bounds-checks using the
    // actual seqlen_{q,k}; the rounded values determine shared-memory and LSE
    // buffer sizing.
    fa.seqlen_q_rounded = seqlen_q_rounded;
    fa.seqlen_k_rounded = seqlen_k_rounded;

    // ── Head dimension ────────────────────────────────────────────────────────

    // d: actual head dimension (= embed_dim / num_heads).  For this model: 128
    // channels / num_heads.  Must be one of {32, 64, 96, 128, 192, 256} for the
    // SM80 FA2 kernel binaries compiled in external/flash-attention/csrc/.
    fa.d = D;

    // d_rounded: head dimension rounded up to the next multiple of 32.  The
    // kernel uses this to size its register tiles; with the supported head dims
    // it equals d (all are already multiples of 32).
    fa.d_rounded = d_rounded;

    // ── Softmax scaling ───────────────────────────────────────────────────────

    // scale_softmax: the scalar s applied before softmax: softmax(s · QKᵀ).
    // Standard choice is 1/√D to keep the dot-product variance independent of
    // head dimension, preventing gradient saturation in deep softmax.
    fa.scale_softmax = softmax_scale;

    // scale_softmax_log2: scale_softmax × log₂(e) ≈ scale_softmax × 1.4427.
    // FA2 implements softmax via exp2 rather than exp for speed (hardware has a
    // native exp2 instruction).  Converting the scale to log₂ space lets the
    // kernel compute exp2(scale_log2 · x) == exp(scale · x) exactly.
    fa.scale_softmax_log2 = softmax_scale * (float)M_LOG2E;

    // softcap: if > 0 the kernel applies tanh(x / softcap) before softmax to
    // cap logit magnitude (used in some Gemini / Gemma variants).  0 disables
    // it.  When non-zero, scale_softmax and scale_softmax_log2 are repurposed
    // to hold softcap and softcap × log₂(e); see set_params_fprop in flash_api.
    fa.softcap = 0.0f;

    // ── Dropout (disabled for inference) ─────────────────────────────────────

    // p_dropout: probability of KEEPING an attention weight (not dropping it).
    // 1.0 means keep everything — no dropout.  During training values < 1 would
    // randomly zero out attention weights and require the Philox RNG state.
    fa.p_dropout = 1.0f;

    // rp_dropout: reciprocal of p_dropout (= 1 / p_dropout).  Surviving weights
    // are multiplied by this to maintain the expected value of the output.  With
    // p_dropout == 1.0 this is also 1.0 — no rescaling.
    fa.rp_dropout = 1.0f;

    // scale_softmax_rp_dropout: the two scales fused into one multiply inside
    // the kernel: scale_softmax × rp_dropout.  Avoids a separate multiply per
    // element.  With no dropout this equals scale_softmax.
    fa.scale_softmax_rp_dropout = softmax_scale;

    // p_dropout_in_uint8_t: uint8 representation of p_dropout used for the fast
    // Bernoulli comparison inside the kernel.  A random uint8 is drawn; if it
    // is ≤ this threshold the weight survives, otherwise it is zeroed.
    // floor(1.0 × 255) == 255: the threshold covers the entire [0, 255] range,
    // so no weight is ever dropped.
    fa.p_dropout_in_uint8_t = 255u;

    // ── Optional features (all disabled) ──────────────────────────────────────

    // window_size_left / window_size_right: token radius of a sliding-window
    // (local) attention mask.  -1 means unbounded — attend to the full sequence
    // on that side.  Both -1 → standard global attention (no local window).
    fa.window_size_left  = -1;
    fa.window_size_right = -1;

    // cu_seqlens_q / cu_seqlens_k: cumulative sequence-length arrays for
    // variable-length (packed / unpadded) batches.  nullptr selects the standard
    // padded-batch path where every item has the same seqlen_{q,k}.
    fa.cu_seqlens_q = nullptr;
    fa.cu_seqlens_k = nullptr;

    // seqused_k: per-batch-item actual K length override (paged-KV / dynamic
    // length).  nullptr → use seqlen_k uniformly for all batch items.
    fa.seqused_k = nullptr;

    // alibi_slopes_ptr: per-head ALiBi position bias slopes.  nullptr disables
    // ALiBi (Attention with Linear Biases), which this model does not use.
    fa.alibi_slopes_ptr = nullptr;

    // num_splits: controls the split-KV parallel reduction path.  0 lets FA2
    // decide automatically; for single-token decoding (Tq == 1) FA2 may choose
    // split-KV to increase parallelism.  0 is safe for all sequence lengths.
    fa.num_splits = 0;

    // oaccum_ptr: accumulator buffer used by the split-KV combine kernel.
    // nullptr tells FA2 to write the final result directly to o_ptr without an
    // intermediate accumulation step.
    fa.oaccum_ptr = nullptr;

    // p_ptr: if non-null FA2 stores the full (B, H, Tq, Tk) attention-weight
    // matrix P = softmax(QKᵀ) here.  Only needed when return_softmax == true
    // (e.g. for debugging or certain training losses).  nullptr for inference.
    fa.p_ptr = nullptr;

    if (is_causal)
        dispatch_flash_fwd_bf16<true> (fa, stream, D);
    else
        dispatch_flash_fwd_bf16<false>(fa, stream, D);

    cudaFreeAsync(q_bf16,      stream);
    cudaFreeAsync(k_bf16,      stream);
    cudaFreeAsync(v_bf16,      stream);
    cudaFreeAsync(softmax_lse, stream);

    // ── 5. Quantise attention output → int8  (for the output projection) ─────
    // attn_out_bf16 is [B, Tq, H, D] = [total_q, E] — already flat and contiguous.
    KernelType *attn_out_int8;
    cudaMallocAsync(&attn_out_int8, sizeof(KernelType) * total_q * E, stream);
    {
        const int threads = 256;
        quantize_bf16_to_int8_kernel<<<(total_q * E + threads - 1) / threads,
                                       threads, 0, stream>>>(
            attn_out_int8, attn_out_bf16, inv_scale, total_q * E);
    }
    cudaFreeAsync(attn_out_bf16, stream);

    // ── 6. Output projection  (INT8 × INT8 → int32, bias fused) ─────────────
    BiasType *out_int32;
    cudaMallocAsync(&out_int32, sizeof(BiasType) * total_q * E, stream);
    out_proj->forward(attn_out_int8, out_int32, total_q, E, E, stream);
    cudaFreeAsync(attn_out_int8, stream);

    // ── 7. Dequantise int32 → float32  (final output consumed by LSTM) ───────
    {
        const int threads = 256;
        dequant_int32_to_fp32_kernel<<<(total_q * E + threads - 1) / threads,
                                       threads, 0, stream>>>(
            output, out_int32, out_proj_kernel_scale * x_quant_scale, total_q * E);
    }
    cudaFreeAsync(out_int32, stream);
}


template class MultiHeadAttention<int8_t, int32_t>;
