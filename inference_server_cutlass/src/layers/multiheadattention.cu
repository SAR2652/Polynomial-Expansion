#include "layers/multiheadattention.h"
#include "layers/gemm_config.h"
#include "utils/utils.cuh"
#include <cuda_bf16.h>
#include <cmath>
#include <cfloat>
#include <cstdio>


// ── Constructor ────────────────────────────────────────────────────────────────

template <typename EmbeddingType, typename KernelType, typename BiasType>
MultiHeadAttention<EmbeddingType, KernelType, BiasType>::MultiHeadAttention(
    const nlohmann::json   mha_metadata,
    const WeightsMetadata& metadata,
    int                    num_heads_,
    bool                   use_cache_,
    Mode                   mode_)
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

    embed_dim = mha_metadata["query_proj"]["kernel"]["shape"][0];
    num_heads = num_heads_;
    head_dim  = embed_dim / num_heads;
    use_cache = use_cache_;
    mode      = mode_;
}


// ── Destructor ─────────────────────────────────────────────────────────────────

template <typename EmbeddingType, typename KernelType, typename BiasType>
MultiHeadAttention<EmbeddingType, KernelType, BiasType>::~MultiHeadAttention()
{
    delete query_proj;
    delete key_proj;
    delete value_proj;
    delete out_proj;
}


// ── Device kernels ─────────────────────────────────────────────────────────────

__global__ void dequant_int32_to_bf16_kernel(
    __nv_bfloat16* out, const int32_t* in, float scale, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    out[idx] = __float2bfloat16((float)in[idx] * scale);
}

__global__ void dequant_int32_to_fp32_kernel(
    float* out, const int32_t* in, float scale, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    out[idx] = (float)in[idx] * scale;
}


// [B, T, H, D] -> [B, H, T, D]: makes each head's [T, D] block contiguous
// for CUTLASS GemmBatched (batch stride = T*D, constant across all B*H batches).
__global__ void transpose_BTHD_to_BHTD(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16*       __restrict__ dst,
    int B, int T, int H, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T * H * D) return;
    int d = idx % D;
    int h = (idx / D) % H;
    int t = (idx / (D * H)) % T;
    int b = idx / (D * H * T);
    dst[b * H * T * D + h * T * D + t * D + d] =
        src[b * T * H * D + t * H * D + h * D + d];
}

// [B, H, T, D] -> [B, T, H, D]: reverse of the above, used for the output.
__global__ void transpose_BHTD_to_BTHD(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16*       __restrict__ dst,
    int B, int T, int H, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T * H * D) return;
    int d = idx % D;
    int t = (idx / D) % T;
    int h = (idx / (D * T)) % H;
    int b = idx / (D * T * H);
    dst[b * T * H * D + t * H * D + h * D + d] =
        src[b * H * T * D + h * T * D + t * D + d];
}

// In-place row-wise softmax on [rows, cols] bf16 matrix.
// Upcasts to float for exp/sum, writes bf16 back.
// One block per row; blockDim.x must be a power of 2, >= cols or loops.
__global__ void softmax_rows_bf16(__nv_bfloat16* x, int rows, int cols)
{
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows) return;
    __nv_bfloat16* p = x + (size_t)row * cols;

    float mx = -FLT_MAX;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        mx = fmaxf(mx, __bfloat162float(p[i]));
    smem[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    mx = smem[0];
    __syncthreads();

    float sum = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = expf(__bfloat162float(p[i]) - mx);
        p[i] = __float2bfloat16(v);
        sum += v;
    }
    smem[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    sum = smem[0];
    __syncthreads();

    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        p[i] = __float2bfloat16(__bfloat162float(p[i]) / sum);
}


// ── Helpers ────────────────────────────────────────────────────────────────────

static int next_pow2(int n) { int p = 1; while (p < n) p <<= 1; return p; }

template <typename GemmOp>
static cutlass::Status run_batched_gemm(
    GemmOp& op, typename GemmOp::Arguments args, cudaStream_t stream)
{
    size_t ws = op.get_workspace_size(args);
    void* wsbuf = nullptr;
    if (ws > 0) cudaMallocAsync(&wsbuf, ws, stream);

    cutlass::Status st = op.initialize(args, wsbuf, stream);
    if (st != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "GemmBatched init failed: %d\n", (int)st);
        if (wsbuf) cudaFreeAsync(wsbuf, stream);
        return st;
    }
    st = op(stream);
    if (st != cutlass::Status::kSuccess)
        std::fprintf(stderr, "GemmBatched run failed: %d\n", (int)st);

    if (wsbuf) cudaFreeAsync(wsbuf, stream);
    return st;
}


// ── Forward ────────────────────────────────────────────────────────────────────

template <typename EmbeddingType, typename KernelType, typename BiasType>
void MultiHeadAttention<EmbeddingType, KernelType, BiasType>::forward(
    EmbeddingType* query,
    EmbeddingType* key,
    EmbeddingType* value,
    float*         output,
    int            batch_size,
    int            query_seq_len,
    int            key_seq_len,
    float          x_quant_scale,
    int            decoder_step,
    KVCache*       kv_cache,
    cudaStream_t   stream)
{
    const int B       = batch_size;
    const int Tq      = query_seq_len;
    const int Tk      = key_seq_len;
    const int E       = embed_dim;
    const int H       = num_heads;
    const int D       = head_dim;
    const int total_q = B * Tq;
    const int total_k = B * Tk;
    const float inv_scale = 1.0f / x_quant_scale;
    int blocks;

    // ── 1. Quantise Q -> int8 ─────────────────────────────────────────────────
    KernelType* q_int8;
    cudaMallocAsync(&q_int8, sizeof(KernelType) * total_q * E, stream);
    {
        const int threads = 256;
        blocks = (total_q * E + threads - 1) / threads;
        quantize_to_int8<<<blocks, threads, 0, stream>>>(
            query, q_int8, total_q * E, inv_scale);
    }

    // ── 2. Q projection (INT8 x INT8 -> int32) ────────────────────────────────
    BiasType* q_int32;
    cudaMallocAsync(&q_int32, sizeof(BiasType) * total_q * E, stream);
    query_proj->forward(q_int8, q_int32, total_q, E, E, stream);
    cudaFreeAsync(q_int8, stream);

    // ── 3. Dequantise Q -> bfloat16 ───────────────────────────────────────────
    __nv_bfloat16* q_bf16;
    cudaMallocAsync(&q_bf16, sizeof(__nv_bfloat16) * total_q * E, stream);
    {
        const int threads = 256;
        dequant_int32_to_bf16_kernel<<<blocks,threads, 0, stream>>>(
            q_bf16, q_int32, query_proj_kernel_scale * x_quant_scale,
            total_q * E);
    }
    cudaFreeAsync(q_int32, stream);

    // ── 4. Transpose Q: [B, Tq, H, D] -> [B, H, Tq, D] ──────────────────────
    __nv_bfloat16* q_t;
    cudaMallocAsync(&q_t, sizeof(__nv_bfloat16) * B * H * Tq * D, stream);
    {
        const int threads = 256;
        int total = B * Tq * H * D;
        blocks = (total + threads - 1) / threads;
        transpose_BTHD_to_BHTD<<<blocks, threads, 0, stream>>>(
            q_bf16, q_t, B, Tq, H, D);
    }
    cudaFreeAsync(q_bf16, stream);

    // ── 5. Conditional K/V projection + KV cache logic ────────────────────────
    // First IF (mirrors FLAX): compute K/V when mode is SELF/NONE, or CROSS
    // and the cache is still empty (first decode step for cross-attention).
    bool need_kv_proj =
        (mode == Mode::SELF || mode == Mode::NONE) ||
        (mode == Mode::CROSS && kv_cache != nullptr &&
         kv_cache->key == nullptr && kv_cache->value == nullptr);

    int  Tk_eff = Tk;               // effective key length for the GEMMs
    __nv_bfloat16* k_for_attn = nullptr;
    __nv_bfloat16* v_for_attn = nullptr;
    bool free_kv_after_sv = false;  // true when we own k/v_for_attn

    if (need_kv_proj) {
        // Quantise K, V.
        KernelType *k_int8, *v_int8;
        cudaMallocAsync(&k_int8, sizeof(KernelType) * total_k * E, stream);
        cudaMallocAsync(&v_int8, sizeof(KernelType) * total_k * E, stream);
        {
            const int threads = 256;
            blocks = (total_k * E + threads - 1) / threads;
            quantize_to_int8<<<blocks, threads, 0, stream>>>(key,   k_int8, total_k * E, inv_scale);
            quantize_to_int8<<<blocks, threads, 0, stream>>>(value, v_int8, total_k * E, inv_scale);
        }

        // K, V projections (INT8 x INT8 -> int32).
        BiasType *k_int32, *v_int32;
        cudaMallocAsync(&k_int32, sizeof(BiasType) * total_k * E, stream);
        cudaMallocAsync(&v_int32, sizeof(BiasType) * total_k * E, stream);
        key_proj  ->forward(k_int8, k_int32, total_k, E, E, stream);
        value_proj->forward(v_int8, v_int32, total_k, E, E, stream);
        cudaFreeAsync(k_int8, stream);
        cudaFreeAsync(v_int8, stream);

        // Dequantise K, V -> bfloat16.
        __nv_bfloat16 *k_bf16, *v_bf16;
        cudaMallocAsync(&k_bf16, sizeof(__nv_bfloat16) * total_k * E, stream);
        cudaMallocAsync(&v_bf16, sizeof(__nv_bfloat16) * total_k * E, stream);
        {
            const int threads = 256;
            blocks = (total_k * E + threads - 1) / threads;
            dequant_int32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(
                k_bf16, k_int32, key_proj_kernel_scale   * x_quant_scale, total_k * E);
            dequant_int32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(
                v_bf16, v_int32, value_proj_kernel_scale * x_quant_scale, total_k * E);
        }
        cudaFreeAsync(k_int32, stream);
        cudaFreeAsync(v_int32, stream);

        // Transpose K, V: [B, Tk, H, D] -> [B, H, Tk, D].
        // Cache and GEMMs both use [B, H, T, D] layout.
        __nv_bfloat16 *k_bf16_transpose, *v_bf16_transpose;
        cudaMallocAsync(&k_bf16_transpose, sizeof(__nv_bfloat16) * B * H * Tk * D, stream);
        cudaMallocAsync(&v_bf16_transpose, sizeof(__nv_bfloat16) * B * H * Tk * D, stream);
        {
            const int threads = 256;
            int total = B * Tk * H * D;
            blocks = (total + threads - 1) / threads;
            transpose_BTHD_to_BHTD<<<blocks, threads, 0, stream>>>(
                k_bf16, k_bf16_transpose, B, Tk, H, D);
            transpose_BTHD_to_BHTD<<<blocks, threads, 0, stream>>>(
                v_bf16, v_bf16_transpose, B, Tk, H, D);
        }
        cudaFreeAsync(k_bf16, stream);
        cudaFreeAsync(v_bf16, stream);

        if (mode == Mode::CROSS && kv_cache != nullptr) {
            // Transfer ownership to cache; do NOT free k_t/v_t.
            kv_cache->key   = k_bf16_transpose;
            kv_cache->value = v_bf16_transpose;
            k_for_attn = kv_cache->key;
            v_for_attn = kv_cache->value;
        } else if (mode == Mode::SELF && kv_cache != nullptr) {
            // Write the new token's K/V into slot decoder_step of the cache.
            // Cache layout [B, H, max_seq_len, D] has uniform stride max_seq_len*D
            // per (b,h), so a single Memcpy2D covers all B*H slices.
            size_t row_bytes = (size_t)D * sizeof(__nv_bfloat16);
            cudaMemcpy2DAsync(
                kv_cache->key   + (size_t)decoder_step * D,
                (size_t)kv_cache->max_seq_len * D * sizeof(__nv_bfloat16),
                k_bf16_transpose, row_bytes, row_bytes, B * H,
                cudaMemcpyDeviceToDevice, stream);
            cudaMemcpy2DAsync(
                kv_cache->value + (size_t)decoder_step * D,
                (size_t)kv_cache->max_seq_len * D * sizeof(__nv_bfloat16),
                v_bf16_transpose, row_bytes, row_bytes, B * H,
                cudaMemcpyDeviceToDevice, stream);
            cudaFreeAsync(k_bf16_transpose, stream);
            cudaFreeAsync(v_bf16_transpose, stream);
            k_for_attn = kv_cache->key;
            v_for_attn = kv_cache->value;
            Tk_eff = decoder_step + 1;  // only attend to valid (written) positions
        } else {
            // No cache — use the freshly transposed buffers directly.
            k_for_attn = k_bf16_transpose;
            v_for_attn = v_bf16_transpose;
            free_kv_after_sv = true;
        }
    } else {
        // CROSS + cache already populated: skip projection, read from cache.
        k_for_attn = kv_cache->key;
        v_for_attn = kv_cache->value;
    }

    // ── 6. QK^T batched GEMM ──────────────────────────────────────────────────
    // [B*H, Tq, D] x [B*H, Tk_eff, D]^T -> [B*H, Tq, Tk_eff]
    // k_for_attn is [Tk_eff, D] RowMajor in memory; declaring B as ColumnMajor
    // makes CUTLASS read it as [D, Tk_eff] = K^T (leading dim = D in both views).
    __nv_bfloat16* scores;
    cudaMallocAsync(&scores, sizeof(__nv_bfloat16) * B * H * Tq * Tk_eff, stream);
    {
        using QKGemm = AttnGemmQK::Gemm;
        // cutlass::bfloat16_t is bit-compatible with __nv_bfloat16 but is a distinct
        // C++ type with no implicit conversion between them; TensorRef needs the
        // former, our buffers are the latter, so bridge with reinterpret_cast.
        auto cb16 = [](__nv_bfloat16* p) { return reinterpret_cast<cutlass::bfloat16_t*>(p); };
        QKGemm qk_op;
        typename QKGemm::Arguments args{
            {Tq, Tk_eff, D},
            {cb16(q_t),        D},     (int64_t)Tq     * D,      // A: Q  [Tq,D]        RowMajor,    lda=D, batch_stride_A
            {cb16(k_for_attn), D},     (int64_t)Tk_eff * D,      // B: K  [Tk_eff,D]    ColumnMajor, ldb=D (= K^T), batch_stride_B
            {cb16(scores),     Tk_eff},(int64_t)Tq     * Tk_eff, // C (beta=0, unused), batch_stride_C
            {cb16(scores),     Tk_eff},(int64_t)Tq     * Tk_eff, // D: scores [Tq,Tk_eff] RowMajor, batch_stride_D
            {1.0f / sqrtf((float)D), 0.0f},
            B * H
        };
        run_batched_gemm(qk_op, args, stream);
    }
    cudaFreeAsync(q_t, stream);

    // ── 7. Softmax over key dimension ─────────────────────────────────────────
    {
        int rows    = B * H * Tq;
        int cols    = Tk_eff;
        int threads = min(next_pow2(cols), 1024);
        softmax_rows_bf16<<<rows, threads, threads * sizeof(float), stream>>>(
            scores, rows, cols);
    }

    // ── 8. SV batched GEMM ────────────────────────────────────────────────────
    // [B*H, Tq, Tk_eff] x [B*H, Tk_eff, D] -> [B*H, Tq, D]
    __nv_bfloat16* attn_out_t;
    cudaMallocAsync(&attn_out_t, sizeof(__nv_bfloat16) * B * H * Tq * D, stream);
    {
        using SVGemm = AttnGemmSV::Gemm;
        auto cb16 = [](__nv_bfloat16* p) { return reinterpret_cast<cutlass::bfloat16_t*>(p); };
        SVGemm sv_op;
        typename SVGemm::Arguments args{
            {Tq, D, Tk_eff},
            {cb16(scores),     Tk_eff}, (int64_t)Tq     * Tk_eff, // A: scores [Tq,Tk_eff] RowMajor, lda=Tk_eff, batch_stride_A
            {cb16(v_for_attn), D},      (int64_t)Tk_eff * D,      // B: V      [Tk_eff,D]  RowMajor, ldb=D, batch_stride_B
            {cb16(attn_out_t), D},      (int64_t)Tq     * D,      // C, batch_stride_C
            {cb16(attn_out_t), D},      (int64_t)Tq     * D,      // D: output [Tq,D]      RowMajor, batch_stride_D
            {1.0f, 0.0f},
            B * H
        };
        run_batched_gemm(sv_op, args, stream);
    }
    cudaFreeAsync(scores, stream);
    if (free_kv_after_sv) {
        cudaFreeAsync(k_for_attn, stream);
        cudaFreeAsync(v_for_attn, stream);
    }

    // ── 9. Transpose output: [B, H, Tq, D] -> [B, Tq, H, D] = [B*Tq, E] ──────
    __nv_bfloat16* attn_out_bf16;
    cudaMallocAsync(&attn_out_bf16, sizeof(__nv_bfloat16) * total_q * E, stream);
    {
        const int threads = 256;
        int total = B * Tq * H * D;
        blocks = (total + threads - 1) / threads;
        transpose_BHTD_to_BTHD<<<blocks, threads, 0, stream>>>(
            attn_out_t, attn_out_bf16, B, Tq, H, D);
    }
    cudaFreeAsync(attn_out_t, stream);

    // ── 10. Quantise attention output -> int8 ─────────────────────────────────
    KernelType* attn_out_int8;
    cudaMallocAsync(&attn_out_int8, sizeof(KernelType) * total_q * E, stream);
    {
        const int threads = 256;
        blocks = (total_q * E + threads - 1) / threads;
        quantize_to_int8<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            attn_out_bf16, attn_out_int8, total_q * E, inv_scale);
    }
    cudaFreeAsync(attn_out_bf16, stream);

    // ── 11. Output projection (INT8 x INT8 -> int32) ──────────────────────────
    BiasType* out_int32;
    cudaMallocAsync(&out_int32, sizeof(BiasType) * total_q * E, stream);
    out_proj->forward(attn_out_int8, out_int32, total_q, E, E, stream);
    cudaFreeAsync(attn_out_int8, stream);

    // ── 12. Dequantise -> float32 ─────────────────────────────────────────────
    {
        const int threads = 256;
        blocks = (total_q * E + threads - 1) / threads;
        dequant_int32_to_fp32_kernel<<<blocks, threads, 0, stream>>>(
            output, out_int32, out_proj_kernel_scale * x_quant_scale, total_q * E);
    }
    cudaFreeAsync(out_int32, stream);
}


template class MultiHeadAttention<__nv_bfloat16, int8_t, int32_t>;
template class MultiHeadAttention<float, int8_t, int32_t>;
