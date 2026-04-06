#include "layers/Encoder.h"
#include "utils/utils.cuh"


Encoder::Encoder(const nlohmann::json encoder_metadata,
    WeightsMetadata* wmd)
{
    auto embedding_wmd = encoder_metadata["embedding"]["embedding"];
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

    auto encoder_fwd_lstm_wmd = encoder_metadata["forward_lstm"];

    auto fwd_kernel_tag = dtype_to_tag(
        encoder_fwd_lstm_wmd["hf"]["kernel"]["dtype"]
    );
    auto fwd_bias_tag = dtype_to_tag(
        encoder_fwd_lstm_wmd["hf"]["bias"]["dtype"]
    );

    hidden_dim =
        encoder_fwd_lstm_wmd["if"]["kernel"]["shape"][0].get<int>();

    if (fwd_kernel_tag == DTypeTag::Int8 && fwd_bias_tag == DTypeTag::Int32)
    {
        using KernelType = int8_t;
        using BiasType   = int32_t;

        forward_lstmcell = new LSTMCell<KernelType, BiasType>(
            encoder_fwd_lstm_wmd, *wmd
        );
    }
    
    if(encoder_metadata.contains("backward_lstm"))
    {
        auto encoder_bkwd_lstm_wmd = encoder_metadata["backward_lstm"];

        auto bkwd_kernel_tag = dtype_to_tag(
            encoder_bkwd_lstm_wmd["hf"]["kernel"]["dtype"]
        );
        auto bkwd_bias_tag = dtype_to_tag(
            encoder_bkwd_lstm_wmd["hf"]["bias"]["dtype"]
        );

        if (bkwd_kernel_tag == DTypeTag::Int8 &&
            bkwd_bias_tag == DTypeTag::Int32)
        {
            using KernelType = int8_t;
            using BiasType   = int32_t;

            backward_lstmcell = new LSTMCell<KernelType, BiasType>(
                encoder_bkwd_lstm_wmd, *wmd
            );
        }
    }
}


Encoder::~Encoder()
{
    delete embedding;
    delete forward_lstmcell;
    if (backward_lstmcell) delete backward_lstmcell;
}


__global__ void concat_outputs_kernel(
    float*       out,    // [seq_len, B, 2*H]
    const float* fwd,    // [seq_len, B, H]
    const float* bkwd,   // [seq_len, B, H]
    int total,           // seq_len * B * H
    int H)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // idx encodes a (seq*B, h) pair flat over [seq_len*B, H]
    int row = idx / H;   // which (t, b) pair
    int h   = idx % H;

    out[row * 2 * H + h]     = fwd[idx];   // first half
    out[row * 2 * H + H + h] = bkwd[idx];  // second half
}



void Encoder::forward(int* d_input_indices, float* encoder_outputs,
    int batch_size, int seq_len, float scale_x,
    cudaStream_t stream)
{
    const int total_tokens = batch_size * seq_len;

    // Allocate embedding output  [total_tokens, embedding_dim]
    int mul_factor = (embedding_dtype == "float32")  ? sizeof(float)
                   : (embedding_dtype == "float16")  ? sizeof(__half)
                   : sizeof(__nv_bfloat16);

    const int total_embedding_size = total_tokens * embedding_dim;

    void* embedding_output = nullptr;
    cudaMallocAsync(&embedding_output, total_embedding_size * mul_factor, stream);

    // Quantized int8 buffer  [total_tokens, embedding_dim]
    int8_t* quantized_embedding_int8;
    cudaMallocAsync(
        &quantized_embedding_int8,
        total_embedding_size * sizeof(int8_t),
        stream
    );

    // -------------------------
    // Run embedding forward for all tokens at once
    // -------------------------
    embedding->forward(
        d_input_indices,
        batch_size,
        seq_len,
        embedding_output,
        stream
    );

    // -------------------------
    // Quantize entire embedding output → INT8
    // -------------------------
    {
        int block = 256;
        int grid  = (total_embedding_size + block - 1) / block;
        float inv_scale = 1.0f / scale_x;
        if (embedding_dtype == "bfloat16") {
            quantize_to_int8<<<grid, block, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(embedding_output),
                quantized_embedding_int8,
                total_embedding_size,
                inv_scale
            );
        } else if (embedding_dtype == "float16") {
            quantize_to_int8<<<grid, block, 0, stream>>>(
                static_cast<const __half*>(embedding_output),
                quantized_embedding_int8,
                total_embedding_size,
                inv_scale
            );
        } else if (embedding_dtype == "float32") {
            quantize_to_int8<<<grid, block, 0, stream>>>(
                static_cast<const float*>(embedding_output),
                quantized_embedding_int8,
                total_embedding_size,
                inv_scale
            );
        } else {
            throw std::runtime_error("Unsupported embedding dtype: " + embedding_dtype);
        }
    }

    // Double-buffer h and c: fwd_* is the current state,
    // new_* receives the next state; pointers are swapped after each step.
    float* fwd_hidden;
    float* fwd_cell;
    float* fwd_new_hidden;
    float* fwd_new_cell;

    cudaMallocAsync(&fwd_hidden, batch_size * hidden_dim * sizeof(float), stream);
    cudaMallocAsync(&fwd_cell,   batch_size * hidden_dim * sizeof(float), stream);
    cudaMallocAsync(&fwd_new_hidden, batch_size * hidden_dim * sizeof(float), stream);
    cudaMallocAsync(&fwd_new_cell,   batch_size * hidden_dim * sizeof(float), stream);

    // Zero-initialise h_0 and c_0
    cudaMemsetAsync(fwd_hidden, 0, batch_size * hidden_dim * sizeof(float), stream);
    cudaMemsetAsync(fwd_cell,   0, batch_size * hidden_dim * sizeof(float), stream);

    // outputs[t] holds fwd_hidden after step t  →  shape [seq_len, batch, hidden]
    float* fwd_lstm_outputs;
    cudaMallocAsync(
        &fwd_lstm_outputs,
        seq_len * batch_size * hidden_dim * sizeof(float),
        stream
    );

    // -------------------------
    // Iterate over sequence  (mirrors the Python loop above)
    // -------------------------
    const int step_embedding_elems = batch_size * embedding_dim;

    for (int t = 0; t < seq_len; ++t) {
        // x_t: quantized embeddings for this timestep [batch, embedding_dim]
        const int8_t* x_t =
            quantized_embedding_int8 + t * step_embedding_elems;

        forward_lstmcell->forward(
            fwd_cell,
            fwd_hidden,
            x_t,
            fwd_new_cell,
            fwd_new_hidden,
            batch_size,
            scale_x,
            stream
        );

        // outputs.append(fwd_hidden)  →  copy new_hidden into outputs[t]
        cudaMemcpyAsync(
            fwd_lstm_outputs + t * batch_size * hidden_dim,
            fwd_new_hidden,
            batch_size * hidden_dim * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream
        );

        // Advance state: (fwd_hidden, fwd_cell) ← (new_hidden, new_cell)
        std::swap(fwd_hidden, fwd_new_hidden);
        std::swap(fwd_cell,   fwd_new_cell);
    }

    cudaFreeAsync(fwd_hidden,     stream);
    cudaFreeAsync(fwd_cell,       stream);
    cudaFreeAsync(fwd_new_hidden, stream);
    cudaFreeAsync(fwd_new_cell,   stream);
    cudaFreeAsync(embedding_output, stream);
    cudaFreeAsync(quantized_embedding_int8, stream);

    if(backward_lstmcell)
    {
        // Double-buffer h and c: fwd_* is the current state,
        // new_* receives the next state; pointers are swapped after each step.
        float* bkwd_hidden;
        float* bkwd_cell;
        float* bkwd_new_hidden;
        float* bkwd_new_cell;

        cudaMallocAsync(&bkwd_hidden, batch_size * hidden_dim * sizeof(float),
        stream);
        cudaMallocAsync(&bkwd_cell, batch_size * hidden_dim * sizeof(float),
        stream);
        cudaMallocAsync(&bkwd_new_hidden, batch_size * hidden_dim * sizeof(float),
        stream);
        cudaMallocAsync(&bkwd_new_cell, batch_size * hidden_dim * sizeof(float),
        stream);

        // Zero-initialise h_0 and c_0
        cudaMemsetAsync(bkwd_hidden, 0, batch_size * hidden_dim * sizeof(float),
        stream);
        cudaMemsetAsync(bkwd_cell, 0, batch_size * hidden_dim * sizeof(float),
        stream);

        // outputs[t] holds fwd_hidden after step t  →
        // shape [seq_len, batch, hidden]
        float* bkwd_lstm_outputs;
        cudaMallocAsync(
            &bkwd_lstm_outputs,
            seq_len * batch_size * hidden_dim * sizeof(float),
            stream
        );

        for (int t = seq_len - 1; t >= 0; t--) {
            // x_t: quantized embeddings for this timestep
            // [batch, embedding_dim]
            const int8_t* x_t =
                quantized_embedding_int8 + t * step_embedding_elems;

            backward_lstmcell->forward(
                bkwd_cell,
                bkwd_hidden,
                x_t,
                bkwd_new_cell,
                bkwd_new_hidden,
                batch_size,
                scale_x,
                stream
            );

            // outputs.append(fwd_hidden)  →  copy new_hidden into outputs[t]
            cudaMemcpyAsync(
                bkwd_lstm_outputs + t * batch_size * hidden_dim,
                bkwd_new_hidden,
                batch_size * hidden_dim * sizeof(float),
                cudaMemcpyDeviceToDevice,
                stream
            );

            // Advance state: (fwd_hidden, fwd_cell) ← (new_hidden, new_cell)
            std::swap(bkwd_hidden, bkwd_new_hidden);
            std::swap(bkwd_cell,   bkwd_new_cell);
        }

        cudaFreeAsync(bkwd_hidden,     stream);
        cudaFreeAsync(bkwd_cell,       stream);
        cudaFreeAsync(bkwd_new_hidden, stream);
        cudaFreeAsync(bkwd_new_cell,   stream);

        // concatenation step
        int total = seq_len * batch_size * hidden_dim;
        int block = 256;
        int grid  = (total + block - 1) / block;
        concat_outputs_kernel<<<grid, block, 0, stream>>>(
            encoder_outputs, fwd_lstm_outputs, bkwd_lstm_outputs,
            total, hidden_dim
        );
        cudaFreeAsync(fwd_lstm_outputs,  stream);
        cudaFreeAsync(bkwd_lstm_outputs, stream);

    }
    else
    {
        cudaMemcpyAsync(
            encoder_outputs,
            fwd_lstm_outputs,
            seq_len * batch_size * hidden_dim * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream
        );
        cudaFreeAsync(fwd_lstm_outputs, stream);
    }
}