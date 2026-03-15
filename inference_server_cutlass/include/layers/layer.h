#pragma once
#include "../utils/utils.h"
#include <cuda_runtime.h>


// Each Layer stores an internal CUDA stream (stream_) used for all
// lifetime-related GPU operations such as cudaMallocAsync, cudaFreeAsync,
// and asynchronous weight loading. This ensures that memory allocation and
// deallocation occur on the same stream, preserving correct ordering and
// avoiding cross-stream hazards.
//
// The forward() method of each derived layer receives a separate stream
// argument from the caller. That external stream controls *execution order*
// of the layer's compute kernels (GEMMs, pointwise ops, memcpys, etc.).
// This separation is intentional:
//
//   - stream_ (internal) → object lifetime: allocation, initialization,
//     weight loading, and destruction.
//   - stream  (external) → execution: forward-pass kernels scheduled by
//     the caller.
//
// These two streams do not conflict because they serve different roles and
// live in different scopes. The caller may choose any execution stream per
// forward() call, while the layer consistently uses stream_ for its own
// internal resource management.


class Layer {
    protected:
        cudaStream_t stream_;

        template <typename T>
        T* load_to_device(const std::vector<char>& buffer) const;

        template <typename T>
        T* extract_details_and_load_parameters(nlohmann::json param_md,
            const std::string& param_type, const WeightsMetadata& wmd) const;

    public:
        // Internal lifetime stream: used for cudaMallocAsync, cudaFreeAsync,
        // and asynchronous weight loading. This stream is created once per
        // layer object and destroyed when the layer is destroyed.
        Layer() {
            cudaStreamCreate(&stream_);
        }

        virtual ~Layer() {
            cudaStreamDestroy(stream_);
        }
};
