#pragma once
// Minimal stubs that let the FA2 csrc headers compile without a full libtorch
// installation. Only the inference (no dropout) path is used; dropout-related
// fields in Flash_fwd_params are present but are never read by the non-dropout
// kernel specialisation.

#include <cstdint>

// at::PhiloxCudaState is embedded in Flash_fwd_params. It holds two uint64_t
// values that seed the Philox RNG for dropout. We provide a POD replacement.
namespace at {
struct PhiloxCudaState {
    uint64_t seed_   = 0;
    uint64_t offset_ = 0;
};
}  // namespace at

// c10 CUDA error-check macros — not needed for inference; define them as no-ops.
#ifndef C10_CUDA_KERNEL_LAUNCH_CHECK
#define C10_CUDA_KERNEL_LAUNCH_CHECK() do {} while (0)
#endif
#ifndef C10_CUDA_CHECK
#define C10_CUDA_CHECK(expr) (expr)
#endif
