#pragma once
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/arch/arch.h>
#include <cutlass/epilogue/thread/linear_combination_clamp.h>
#include <cutlass/epilogue/thread/linear_combination.h>

using LayoutRM = cutlass::layout::RowMajor;
using LayoutCM = cutlass::layout::ColumnMajor;

// Epilogue vector width of 1 (instead of CUTLASS's default 128-bit / 4-element
// width) so the output store never touches more columns than the GEMM's N —
// with the default width-4 store, an N not divisible by 4 (e.g. fc_out's
// vocab_size=31) reads/writes past the end of the output/bias buffer.
// ThreadblockShape/WarpShape/InstructionShape/Stages/AlignmentA/AlignmentB
// below are CUTLASS's own Sm80+int8 defaults (unchanged) — only the epilogue
// width is overridden, which requires re-stating everything before it in the
// template parameter list.
template <typename KernelType, typename BiasType>
struct GemmConfigSm80 {
    // EpilogueOutputOp<ElementOutput, Count, ElementAccumulator, ElementCompute>:
    // how the accumulator (D = alpha*A@B + beta*C) is computed and written out.
    // Count=1 means one output element per epilogue op (scalar store) instead
    // of CUTLASS's default 4-element (128-bit) vectorized store — see the
    // out-of-bounds explanation below.
    using EpilogueOp = cutlass::epilogue::thread::LinearCombinationClamp<
        BiasType, 1, BiasType, float>;

    using Gemm = cutlass::gemm::device::Gemm<
        KernelType, LayoutRM,   // ElementA, LayoutA: operand A = the activations (int8), row-major [M, K]
        KernelType, LayoutCM,   // ElementB, LayoutB: operand B = the weight kernel (int8); declared column-major
                                 //   so CUTLASS reads the [N, K]-shaped weight matrix as its transpose [K, N]
                                 //   without an explicit transpose step
        BiasType,  LayoutRM,    // ElementC, LayoutC: operand C = the bias (broadcast via stride 0) / output D, row-major [M, N]
        BiasType,               // ElementAccumulator: accumulator type for the MMA (int32) — matches BiasType,
                                 //   the natural int8x int8 accumulation width, no extra rounding needed
        cutlass::arch::OpClassTensorOp,   // OperatorClass: run on Tensor Cores (vs. OpClassSimt = plain CUDA cores)
        cutlass::arch::Sm80,              // ArchTag: target Tensor Core generation/instruction set this kernel is
                                           //   written against (Ampere mma.sync shapes/pipelining)
        cutlass::gemm::GemmShape<128, 256, 64>,  // ThreadblockShape (M,N,K): tile each threadblock computes per
                                                  //   main-loop iteration. Left at CUTLASS's own Sm80+int8 default —
                                                  //   confirmed (via compute-sanitizer) to fit this GPU's shared-memory
                                                  //   budget for int8, so it didn't need shrinking like the bf16
                                                  //   attention configs below did
        cutlass::gemm::GemmShape<64, 64, 64>,    // WarpShape (M,N,K): sub-tile each warp computes; CUTLASS's default,
                                                  //   kept for the same reason
        cutlass::gemm::GemmShape<16, 8, 32>,     // InstructionShape (M,N,K): shape of a single mma.sync instruction
                                                  //   for int8 on Sm80 — fixed by the hardware ISA, not a tunable choice
        EpilogueOp                                // EpilogueOutputOp: overridden to Count=1 (see above) — CUTLASS's
                                                    //   default Count=4 store would read/write 4 columns per op, and
                                                    //   fc_out's N=vocab_size=31 isn't a multiple of 4, so the last
                                                    //   partial group of columns spilled 1 element past the end of the
                                                    //   124-byte bias/output buffer (a real out-of-bounds access,
                                                    //   confirmed under compute-sanitizer before this fix)
    >;
};

// QK^T batched attention GEMM: bf16 RowMajor x bf16 ColumnMajor -> bf16, float accumulator.
// K is held in [Tk, D] RowMajor memory; declaring it ColumnMajor makes CUTLASS read it as
// [D, Tk], i.e. K^T, without an explicit transpose buffer.
//
// Problem shape is {M=Tq, N=Tk_eff, K=D}: the reduction dimension K is D
// (head_dim, a fixed 32 here — already a safe multiple of the default
// alignment), but the output's N dimension is Tk_eff, the self-attention
// KV-cache length, which is as small as 1 on the first decode step. That's
// why only the epilogue width (not AlignmentA/AlignmentB) is strictly
// required to shrink below — AlignmentA/AlignmentB=1 here is a conservative,
// harmless-cost choice kept consistent with AttnGemmSV, where K really is
// Tk_eff and shrinking those two is required, not just conservative.
//
// CUTLASS's default Sm80+bf16 tile (128x256x64, 3 stages) needs ~144KB of shared
// memory per block, which exceeds this GPU's ~99KB max opt-in shared memory
// (consumer/laptop parts have far less than datacenter Ampere's ~164KB), so
// cudaFuncSetAttribute fails and the kernel never launches. A much smaller tile
// keeps shared memory usage low.
struct AttnGemmQK {
    // Count=1: same reasoning as GemmConfigSm80's epilogue — Tk_eff (this
    // GEMM's N) is the self-attention KV-cache length, which can be as small
    // as 1 on the first decode step, so any vector width > 1 risks writing
    // past the true output extent.
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        cutlass::bfloat16_t, 1, float, float>;

    using Gemm = cutlass::gemm::device::GemmBatched<
        cutlass::bfloat16_t, LayoutRM,   // ElementA, LayoutA: Q, bf16, row-major [Tq, D]
        cutlass::bfloat16_t, LayoutCM,   // ElementB, LayoutB: K, bf16; declared column-major so CUTLASS reads the
                                          //   [Tk, D]-shaped tensor as its transpose [D, Tk] = K^T, giving Q @ K^T
                                          //   directly with no separate transpose buffer
        cutlass::bfloat16_t, LayoutRM,   // ElementC, LayoutC: the attention score matrix (C is unused, beta=0), row-major [Tq, Tk]
        float,                           // ElementAccumulator: accumulate the dot products in float32 for precision,
                                          //   then round back down to bf16 for the scores
        cutlass::arch::OpClassTensorOp,  // OperatorClass: Tensor Cores
        cutlass::arch::Sm80,             // ArchTag: Ampere Tensor Core instruction/pipelining generation
        cutlass::gemm::GemmShape<64, 64, 32>,   // ThreadblockShape (M,N,K): shrunk from CUTLASS's Sm80+bf16 default of
                                                 //   128x256x64 (which needs ~144KB shared memory/block, more than this
                                                 //   GPU's ~99KB max opt-in — cudaFuncSetAttribute failed outright with
                                                 //   the default). This smaller tile needs only ~16KB, comfortably inside budget
        cutlass::gemm::GemmShape<32, 32, 32>,   // WarpShape (M,N,K): sized so ThreadblockShape divides evenly into
                                                 //   whole warps (2x2 = 4 warps/block) at the smaller tile size
        cutlass::gemm::GemmShape<16, 8, 16>,    // InstructionShape (M,N,K): the mma.sync shape for bf16 on Sm80 —
                                                 //   fixed by hardware, not tunable
        EpilogueOp,                              // EpilogueOutputOp: see Count=1 note above
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,  // ThreadblockSwizzle: maps threadblocks
                                                 //   to (batch, tile) coordinates; identity = no remapping, CUTLASS's
                                                 //   standard choice for GemmBatched, left unchanged
        2,                                       // Stages: number of pipelined main-loop stages (prefetch depth).
                                                 //   Lowered from the default 3 — fewer stages means less shared
                                                 //   memory, and with the already-small tile above, 2 is enough to
                                                 //   keep the pipeline fed
        1,                                       // AlignmentA: elements-per-vectorized-access for operand A (Q).
                                                 //   K=D=32 here is already safely aligned to the default width of 8,
                                                 //   so this isn't strictly required for QK — lowered anyway to stay
                                                 //   consistent with AttnGemmSV below, at negligible cost for a GEMM
                                                 //   this small (Tq=1 always, in autoregressive decoding)
        1                                        // AlignmentB: same reasoning as AlignmentA, for operand B (K)
    >;
};

// SV batched attention GEMM: bf16 RowMajor x bf16 RowMajor -> bf16, float accumulator.
// Problem shape is {M=Tq, N=D, K=Tk_eff}: here the reduction dimension K really
// is Tk_eff, the KV-cache length — as small as 1 on the first decode step, far
// below the default 8-element alignment. Unlike AttnGemmQK, shrinking
// AlignmentA/AlignmentB here is required, not just conservative: a 1-valid-
// element K with an 8-wide vectorized load would read 7 elements past the end
// of the scores/V tensors.
struct AttnGemmSV {
    // Count=1: same reasoning as AttnGemmQK's epilogue — this GEMM's N is D
    // (head_dim=32, actually a safe value here), but kept at 1 for the same
    // low-cost-consistency reason as the alignment choice above.
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        cutlass::bfloat16_t, 1, float, float>;

    using Gemm = cutlass::gemm::device::GemmBatched<
        cutlass::bfloat16_t, LayoutRM,   // ElementA, LayoutA: softmax(scores), bf16, row-major [Tq, Tk_eff]
        cutlass::bfloat16_t, LayoutRM,   // ElementB, LayoutB: V, bf16, row-major [Tk_eff, D] (no transpose needed here)
        cutlass::bfloat16_t, LayoutRM,   // ElementC, LayoutC: attention output (C unused, beta=0), row-major [Tq, D]
        float,                           // ElementAccumulator: float32 accumulation, rounded to bf16 for the output
        cutlass::arch::OpClassTensorOp,  // OperatorClass: Tensor Cores
        cutlass::arch::Sm80,             // ArchTag: Ampere Tensor Core instruction/pipelining generation
        cutlass::gemm::GemmShape<64, 64, 32>,   // ThreadblockShape (M,N,K): same shrink as AttnGemmQK, same reason
                                                 //   (default 128x256x64 exceeds this GPU's shared-memory budget)
        cutlass::gemm::GemmShape<32, 32, 32>,   // WarpShape (M,N,K): matches AttnGemmQK's warp tiling
        cutlass::gemm::GemmShape<16, 8, 16>,    // InstructionShape (M,N,K): the mma.sync shape for bf16 on Sm80
        EpilogueOp,                              // EpilogueOutputOp: see Count=1 note above
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,  // ThreadblockSwizzle: identity mapping,
                                                 //   CUTLASS's standard choice for GemmBatched
        2,                                       // Stages: pipeline depth, lowered from default 3 to match the
                                                 //   smaller tile's reduced shared-memory footprint
        1,                                       // AlignmentA: required here (not just conservative) — K=Tk_eff can
                                                 //   be 1, so any wider vectorized access on operand A (scores) would
                                                 //   read past the valid region
        1                                        // AlignmentB: same requirement as AlignmentA, for operand B (V)
    >;
};
