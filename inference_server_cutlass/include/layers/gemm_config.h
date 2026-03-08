#pragma once
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/arch/arch.h>

template <typename KernelType, typename BiasType>
struct GemmConfigSm80 {
    using LayoutRM = cutlass::layout::RowMajor;
    using LayoutCM = cutlass::layout::ColumnMajor;

    using Gemm = cutlass::gemm::device::Gemm<
        KernelType, LayoutRM,
        KernelType, LayoutCM,
        BiasType,  LayoutRM,
        BiasType,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80
    >;
};
