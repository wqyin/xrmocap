/*!
# --------------------------------------------------------------------------------------------------------------------
# Modified from Deformable-ConvNets-V2 https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0 and
# Deformable DETR https://github.com/fundamentalvision/Deformable-DETR
# --------------------------------------------------------------------------------------------------------------------
*/

#pragma once

#include "cpu/deform_cpu.h"

#ifdef WITH_CUDA
#include "cuda/deform_cuda.h"
#endif


at::Tensor
deform_forward(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step)
{
    if (value.type().is_cuda())
    {
#ifdef WITH_CUDA
        return deform_cuda_forward(
            value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

std::vector<at::Tensor>
deform_backward(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int im2col_step)
{
    if (value.type().is_cuda())
    {
#ifdef WITH_CUDA
        return deform_cuda_backward(
            value, spatial_shapes, level_start_index, sampling_loc, attn_weight, grad_output, im2col_step);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}
