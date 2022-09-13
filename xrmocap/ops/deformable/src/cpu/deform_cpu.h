/*!
# --------------------------------------------------------------------------------------------------------------------
# Modified from Deformable-ConvNets-V2 https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0 and
# Deformable DETR https://github.com/fundamentalvision/Deformable-DETR
# --------------------------------------------------------------------------------------------------------------------
*/

#pragma once
#include <torch/extension.h>

at::Tensor
deform_cpu_forward(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step);

std::vector<at::Tensor>
deform_cpu_backward(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int im2col_step);
