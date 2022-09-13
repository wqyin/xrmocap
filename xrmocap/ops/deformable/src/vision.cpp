/*!
# --------------------------------------------------------------------------------------------------------------------
# Modified from Deformable-ConvNets-V2 https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0 and
# Deformable DETR https://github.com/fundamentalvision/Deformable-DETR
# --------------------------------------------------------------------------------------------------------------------
*/

#include "deform.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("deform_forward", &deform_forward, "deform_forward");
  m.def("deform_backward", &deform_backward, "deform_backward");
}
