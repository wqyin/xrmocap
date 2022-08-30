# yapf: disable

import Deformable as DF
import math
import torch
import torch.nn.functional as F
import warnings
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.init import constant_, xavier_uniform_

# yapf: enable


class ProjAttn(nn.Module):

    def __init__(self,
                 d_model=256,
                 n_levels=4,
                 n_heads=8,
                 n_points=4,
                 projattn_posembed_mode='use_rayconv'):
        """Projective Attention Module Modified from DeformableConvV2 and
        Deformable DETR https://github.com/chengdazhi/Deformable-
        Convolution-V2-PyTorch/tree/pytorch_1.0.0
        https://github.com/fundamentalvision/Deformable-DETR.

        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling
        points per attention head per feature level

        :param projattn_posembed_mode      the
        positional embedding mode of projective attention
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, '
                             'but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of
        # 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in Deform to "
                          'make the dimension of each attention '
                          'head a power of 2 which is more efficient '
                          'in our CUDA implementation.')

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model,
                                          n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model,
                                           n_heads * n_levels * n_points)
        if projattn_posembed_mode == 'use_rayconv':
            self.rayconv = nn.Linear(d_model + 3, d_model)
        elif projattn_posembed_mode == 'use_2d_coordconv':
            self.rayconv = nn.Linear(d_model + 2, d_model)
        elif projattn_posembed_mode == 'ablation_not_use_rayconv':
            self.rayconv = nn.Linear(d_model, d_model)
        else:
            raise ValueError('invalid projective attention posembed mode')
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

        self.projattn_posembed_mode = projattn_posembed_mode

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) \
            * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0])\
            .view(self.n_heads, 1, 1, 2).\
            repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.rayconv.weight.data)
        constant_(self.rayconv.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self,
                query,
                reference_points,
                src_views,
                camera_ray_embeds,
                input_spatial_shapes,
                input_level_start_index,
                input_padding_mask=None):
        """

        Args:
            query :
                (n_views, Length_{query}, C)
            reference_points :
                (n_views, Length_{query}, n_levels, 2),
                range in [0, 1], top-left (0,0), bottom-right (1, 1),
                including padding area or (n_views, Length_{query}, n_levels,
                4), add additional (w, h) to form reference boxes
            src_views :
                list of (n_views, C, H, W), size n_levels,
                [(n_views, C, H_0, W_0), (n_views, C, H_0, W_0), ...,
                (n_views, C, H_{L-1}, W_{L-1})]
            camera_ray_embeds :
                Embedded camera rayts.
            input_spatial_shapes :
                (n_levels, 2), [(H_0, W_0), (H_1, W_1), ...,
                (H_{L-1}, W_{L-1})]
            input_level_start_index :
                (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1,
                H_0*W_0+H_1*W_1+H_2*W_2, ...,
                H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
            input_padding_mask :
                True for padding elements, False for non-padding elements.
                Defaults to None.


        Returns:
            output : (n_views, Length_{query}, C)
        """

        n_views, Len_q, c = query.shape
        feat_lvls = len(src_views)
        sample_grid = torch.clamp(reference_points * 2.0 - 1.0, -1.1, 1.1)
        ref_point_feat_views_alllvs = []

        if 'use_rayconv' in self.projattn_posembed_mode or \
                'use_2d_coordconv' in self.projattn_posembed_mode:
            for lvl in range(feat_lvls):
                ref_point_feat_views_alllvs.append(
                    F.grid_sample(src_views[lvl],
                                  sample_grid[:, :, lvl:lvl +
                                              1, :]).squeeze(-1).permute(
                                                  0, 2, 1))
            input_flatten = torch.cat([
                torch.cat([src.flatten(2)
                           for src in src_views], dim=-1).permute(0, 2, 1),
                torch.cat([cam.flatten(1, 2) for cam in camera_ray_embeds],
                          dim=1)
            ],
                                      dim=-1)

        elif self.projattn_posembed_mode == 'ablation_not_use_rayconv':
            for lvl in range(feat_lvls):
                ref_point_feat_views_alllvs.append(
                    F.grid_sample(src_views[lvl],
                                  sample_grid[:, :, lvl:lvl +
                                              1, :]).squeeze(-1).permute(
                                                  0, 2, 1))
            input_flatten = torch.cat([src.flatten(2) for src in src_views],
                                      dim=-1).permute(0, 2, 1)

        n_views, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] *
                input_spatial_shapes[:, 1]).sum() == Len_in
        value = self.rayconv(input_flatten)

        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(n_views, Len_in, self.n_heads,
                           self.d_model // self.n_heads)

        # combine the view-specific ref point feature and the joint-specific
        # query feature
        sampling_offsets = self.sampling_offsets(
            torch.stack(ref_point_feat_views_alllvs, dim=2) +
            query.unsqueeze(2)).view(n_views, Len_q, self.n_heads, feat_lvls,
                                     self.n_points, 2)
        attention_weights = self.attention_weights(
            torch.stack(ref_point_feat_views_alllvs, dim=2) +
            query.unsqueeze(2)).view(n_views, Len_q, self.n_heads,
                                     feat_lvls * self.n_points)
        attention_weights = F.softmax(attention_weights,
                                      -1).view(n_views, Len_q, self.n_heads,
                                               feat_lvls, self.n_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]],
                -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.n_points \
                * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError('Last dim of reference_points must be 2 or 4, \
                but get {} instead.'.format(reference_points.shape[-1]))
        output = DeformFunction.apply(value, input_spatial_shapes,
                                      input_level_start_index,
                                      sampling_locations, attention_weights,
                                      self.im2col_step)
        output = self.output_proj(output)
        return output


class DeformFunction(Function):

    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = DF.deform_forward(value, value_spatial_shapes,
                                   value_level_start_index, sampling_locations,
                                   attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (value, value_spatial_shapes, value_level_start_index,
         sampling_locations, attention_weights) = ctx.saved_tensors

        grad_value, grad_sampling_loc, grad_attn_weight = \
            DF.deform_backward(
                value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights,
                grad_output, ctx.im2col_step)

        return \
            grad_value, \
            None, None, \
            grad_sampling_loc, \
            grad_attn_weight, None


def deform_core_pytorch(value, value_spatial_shapes, sampling_locations,
                        attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes],
                             dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_,
        # M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).\
            transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_]\
            .transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) ->
    # (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.\
        transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).view(N_, M_ * D_, Lq_)
    return output.transpose(1, 2).contiguous()


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError('invalid input '
                         'for _is_power_of_2: '
                         '{} (type: {})'.format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0
