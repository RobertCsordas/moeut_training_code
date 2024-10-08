import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd
from typing import Any, Dict, List, Optional
from torch import Tensor
from layers.cvmm import cvmm, cvmm_prepare_sel


class ParallelLinear(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, expert_size, weight, bias=None):
        output = ParallelLinear.forward_scriptable(input, expert_size, weight, bias)
        # assert torch.allclose(ParallelLinear._forward_scriptable(input, expert_size, weight, bias),  output)
        ctx.save_for_backward(input, expert_size, weight, bias)
        return output

    @staticmethod
    @torch.jit.script
    def forward_scriptable(input: Tensor, expert_size: Tensor,
                           weight: Tensor, bias: Optional[Tensor]):
        output_buf: Tensor = torch.empty((input.size(0), weight.size(2)),
                                         device=input.device, dtype=input.dtype)
        num_linears = weight.size(0)

        expert_size_list: List[int] = expert_size.tolist()
        input_list = input.split(expert_size_list, dim=0)
        output_buf_list = output_buf.split(expert_size_list)

        for i in range(num_linears):
            torch.mm(input_list[i], weight[i], out=output_buf_list[i])

        if bias is not None:
            for i in range(num_linears):
                output_buf_list[i].add_(bias[i])

        output = output_buf
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        input, expert_size, weight, bias = ctx.saved_tensors
        return ParallelLinear.backward_scriptable(
            grad_out, input, expert_size,
            weight, bias
        )

    @staticmethod
    @torch.jit.script
    def backward_scriptable(grad_out: Tensor,
                 input: Tensor, expert_size: Tensor,
                 weight: Tensor, bias: Optional[Tensor]):
        num_linears = weight.size(0)
        expert_size_list: List[int] = expert_size.tolist()
        input_list = input.t().split(expert_size_list, dim=1)
        grad_list = grad_out.split(expert_size_list, dim=0)

        d_input_buf = torch.empty_like(input)
        d_input_buf_list = d_input_buf.split(expert_size_list, dim=0)
        d_weight_buf = torch.empty_like(weight)

        weight_t = weight.permute(0, 2, 1)

        for i in range(num_linears):
            torch.mm(grad_list[i], weight_t[i], out=d_input_buf_list[i])
            torch.mm(input_list[i], grad_list[i], out=d_weight_buf[i])

        d_input = d_input_buf
        d_weight = d_weight_buf

        if bias is not None:
            d_bias_buf = torch.empty_like(bias)
            for i in range(num_linears):
                torch.sum(grad_list[i], dim=0, keepdim=False, out=d_bias_buf[i])
            d_bias = d_bias_buf
        else:
            d_bias = None

        return d_input, None, d_weight, d_bias


class ParallelExperts(nn.Module):
    def __init__(self, num_experts, input_size, output_size, bias=False) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.empty(num_experts, input_size, output_size))
        if bias:
            self.b = nn.Parameter(torch.zeros(num_experts, output_size))
        else:
            self.b = None
        self.reset_parameters()

    def extra_repr(self):
        return 'num_experts={}, input_size={}, output_size={}'.format(
            self.w.size(0), self.w.size(1), self.w.size(2))

    def reset_parameters(self) -> None:
        # std = math.sqrt(2.0 / float(self.w.size(1) + self.w.size(2)))
        # a = math.sqrt(3.0) * std
        nn.init.uniform_(self.w, -1. / self.w.size(1), 1. / self.w.size(1))
        if self.b is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w[0])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.b, -bound, bound)

    def forward(self, inputs, expert_size):
        # if inputs.ndim > 2:
        #     other_dims = inputs[0,...,0].flatten().shape[0]
        # else:
        #     other_dims = 1

        # if inputs.numel() == 0:
        #     return torch.zeros([*inputs.shape[:-1], self.w.shape[-1]], device=inputs.device, dtype=inputs.dtype)

        # indices = torch.cat([torch.full([s*other_dims], i, device=inputs.device) for i, s in enumerate(expert_size)])
        # sel = cvmm_prepare_sel(indices, self.w.shape[0])
        # out = cvmm(inputs.flatten(end_dim=-2), sel, self.w)

        # if self.b is not None:
        #     out = out + self.b[indices]

        # return out.view(*inputs.shape[:-1], self.w.shape[-1])

        results = ParallelLinear.apply(inputs, expert_size, self.w, self.b)
        return results
