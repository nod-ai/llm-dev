import torch.nn as nn
import torch

def quantize(tensor, scale, zero_point, is_asym=False):
    if is_asym:
        clamp_min, clamp_max = torch.tensor(0.), torch.tensor(15)
    else:
        clamp_min, clamp_max = torch.tensor(-7), torch.tensor(7)
    quant_tensor = torch.clamp(torch.round(tensor/scale + zero_point), clamp_min, clamp_max) 
    return quant_tensor

def dequantize(tensor, scale, zero_point):
    return (tensor - zero_point) * scale


class QuantLinear(nn.Module):
    def __init__(self, in_ch, out_ch, quant_param):
        super().__init__()
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.linear = nn.Linear(in_ch, out_ch)
        self.group_dim = 1
        self.group_size = 32 # Can be parametrized
        
        # Fields name are temporary
        # weight_scale has shape [out_ch, in_ch//self.group_size, 1]
        po2_weight_scale = torch.tensor(quant_param['weight_scale']).view(quant_param['weight_scale_shape'])
        assert po2_weight_scale.dtype == torch.int8
        assert po2_weight_scale.shape == [out_ch, in_ch//self.group_size, 1]
        weight_scale = torch.pow(2, po2_weight_scale.to(torch.float16)) # Assuming fp16 dtype

        # weight_zp has shape [out_ch, in_ch//self.group_size, 1]
        weight_zp = torch.tensor(quant_param['weight_zp']).view(quant_param['weight_zp_shape'])
        assert quant_param['weight_zp_dtype'] == 'torch.int8', f"Weight Zero-Point dtype should be 'torch.int8', found: {quant_param['weight_zp_dype']}"
        assert weight_zp.shape == [out_ch, in_ch//self.group_size, 1]
        assert torch.max(weight_zp) <= 15., "Max value is above uint4"

        self.register_buffer('weight_scale', weight_scale)
        self.register_buffer('weight_zp', weight_zp)


    # I.e., "fake quantization"
    def qdq_forward(self, x):
        weight = weight.view(self.out_ch, self.in_ch//self.group_size, self.group_size)
        quant_weight = quantize(weight, self.weight_scale, self.weight_zp, is_asym=True)
        dequantized_weight = dequantize(quant_weight, self.weight_scale, self.weight_zp)
        # Go from  [out_ch, in_ch // group_size, group_size] to [out_ch, in_ch]
        dequantized_weight = dequantized_weight.flatten(1)
        out = torch.nn.functional.linear(x, dequantized_weight, self.linear.bias)
        return out
