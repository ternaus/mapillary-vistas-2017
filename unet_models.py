import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.module import Module

import utils
import dataset

class SELU(Module):
    """Applies element-wise, :math:`f(x) = max(0,x) + min(0, alpha * (exp(x) - 1))`

    Args:
        alpha: the alpha value for the ELU formulation
        inplace: can optionally do the operation in-place

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = nn.ELU()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def __init__(self, alpha=1.6732632423543772848170429916717, inplace=False):
        super(SELU, self).__init__()
        self.alpha = alpha
        self.inplace = inplace
        self.scale = 1.0507009873554804934193349852946

    def forward(self, input):
        return F.elu(input, self.alpha, self.inplace) * self.scale

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + 'alpha=' + str(self.alpha) \
            + inplace_str + ')'


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


def concat(xs):
    return torch.cat(xs, 1)


class Conv3BN(nn.Module):
    def __init__(self, in_: int, out: int, bn=True):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.bn = nn.BatchNorm2d(out) if bn else None
        # self.activation = nn.ReLU(inplace=True)
        self.activation = nn.ELU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x


class UNetModule(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.l1 = Conv3BN(in_, out)
        self.l2 = Conv3BN(out, out)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


class UNet(nn.Module):
    module = UNetModule

    def __init__(self,
                 input_channels: int=3,
                 filters_base: int=32,
                 filter_factors=(1, 2, 4, 8, 16)):
        super().__init__()
        filter_sizes = [filters_base * s for s in filter_factors]
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        for i, nf in enumerate(filter_sizes):
            low_nf = input_channels if i == 0 else filter_sizes[i - 1]
            self.down.append(self.module(low_nf, nf))
            if i != 0:
                self.up.append(self.module(low_nf + nf, low_nf))
        bottom_s = 4
        pool = nn.MaxPool2d(2, 2)
        pool_bottom = nn.MaxPool2d(bottom_s, bottom_s)
        upsample = nn.UpsamplingNearest2d(scale_factor=2)
        upsample_bottom = nn.UpsamplingNearest2d(scale_factor=bottom_s)
        self.downsamplers = [None] + [pool] * (len(self.down) - 1)
        self.downsamplers[-1] = pool_bottom
        self.upsamplers = [upsample] * len(self.up)
        self.upsamplers[-1] = upsample_bottom
        self.conv_final = nn.Conv2d(filter_sizes[0], dataset.N_CLASSES, 1)

    def forward(self, x):
        xs = []
        for downsample, down in zip(self.downsamplers, self.down):
            x_in = x if downsample is None else downsample(xs[-1])
            x_out = down(x_in)
            xs.append(x_out)

        x_out = xs[-1]
        for x_skip, upsample, up in reversed(
                list(zip(xs[:-1], self.upsamplers, self.up))):
            x_out = upsample(x_out)
            x_out = up(concat([x_out, x_skip]))

        x_out = self.conv_final(x_out)
        return F.log_softmax(x_out)


class Loss:
    def __init__(self, dice_weight=0.0):
        self.nll_loss = nn.NLLLoss2d()
        self.dice_weight = dice_weight

    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)
        if self.dice_weight:
            cls_weight = self.dice_weight / dataset.N_CLASSES
            eps = 1e-5
            for cls in range(dataset.N_CLASSES):
                dice_target = (targets == cls).float()
                dice_output = outputs[:, cls].exp()
                intersection = (dice_output * dice_target).sum()
                # union without intersection
                uwi = dice_output.sum() + dice_target.sum() + eps
                loss += (1 - intersection / uwi) * cls_weight
            loss /= (1 + self.dice_weight)
        return loss
