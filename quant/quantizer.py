import numpy as np
import torch
import torch.nn as nn
import math


class Quantizer(nn.Module):

    def __init__(self, shape=1, outliers=False):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))
        self.outliers = outliers

    def configure(self, bits, perchannel=False, sym=True, mse=False, norm=2.4, grid=100, maxshrink=.8, trits=False):
        self.bits = bits
        self.maxq = torch.tensor(2**bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if trits:
            self.maxq = torch.tensor(-1)
        self.scale = torch.zeros_like(self.scale)
        self.outlier_min = torch.zeros_like (self.scale).fill_(float('inf'))
        self.outlier_max = torch.zeros_like (self.scale).fill_(-float('inf'))

    def change_bits (self, bits):
        self.bits = bits
        self.maxq = torch.tensor (2**bits - 1)

    def _quantize_without_outliers (self, x, scale, zero, maxq):
        mask = (x < self.outlier_min) | (x > self.outlier_max)
        outliers = mask * x
        x_without_outlier = x.clone()
        # x_without_outlier = torch.clamp(x, self.outlier_min, self.outlier_max)
        if maxq < 0:
            res = (x_without_outlier > scale / 2).float() * scale + (x_without_outlier < zero / 2).float() * zero
            res[mask] = 0
            res += outliers
            return res
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        res = scale * (q - zero)
        res[mask] = 0
        res += outliers
        return res

    def _quantize(self, x, scale, zero, maxq):
        if maxq < 0:
            return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        return scale * (q - zero)

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)
        self.outlier_min = torch.zeros_like (self.scale).fill_(float('inf'))
        self.outlier_max = torch.zeros_like (self.scale).fill_(-float('inf'))
        self.outlier_min = self.outlier_min.to(dev)
        self.outlier_max = self.outlier_max.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.outliers:
            row_means = torch.mean (x, dim=1)
            row_stds = torch.std (x, dim=1)
            self.outlier_min = self.outlier_min.squeeze ()
            self.outlier_max = self.outlier_max.squeeze ()
            self.outlier_min = torch.minimum (self.outlier_min, row_means - 3 * row_stds)
            self.outlier_max = torch.maximum (self.outlier_max, row_means + 3 * row_stds)
            xmin = torch.maximum (xmin, self.outlier_min)
            xmax = torch.minimum (xmax, self.outlier_max)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
            self.scale = xmax
            self.zero = xmin
        else:
            self.scale = (xmax - xmin) / self.maxq
            if self.sym:
                self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
            else:
                self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = self._quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            self.outlier_min = self.outlier_min.reshape(shape)
            self.outlier_max = self.outlier_max.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
            self.outlier_min = self.outlier_min.reshape((1, -1, 1, 1))
            self.outlier_max = self.outlier_max.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
            self.outlier_min = self.outlier_min.reshape((1, 1, -1))
            self.outlier_max = self.outlier_max.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)
            self.outlier_min = self.outlier_min.unsqueeze(0)
            self.outlier_max = self.outlier_max.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            if not self.outliers:            
                return self._quantize(x, self.scale, self.zero, self.maxq)
            else:
                return self._quantize_without_outliers (x, self.scale, self.zero, self.maxq)

        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)
