from torch import nn, Tensor
import torch.nn.functional as F
from math import ceil

cfg = {
    'A' : [256, 'M', 256, 'M'],
    'B' : [128, 'M', 128, 'M', 256, 'M', 512, 'M'],
    'C' : [64,  'M', 64,  64,  'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512],
    'D' : [64,  'M', 64,  64,  64,  64,  'M', 128, 128, 128, 128, 'M', 256, 256,
        256, 256, 'M', 512, 512, 512, 512],
    'E' : [48,  'M', 48,  48,  48,  'M', 96,  96,  96,  96,  'M', 192, 192, 192,
        192, 192, 192, 'M', 384, 384, 384]
}

def _same_padding(x:Tensor, kernel_size:int, stride:int):
    in_len = x.shape[-1]
    out_len = ceil(in_len / stride)
    pad_width = max(0, (out_len - 1) * stride + kernel_size - in_len)
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    return F.pad(x, (pad_left, pad_right))

class BasicBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=3,
        stride:int=1):
        super(BasicBlock, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x:Tensor):
        x = _same_padding(x, self.kernel_size, self.stride)
        x = F.relu(self.bn(self.conv(x)))
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=3,
        stride:int=1):
        super(ResBlock, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(*[
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            ])
        else:
            self.downsample = None
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride)
        self.bn2 = nn.BatchNorm1d(out_channels)
        if stride != 1:
            self.downsample = nn.Sequential(*[
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            ])

    def forward(self, x):
        residual = x
        x = _same_padding(x, self.kernel_size, self.stride)
        x = F.relu(self.bn1(self.conv1(x)))
        x = _same_padding(x, self.kernel_size, self.stride)
        x = F.relu(self.bn2(self.conv2(x)))
        if self.downsample:
            residual = _same_padding(residual, 1, self.stride)
            residual = self.downsample(residual)
        x = x + residual
        return x

def _make_layers(cfg:list, in_channels:int=1, residual:bool=False):
    layers = []
    kernel_size = 80
    stride = 4
    for i, l in enumerate(cfg):
        if i == 0:
            layers.append(BasicBlock(in_channels, l, 80, 4))
            kernel_size = 3
            stride = 1
            in_channels = l
        else:
            if l == 'M':
                layers.append(nn.MaxPool1d(4))
            else:
                if residual:
                    layers.append(ResBlock(in_channels, l, kernel_size, stride))
                else:
                    layers.append(BasicBlock(in_channels, l, kernel_size,
                        stride))
                in_channels = l
    return nn.Sequential(*layers)

class Mn(nn.Module):
    def __init__(self, cfg:list, num_classes:int=2, residual:bool=False):
        super(Mn, self).__init__()
        if type(cfg[-1]) == str:
            in_features = cfg[-2]
        else:
            in_features = cfg[-1]
        self.features = _make_layers(cfg, 1, residual)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x:Tensor):
        x = self.features(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.classifier(x)
        return F.softmax(x, dim=-1)

def M3(num_classes:int=2):
    return Mn(cfg['A'], num_classes, False)

def M5(num_classes:int=2):
    return Mn(cfg['B'], num_classes, False)

def M11(num_classes:int=2):
    return Mn(cfg['C'], num_classes, False)

def M18(num_classes:int=2):
    return Mn(cfg['D'], num_classes, False)

def M34_res(num_classes:int=2):
    return Mn(cfg['E'], num_classes, True)
