from torch.nn import (Module, Conv1d, BatchNorm1d, Dropout, MaxPool1d, ReLU, Sequential,
                      AvgPool1d, Flatten, Linear, Sigmoid, LSTM)
import numpy as np
import math


class GoodfellowBlock(Module):
    """Single convolution block of the network"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate, p_dropout, padding="same", maxpooling=0):
        super().__init__()
        self.conv = Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation_rate)
        self.batch_norm = BatchNorm1d(out_channels,  momentum=0.1)
        self.maxpooling = maxpooling
        self.p_dropout = p_dropout
        self.dropout = Dropout(p=p_dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = ReLU()(x)
        if self.maxpooling:
            x = MaxPool1d(kernel_size=2, stride=2, padding=0)(x)
        x = self.dropout(x)
        return x


class DepthwiseConvolution(Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, stride):
        super().__init__()

        if stride == 1:
            padding = 'same'
        else:
            padding = math.floor(kernel_size / stride)
        self.conv = Sequential(
            Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                   dilation=dilation, groups=in_channels, padding=padding, stride=stride),
            Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding='same', stride=1)
        )

    def forward(self, x):
        return self.conv(x)


class DepthwiseGoodFellowBlock(Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate, stride, p_dropout, max_pooling=0):
        super().__init__()
        self.layer = None

        if max_pooling != 0:
            self.layer = Sequential(
                DepthwiseConvolution(in_channels, out_channels, kernel_size, dilation=dilation_rate, stride=stride),
                BatchNorm1d(out_channels, momentum=0.1),
                ReLU(),
                MaxPool1d(kernel_size=2, stride=max_pooling, padding=0),
                Dropout(p=p_dropout)
            )
        else:
            self.layer = Sequential(
                DepthwiseConvolution(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation_rate),
                BatchNorm1d(out_channels, momentum=0.1),
                ReLU(),
                Dropout(p=p_dropout)
            )
        self.kernel_size = kernel_size
        self.dr = dilation_rate
        self.stride = stride

    def forward(self, x):
        out = self.layer(x)
        return out


class GoodfellowNet(Module):
    """Network module definition.
    """
    def __init__(self, block=GoodfellowBlock, len_input=1280, out_nodes=1, p_dropout=0.3):
        super().__init__()
        self.p_dropout = p_dropout
        in_channels = [1, 320, 256, 256, 256, 256, 128, 128, 128, 128, 128, 128, 64]   # input channels of each block
        out_channels = in_channels[1:] + [64]                                           # output channels
        kernel_size = [24, 16, 16, 16, 16, 8, 8, 8, 8, 8, 8, 8, 8]
        dilation_rate = [1, 2, 4, 4, 4, 4, 6, 6, 6, 6, 8, 8, 8]
        max_pooling = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
        # Convolutional layers
        blocks = [block(in_channels=in_channels[i], out_channels=out_channels[i], kernel_size=kernel_size[i],
                        dilation_rate=dilation_rate[i], maxpooling=max_pooling[i], p_dropout=self.p_dropout)
                  for i in range(13)]
        self.conv_blocks = Sequential(*blocks)
        # Layers after convolutions
        self.avg_pooling = AvgPool1d(kernel_size=int(len_input/8))
        self.batch_norm = BatchNorm1d(64, momentum=0.1)
        self.linear = Linear(in_features=64, out_features=out_nodes)
        self.sigmoid = Sigmoid()

    def forward(self, ecg):
        out = self.conv_blocks(ecg)             # Convolutions
        out = self.avg_pooling(out).squeeze(2)  # Global average pooling
        out = self.batch_norm(out)              # Batch normalization
        out = self.linear(out)                  # Linear layer
        return self.sigmoid(out)


class DepthwiseGoodFellow(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, max_pooling, p_dropout, dilation_rate=None):
        super().__init__()
        lengths = set([len(in_channels), len(out_channels), len(kernel_size), len(max_pooling)])

        if len(lengths) > 1:
            raise NameError('Input lists of different size')
        lengths = None

        blocks = [DepthwiseGoodFellowBlock(in_channels=in_channels[i],
                                           out_channels=out_channels[i],
                                           kernel_size=kernel_size[i],
                                           dilation_rate=dilation_rate[i],
                                           max_pooling=max_pooling[i],
                                           stride=stride[i],
                                           p_dropout=p_dropout) for i in range(len(in_channels))]
        self.blocks = Sequential(*blocks)

    def forward(self, x):
        x = self.blocks(x)
        return x


class LstmNetwork(Module):
    def __init__(self, in_channels, out_channels, dilation, kernel_size, max_pooling,
                 stride, in_len=1280, ratio=1., num_layers=1, bidirectional=True, many_to_many=False, p_dropout=.3):
        super().__init__()

        self.emb = DepthwiseGoodFellow(
            in_channels=in_channels,
            out_channels=out_channels,
            dilation_rate=dilation,
            kernel_size=kernel_size,
            max_pooling=max_pooling,
            stride=stride,
            p_dropout=p_dropout
        )

        self.ratio = ratio

        self.gru = LSTM(input_size=out_channels[-1], hidden_size=int(out_channels[-1] * self.ratio),
                        batch_first=False, bidirectional=bidirectional, num_layers=num_layers)

        mp = np.array(max_pooling)
        mp[mp == 0] = 1
        d = 2 if bidirectional else 1
        self.many_to_many = many_to_many

        if not many_to_many:
            self.linear = Linear(in_features=int(out_channels[-1] * self.ratio * d), out_features=1)

        self.sigmoid = Sigmoid()
        self.flatten = Flatten()
        self.batchnorm = BatchNorm1d(int(out_channels[-1] * self.ratio * d))
        self.relu = ReLU()
        self.avgpool = AvgPool1d(kernel_size=int(in_len / np.prod(mp)))

    def forward(self, x):

        out = self.emb(x).permute(2, 0, 1)

        out = self.gru(out)[0]
        # out = out[-1,:,:]
        out = out.permute(1, 2, 0)
        if not self.many_to_many:
            out = out[:, :, -1]
            out = self.batchnorm(self.relu(out))
            out = self.flatten(out)
            return self.sigmoid(self.linear(out))
        else:
            out = self.avgpool(out)
            out = self.batchnorm(out)
            out = self.flatten(out)
            return self.sigmoid(out)
