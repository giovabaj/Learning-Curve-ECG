from torch.nn import (Module, Conv1d, BatchNorm1d, Dropout, MaxPool1d, ReLU, Sequential,
                      AvgPool1d, Flatten, Linear, Sigmoid, LSTM)
import numpy as np
import math


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
