from torch.nn import (Module, Conv1d, BatchNorm1d, Dropout, MaxPool1d, ReLU, Sequential,
                      AvgPool1d, Linear, Sigmoid)

class GoodfellowBlock(Module):
    """
    Single convolution block of the network

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        dilation_rate (int): Dilation rate for the convolution.
        p_dropout (float): Probability of dropout.
        padding (str, optional): Padding mode for the convolution. Defaults to "same".
        max_pooling (int, optional): If non-zero, applies max pooling with kernel size 2. Defaults to 0.

    Methods:
        forward(x): Performs a forward pass through the block.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation_rate: int, p_dropout: float,
                 padding="same", max_pooling: bool = 0):
        super().__init__()
        self.conv = Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation_rate)
        self.batch_norm = BatchNorm1d(out_channels,  momentum=0.1)
        self.maxpooling = max_pooling
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


class GoodfellowNet(Module):
    """
    Convolutional neural network module.

    It takes as input an ECG and gives as output a probability. It can be used for
    both classification and prediction.

    Args:
        block (Module, optional): The block type used in the convolutional layers. Defaults to GoodfellowBlock.
        in_channels (int, optional): Number of input nodes. Defaults to 12.
        len_input (int, optional): Length of the input sequence. Defaults to 1280.
        out_nodes (int, optional): Number of output nodes in the final layer. Defaults to 1.
        p_dropout (float, optional): Dropout probability applied in each block. Defaults to 0.3.

    Methods:
        forward(ecg): Performs a forward pass through the network for input ECG data.
    """
    def __init__(self, block=GoodfellowBlock, in_channels: int = 12, len_input: int = 1280, out_nodes: int = 1,
                 p_dropout: float = 0.3):
        super().__init__()
        self.p_dropout = p_dropout
        # Set number of input/output channels, kernel size, dilation rate, and max pooling for each block.
        in_channels = [in_channels, 320, 256, 256, 256, 256, 128, 128, 128, 128, 128, 128, 64]
        out_channels = in_channels[1:] + [64]
        kernel_size = [24, 16, 16, 16, 16, 8, 8, 8, 8, 8, 8, 8, 8]
        dilation_rate = [1, 2, 4, 4, 4, 4, 6, 6, 6, 6, 8, 8, 8]
        max_pooling = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
        # Convolutional layers
        blocks = [block(in_channels=in_channels[i], out_channels=out_channels[i], kernel_size=kernel_size[i],
                        dilation_rate=dilation_rate[i], max_pooling=max_pooling[i], p_dropout=self.p_dropout)
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
