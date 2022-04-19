import numpy as np
import torch
from torch import nn

from src.train.train import (
    get_conv,
    GlobalAveragePooling2d,
    ResidualBlock,
    train,
    prepare_data,
)

NUM_CLASSES = 10
BATCH_SIZE = 16


def test_get_conv_with_bn():
    conv = get_conv(3, 8, 16)

    assert len(conv) == 3
    assert isinstance(conv[0], nn.Conv2d)
    assert conv[0].in_channels == 8
    assert conv[0].out_channels == 16
    assert isinstance(conv[1], nn.ReLU)
    assert isinstance(conv[2], nn.BatchNorm2d)
    assert conv[2].num_features == 16


def test_test_get_conv_without_bn():
    conv = get_conv(3, 8, 16, with_bn=False, with_relu=False)

    assert len(conv) == 1
    assert isinstance(conv[0], torch.nn.Conv2d)
    assert conv[0].in_channels == 8
    assert conv[0].out_channels == 16


def test_gap():
    gap = GlobalAveragePooling2d()
    x = torch.randn(4, 3, 16, 16)
    y = gap(x)
    assert y.size() == (4, 3)


def test_residual_block():
    block = ResidualBlock(4, 4, 2)

    assert len(block.layers) == 2
    assert len(block.layers[0]) == 3
    assert len(block.layers[1]) == 3
    assert isinstance(block.layers[1][2], nn.BatchNorm2d)

