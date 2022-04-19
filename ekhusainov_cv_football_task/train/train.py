import logging
import os
from PIL import Image

import yaml
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import Adam
from torch.nn.functional import cross_entropy, relu
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms

from ekhusainov_cv_football_task.enities.logging_params import setup_logging

APPLICATION_NAME = "train"
DATA_FILEPATH = "data\GrayScaleTrain"
RESIZE_SIZE = 224
IMAGE_CHANNELS = 3
NUM_CLASSES = 10
NUM_EPOCHS = 10
BATCH_SIZE = 16
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_FILEPATH = "models/resnet.pt"

logger = logging.getLogger(APPLICATION_NAME)


def prepare_data():
    setup_logging()
    transform = transforms.Compose([
        transforms.Resize([RESIZE_SIZE, RESIZE_SIZE]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    images = ImageFolder(DATA_FILEPATH, transform=transform)
    dataloader = DataLoader(
        dataset=images,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    logger.info("Data is ready.")
    return dataloader


def get_conv(kernel_size, in_features, out_features, with_bn=True, with_relu=True):
    """Create conv -> [relu] -> [bn] layers, embedded in torch.nn.Sequential module.

    ! Conv layer must preserve spatial tensor dims (i.e. apply zero padding).

    Args:
        kernel_size: int
        in_features: int
        out_features: int
        with_bn: bool
        with_relu: bool

    Returns:
        torch.nn.Sequential
    """
    layers = [
        nn.Conv2d(in_features, out_features, kernel_size, 1, kernel_size // 2)
    ]

    if with_relu:
        layers.append(nn.ReLU(inplace=(with_bn is True)))

    if with_bn:
        layers.append(nn.BatchNorm2d(out_features))

    return nn.Sequential(*layers)


class GlobalAveragePooling2d(nn.Module):
    def forward(self, x):
        """GAP forward pass.

        Args:
            x: torch.Tensor, size B x C x H x W.

        Returns:
            torch.Tensor, size B x C.
        """
        y = torch.mean(x, dim=(2, 3))
        return y


class ResidualBlock(nn.Module):
    def __init__(self, num_input_features, num_features, num_layers, with_bn=True):
        super().__init__()

        if num_input_features != num_features:
            self.projection = nn.Conv2d(
                num_input_features, num_features, 1, 1, 0)
        else:
            self.projection = None

        layers = []
        for i in range(num_layers):
            conv = get_conv(3, num_input_features,
                            num_features, with_bn=with_bn)
            layers.append(conv)
            num_input_features = num_features
        self.layers = nn.Sequential(*layers)

        self.num_input_features = num_input_features
        self.num_features = num_features
        self.num_layers = num_layers
        self.with_bn = with_bn

    def forward(self, x):
        """Forward pass.
        Applies convolution layers and skip-connection; self.projection, if necessary.

        Args:
            x: torch.Tensor, size B x C x H x W.

        Returns:
            torch.Tensor, size B x C x H x W.
        """
        x_input = x

        for layer in self.layers:
            x = layer(x)

        if self.projection is not None:
            x += self.projection(x_input)
        else:
            x += x_input

        return relu(x)

    def __repr__(self):
        out = f"ResidualBlock(num_input_features={self.num_input_features}, num_features={self.num_features}, num_layers={self.num_layers}, with_bn={self.with_bn})"
        for l in self.layers:
            out += "\n" + "\t" + repr(l)
        return out


class Block(nn.Module):
    def __init__(self, num_input_features, num_features, num_layers, with_bn=True):
        super().__init__()

        layers = []
        for i in range(num_layers):
            conv = get_conv(3, num_input_features,
                            num_features, with_bn=with_bn)
            layers.append(conv)
            num_input_features = num_features
        self.layers = nn.Sequential(*layers)

        self.num_input_features = num_input_features
        self.num_features = num_features
        self.num_layers = num_layers
        self.with_bn = with_bn

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        out = f"Block(num_input_features={self.num_input_features}, num_features={self.num_features}, num_layers={self.num_layers}, with_bn={self.with_bn})"
        for l in self.layers:
            out += "\n" + "\t" + repr(l)
        return out


def create_resnet34(num_input_features=IMAGE_CHANNELS, num_classes=NUM_CLASSES, with_bn=True):
    pool = nn.MaxPool2d((2, 2))
    gap = GlobalAveragePooling2d()
    fc = nn.Linear(512, num_classes)
    return nn.Sequential(
        get_conv(7, num_input_features, 64, with_bn=with_bn),
        pool,
        ResidualBlock(64, 64, 2, with_bn=with_bn),
        ResidualBlock(64, 64, 2, with_bn=with_bn),
        ResidualBlock(64, 64, 2, with_bn=with_bn),
        pool,
        ResidualBlock(64, 128, 2, with_bn=with_bn),
        ResidualBlock(128, 128, 2, with_bn=with_bn),
        ResidualBlock(128, 128, 2, with_bn=with_bn),
        pool,
        ResidualBlock(128, 256, 2, with_bn=with_bn),
        ResidualBlock(256, 256, 2, with_bn=with_bn),
        ResidualBlock(256, 256, 2, with_bn=with_bn),
        ResidualBlock(256, 256, 2, with_bn=with_bn),
        ResidualBlock(256, 256, 2, with_bn=with_bn),
        pool,
        ResidualBlock(256, 512, 2, with_bn=with_bn),
        ResidualBlock(512, 512, 2, with_bn=with_bn),
        ResidualBlock(512, 512, 2, with_bn=with_bn),
        gap,
        fc,
    )


def train(model, dataset, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LR, device=DEVICE):
    """Model training routine function. 
    Uses Adam optimizer & cross-entropy loss.

    Args:
        model: torch.nn.Module
        dataset: torch.utils.data.Dataset
        num_epochs: int
        batch_size: int
        lr: float
        device: str

    Returns:
        losses: list of float values of length num_epochs * len(dataloader)
    """
    setup_logging()
    model.train()
    model = model.to(device)

    dataloader = prepare_data()
    optimizer = Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(num_epochs):
        logger.info("Start epoch %s", repr(epoch + 1))
        for batch in tqdm(dataloader):
            xs, ys_true = batch
            logits_pred = model(xs.to(device)).cpu()
            loss = cross_entropy(logits_pred, ys_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            losses.append(current_loss)
            logger.info("Loss: %s", repr(round(current_loss, 4)))
        torch.save(model, MODEL_FILEPATH)
    return losses


def get_football_command_names():
    setup_logging()
    commands = os.listdir(DATA_FILEPATH)
    logger.info("We have commands %s", repr(commands))


def train_network():
    setup_logging()
    logger.info("Start to train network.")
    dataset = prepare_data()
    resnet34 = create_resnet34()
    train(resnet34, dataset)


if __name__ == "__main__":
    train_network()
