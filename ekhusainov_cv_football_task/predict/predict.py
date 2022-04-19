import logging
from PIL import Image

import yaml

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder


from ekhusainov_cv_football_task.enities.logging_params import setup_logging
from ekhusainov_cv_football_task.train.train import (
    BATCH_SIZE,
    RESIZE_SIZE,
)

APPLICATION_NAME = "predict"
MODEL_FILEPATH = "models/resnet.pt"
DATA_FILEPATH = "data/GrayScaleTrain"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(APPLICATION_NAME)

def predict(img):
    model = torch.load(MODEL_FILEPATH)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize([RESIZE_SIZE, RESIZE_SIZE]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    img = transform(img).to(DEVICE)

    images = ImageFolder(DATA_FILEPATH)
    logit = model(torch.concat([img] * BATCH_SIZE).reshape(BATCH_SIZE, 3, RESIZE_SIZE, RESIZE_SIZE).to(DEVICE))
    idx = torch.argmax(logit).item()
    return images.classes[idx]

def eval(filepath_to_img):
    setup_logging()
    img = Image.open(filepath_to_img).convert("RGB")
    answer = predict(img)
    logger.info("%s is %s", filepath_to_img, answer)
    return answer

