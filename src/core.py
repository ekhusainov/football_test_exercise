import logging
import logging.config

import torch
import click

import yaml

from src.train.train import train_network
from src.predict.predict import eval
from src.enities.logging_params import setup_logging

APPLICATION_NAME = "core"

logger = logging.getLogger(APPLICATION_NAME)

@click.command(name="choose")
@click.argument("action")
@click.argument("img_filepath")
def main(action: str, img_filepath: str):
    setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Current device is '%s'", device)
    if action == "train":
        train_network()
    elif action == "inference":
        print(eval(img_filepath))
    else:
        logger.warning("There is no such command.")
    
    


if __name__ == "__main__":
    main()
