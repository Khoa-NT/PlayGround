import hydra
from omegaconf import DictConfig
import numpy as np
import imageio as iio
import os, logging

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    log.info(f"\n{cfg.pretty()}")

    python_class = hydra.utils.instantiate(cfg.code)
    python_class.run()

if __name__ == "__main__":
    my_app()