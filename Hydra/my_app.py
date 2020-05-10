import hydra
from omegaconf import DictConfig
import numpy as np
import imageio as iio
import os, logging

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(config_path="conf/config.yaml")
def my_app(cfg: DictConfig) -> "args":
    print("\nCurrent working directory  : {}".format(os.getcwd()))
    print("Original working directory : {}".format(hydra.utils.get_original_cwd()))
    print("to_absolute_path('foo')    : {}".format(hydra.utils.to_absolute_path("foo")))
    print("to_absolute_path('/foo')   : {}\n".format(hydra.utils.to_absolute_path("/foo")))

    # print(cfg.pretty())
    log.info(f"\n{cfg.pretty()}")

    if cfg.model=="merger":
        Merger = hydra.utils.instantiate(cfg.model)
        Merger.shape()

    text_file_total = open(f"gen_file.txt", "w")
    text_file_total.write("##########################################\n")
    text_file_total.close()

    return cfg


if __name__ == "__main__":
    args = my_app()
    print("args", args)
