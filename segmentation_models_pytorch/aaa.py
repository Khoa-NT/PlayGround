import segmentation_models_pytorch as smp
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

import logging

@hydra.main(config_path="configs/segmentation.yaml")
def main(cfg: DictConfig) -> None:

    model = hydra.utils.instantiate(cfg.model)
    # model2 = smp.unet.model('resnet34', classes=3, activation='softmax')
    model2 = smp.utils.metrics.Accuracy
    # model2 = hydra.utils.instantiate(cfg.model2)
    # model2 = torch.hub.load('pytorch/vision:v0.5.0', 'deeplabv3_resnet101', pretrained=True)
    print(model)
    # print(model2)
    smp.utils.train.TrainEpoch

if __name__ == "__main__":
    main()


