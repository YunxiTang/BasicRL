import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import pdb

@hydra.main(config_path='Config', config_name='config', version_base=None)
def my_app(config):
    import torch
    import numpy as np
    print(config)
    running_dir = str(hydra.utils.get_original_cwd())
    working_dir = str(Path.cwd())
    print(f"The current running directory is {running_dir}")
    print(f"The current working directory is {working_dir}")

    # To access elements of the config
    print(f"The batch size is {config.batch_size}")
    print(f"The learning rate is {config['lr']}")

if __name__ == "__main__":
    # breakpoint()
    my_app()