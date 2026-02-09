import rootutils
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.logger import LOG


def main(cfg: DictConfig) -> None:
    LOG.info("seed: {0}".format(cfg.seed))


if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    cfg = cfg["download_tiles"]

    main(cfg)
