#   General utils
import logging

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from ditod import add_vit_config
from torch import cuda


def initializeModel(
    config_path,
    weights_path,
    threshold=0.75,
    label_map=["text", "title", "list", "table", "figure"],
):
    """Gets predictor and metadata

    Args:
        config_path (str): path to model configuration file (.yaml)
        weights (str): path to pre-trained weights
        threshold (float): detection score threshold. Defaults to 0.75
        label_map (list): label map used by the model. Defaults to example label map

    Returns:
        predictor and metadata respectively
    """

    logging.info(
        f"[Utils] Initializing model with a default threshold of {threshold} and the following label map: {label_map}"
    )
    opts = ["MODEL.WEIGHTS", weights_path]

    # instantiate config
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(config_path)

    # add model weights URL to config
    cfg.merge_from_list(opts)

    # set device
    device = "cuda" if cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device

    # set score threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

    # define model & classes
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    metadata.set(thing_classes=label_map)

    return predictor, metadata
