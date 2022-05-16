from pathlib import Path

import detectron2.data.transforms as T
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, get_cfg
from detectron2.modeling import build_model
from detectron2.data import MetadataCatalog
from torch.utils.data import DataLoader, Dataset 
from torch import cuda

from ditod import add_vit_config


# TODO: add docstrings

"""
    Batch predictor, mostly copy/ paste from Kasper Fromm Pedersen
    GitHub: https://github.com/fromm1990
    source: https://github.com/facebookresearch/detectron2/issues/282
"""

class ImageDataset(Dataset):

    def __init__(self, imagery):
        self.imagery = imagery

    def __getitem__(self, index):
        # returns image at given index as ndarrays
        return self.imagery[index]

    def __len__(self):
        return len(self.imagery)


class BatchPredictor:
    def __init__(self, cfg, label_map, batch_size, workers):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.label_map= label_map 
        self.batch_size = batch_size
        self.workers = workers
        self.model = build_model(self.cfg)
        self.model.eval()

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
            cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __collate(self, batch):
        #   https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/defaults.py
        data = []
        for image in batch:
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                image = image[:, :, ::-1]
            height, width = image.shape[:2]

            image = self.aug.get_transform(image).apply_image(image)
            image = image.astype("float32").transpose(2, 0, 1)
            image = torch.as_tensor(image)
            data.append({"image": image, "height": height, "width": width})
        return data

    def __call__(self, imagery):
        """[summary]

        :param imagery: [description]
        :type imagery: List[ndarrays] # CV2 format
        :yield: Predictions for each image
        :rtype: [type]
        """
        dataset = ImageDataset(imagery)
        loader = DataLoader(
            dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=self.__collate,
            pin_memory=True
        )

        predictions= []
        with torch.no_grad():
            for batch in loader:
                results = self.model(batch)
                instances = [result['instances'] for result in results]
                predictions.extend(instances)
        return predictions

class LayoutDetection:
    def __init__(
            self,
            cfg_path, 
            weights_path,
            batch_size=1,
            workers=1,
            threshold=0.75,
            label_map=["text", "title", "list", "table", "figure"]
            ):

        opts = ["MODEL.WEIGHTS", weights_path]
        cfg = get_cfg()
        add_vit_config(cfg)
        cfg.merge_from_file(cfg_path)
        cfg.merge_from_list(opts)
        device = "cuda" if cuda.is_available() else "cpu"
        cfg.MODEL.DEVICE = device
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

        self.cfg = cfg
        self.label_map = label_map
        self.batch_size = batch_size
        self.workers = workers

    def getPredictor(self):
        predictor = BatchPredictor(
                cfg=self.cfg, 
                label_map=self.label_map, 
                batch_size=self.batch_size, 
                workers=self.workers)
        return predictor

    def getMetadata(self):
        metadata = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])
        metadata.set(thing_classes=self.label_map)
        return metadata

