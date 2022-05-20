"""
    Mostly copy/ paste from Kasper Fromm Pedersen and Detectron2's DefaultPredictor class
    Source 1: https://github.com/facebookresearch/detectron2/issues/282
    Source 2: https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/defaults.py
"""
import detectron2.data.transforms as T
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, get_cfg
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from torch import cuda
from torch.utils.data import DataLoader, Dataset

from ditod import add_vit_config


class ImageDataset(Dataset):
    def __init__(self, imagery):
        """Initialize list of images (ndarrays)"""
        self.imagery = imagery

    def __getitem__(self, index):
        """Gets image at given index as (ndarray)"""
        return self.imagery[index]

    def __len__(self):
        """Gets number of samples in the dataset"""
        return len(self.imagery)


class BatchPredictor:
    def __init__(self, cfg, label_map, batch_size, workers):
        """Creates instance of BatchPredictor object

        Args:
            cfg (detectron2.config.config.CfgNode): Detectron2 config instance
            label_map (list): label map used by the model
            batch_size (int): batch size. Defaults to 1
            workers (int): number of workers. Defaults to 1
        """
        self.cfg = cfg.clone()  #   cfg can be modified by model
        self.label_map = label_map
        self.batch_size = batch_size
        self.workers = workers
        self.model = build_model(self.cfg)
        self.model.eval()

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __collate(self, batch):
        """Applies transformations to data in batch
        Source:
            https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/defaults.py

        Args:
            batch (list): list of images

        Returns:
            transformed data of given batch as list
        """
        data = []
        for image in batch:
            # apply pre-processing to image.
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
        """Runs inference

        Args:
            imagery (list): list of images (ndarray)

        Returns:
            layout predictions on each image
        """
        dataset = ImageDataset(imagery)
        loader = DataLoader(
            dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=self.__collate,
            pin_memory=True,
        )

        predictions = []
        with torch.no_grad():
            for batch in loader:
                results = self.model(batch)
                instances = [result["instances"] for result in results]
                predictions.extend(instances)
        #torch.cuda.empty_cache()
        return predictions


class LayoutDetection:
    def __init__(
        self,
        cfg_path,
        weights_path,
        device="cpu",
        batch_size=1,
        workers=1,
        threshold=0.75,
        label_map=["text", "title", "list", "table", "figure"],
    ):
        """Creates instance of LayoutDetection object

        Args:
            cfg_path (string): path to config file
            weights_path (string): path to pre-trained model weights
            device (string): device to be used by model, e.g., "cuda" (GPU) or "cpu"
            batch_size (int, optional): batch size. Defaults to 1
            workers (int, optional): number of workers. Defaults to 1
            threshold (float, optional): score threshold. Defaults to 0.75
            label_map (list, optional): label map used by the model. Defaults to example label map
        """

        opts = ["MODEL.WEIGHTS", weights_path]
        cfg = get_cfg()
        add_vit_config(cfg)
        cfg.merge_from_file(cfg_path)
        cfg.merge_from_list(opts)
        cfg.MODEL.DEVICE = device
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

        self.cfg = cfg
        self.label_map = label_map
        self.batch_size = batch_size
        self.workers = workers

    def get_predictor(self):
        """Gets BatchPredictor instance

        Returns:
            instance of BatchPredictor object
        """
        predictor = BatchPredictor(
            cfg=self.cfg,
            label_map=self.label_map,
            batch_size=self.batch_size,
            workers=self.workers,
        )
        return predictor

    def get_metadata(self):
        """Gets metadata for Detectron2 Visualizer

        Returns:
            instance of Metadata
        """
        metadata = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])
        metadata.set(thing_classes=self.label_map)
        return metadata
