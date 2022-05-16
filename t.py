import cv2
import os
import logging
import warnings
import numpy as np

from modules.document import Document
from modules.utils import runBatchPredictor
from modules.exceptions import DocumentFileFormatError

from pdf2image import convert_from_path

def docToImages(source_path):
    """Converts each page of a document to images"""
    pil_imgs = convert_from_path(source_path)
    images = [cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB) for img in pil_imgs]
    return images

def main():
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s: %(message)s", level=logging.NOTSET
    )
    logging.disable(logging.DEBUG)

    #   suppressing PyTorch & Detectron warnings
    warnings.filterwarnings(
        "ignore", category=UserWarning
    )  # NOTE: comment out for debugging

    #   initialize model & get images from PDF
    config_path = "./resources/model_configs/cascade/cascade_dit_large.yaml"
    weights_path = "./resources/weights/publaynet_dit-l_cascade.pth"
    pdf_path = "./resources/pdfs/example.pdf"
    images = docToImages(pdf_path)

    #   batch inference
    runBatchPredictor(config_path, weights_path, images=images, threshold=0.70)

if __name__ == "__main__":
    main()
