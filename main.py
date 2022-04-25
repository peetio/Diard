import argparse
import os
import warnings

import cv2
import numpy as np
import pandas
import torch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from tqdm import tqdm

from ditod import add_vit_config
from modules.layout import (getJsonObjectsFromLayout, getOrderedLayout,
                            getPredictionData, setExtractedData)
from modules.sectioning import segmentSections
from modules.utils import exportAsJson, getImagesFromPdf, joinJsons, makeDirs


def main():

    parser = argparse.ArgumentParser(description="Detectron2 inference script")
    parser.add_argument(
        "--pdf_dir",
        help="Path to directory containing PDFs",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        help="Name of the output directory",
        type=str,
        required=True
    )

    parser.add_argument(
        "--sectioning",
        help="Segment the sections for each PDF",
        action="store_true" 
    )

    # Suppressing PyTorch & Detectron warnings 
    warnings.filterwarnings("ignore", category=UserWarning) # NOTE: comment out for debugging

    args = parser.parse_args()

    model_config_file = "./resources/model_configs/cascade/cascade_dit_large.yaml"
    opts = ["MODEL.WEIGHTS", "./resources/weights/publaynet_dit-l_cascade.pth"]

    # Step 1: instantiate config
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(model_config_file)
    
    # Step 2: add model weights URL to config
    cfg.merge_from_list(opts)
    
    # Step 3: set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device

    # set score threshold
    threshold = 0.75
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

    # Step 4: define model & classes
    predictor = DefaultPredictor(cfg)
    md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    md.set(thing_classes=["text","title","list","table","figure"])

    # Process multiple PDFs from specified directory
    pdfs = os.listdir(args.pdf_dir)

    for pdf_filename in pdfs:
        pdf_pages = getImagesFromPdf(args.pdf_dir+pdf_filename)
        filename = pdf_filename[:len(pdf_filename)-4]

        for i, page in tqdm(enumerate(pdf_pages), 
                desc=("Processing \'"+pdf_filename+"\'"),
                total=len(pdf_pages)):
            cv2_img = np.array(page) 
            cv2_img = cv2_img[:, :, ::-1].copy()    # convert RGB to BGR 

            # Step 5: run inference
            output = predictor(cv2_img)["instances"]

            # Conversion to Layout Parser objects
            boxes, classes, scores = getPredictionData(output)
            layout = getOrderedLayout(boxes, classes, scores)

            # Content extraction & export
            setExtractedData(cv2_img, layout, filename, i)

            # Create directory per PDF
            output_json_dir = args.output_dir + "jsons/" + filename
            output_viz_dir = args.output_dir + "visualizations/" + filename
            makeDirs([output_json_dir, output_json_dir+"/figures", output_viz_dir])
            
            json_filename = str(i)
            layout_json_obj  = getJsonObjectsFromLayout(layout, i+1)
            layout_json_filename = output_json_dir + '/' + json_filename + ".json"
            exportAsJson(layout_json_obj, layout_json_filename)

            # TODO: add table processing

            v = Visualizer(cv2_img[:, :, ::-1],
                            md,
                            scale=1.0,
                            instance_mode=ColorMode.SEGMENTATION)

            result = v.draw_instance_predictions(output.to("cpu"))
            result_image = result.get_image()[:, :, ::-1]

            # TODO: add original PDF filename to the output filename

            img_filename = str(i) + '.jpg'
            cv2.imwrite(output_viz_dir + '/' + img_filename, result_image)

            # Join JSON files (recomposing PDF)
            if(i == len(pdf_pages)-1):
                joinJsons(output_json_dir)

            torch.cuda.empty_cache() # Prevent CUDA out of memory

    if args.sectioning:
        jsons_dir_path = args.output_dir + "jsons/"
        segmentSections(jsons_dir_path, pdfs)


if __name__ == '__main__':
    main()

