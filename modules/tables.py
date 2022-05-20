import json
import logging
import os
import random
import sys
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from PIL import Image
from pytesseract import image_to_string
from torch.utils.data import DataLoader

sys.path.append("detr")
import datasets.transforms as R
from models import build_model


def cells_to_dataframe(cells):
    """Gets pandas DataFrame from cells

    Args:
        cells (list): table cells with bounding box coordinates

    Returns:
        df (pandas.DataFrame): extracted table data
    """

    d = {}
    for cell in cells:
        if cell["column"] not in d.keys():
            d[cell["column"]] = []
        d[cell["column"]].append(cell["content"])
    df = pd.DataFrame(data=d)
    return df


def set_cell_text(cells, image, lang="eng", clean=False):
    """Extracts and sets the text from each table cell

    Args:
        cells (list): table cells with bounding box coordinates
        image (PIL.Image): Pillow image
        lang (string, optional): language used in the document. Defaults to "eng" or English (ISO 639-3 format)
        clean (bool, optional): if True minor text cleaning is applied. Defaults to False
    """
    for cell in cells:
        #   crop & pad image
        xmin, ymin, xmax, ymax = cell["bbox"]
        roi = image.crop((xmin, ymin, xmax, ymax))
        roi = add_padding(roi, 30)

        #   extract text from region of interest
        text = image_to_string(
            roi, lang=lang
        )  #    TODO: find the optimal PSM (e.g., 'config="--psm 6"')
        if clean:
            text.strip()
            text = text.replace("\n", " ")
        cell["content"] = text


def get_rows_and_columns(objs):
    """Gets the rows and columns predictions

    Args:
        objs (list): list of dictionaries representing predictions

    Returns:
        list of objs containing rows and cols respectively
    """
    cols = [obj for obj in objs if obj["label"] == "table column"]
    rows = [obj for obj in objs if obj["label"] == "table row"]
    #   sort cols bottom right x coordinate
    cols.sort(key=lambda col: col["bbox"][2])
    #   sort rows bottom right y coordinate
    rows.sort(key=lambda row: row["bbox"][3])

    return rows, cols


def get_cells(objs):
    """Gets the cells based on row and column predictions (objs)

    Args:
        objs (list): list of dictionaries representing predictions

    Returns:
        table cells
    """
    rows, cols = get_rows_and_columns(objs)

    cells = []
    for i, col in enumerate(cols):
        c_xmin, c_ymin, c_xmax, c_ymax = col["bbox"]
        for j, row in enumerate(rows):
            r_xmin, r_ymin, r_xmax, r_ymax = row["bbox"]
            xmin, ymin = max(r_xmin, c_xmin), max(r_ymin, c_ymin)
            xmax, ymax = min(r_xmax, c_xmax), min(r_ymax, c_ymax)
            cell = {"column": i, "row": j, "bbox": [xmin, ymin, xmax, ymax]}
            cells.append(cell)

    return cells


#   Source file: postprocess.py
def structure_table(objs, table_bbox):
    """Aligns rows and columns in the following scenario's between objects
            of the same type

       Scenario 1: border is on same line []][]
            then: keep xmin
       Scenario 2: border 1 is smaller than border 2, there is gap [] []
       Scenario 3: border 1 is bigger than border 2, there is an overlap [[]]
           then: make xmin of current cell the xmax of the previous bbox

    Args:
        objs (list): list of dictionaries representing predictions
        table_bbox (list): xmin, ymin, xmax, ymax coordinates of table bbox

    Returns:
        structured table objects
    """
    rows, cols = get_rows_and_columns(objs)

    #   initial values are top and most left coordinates
    p_xmin, p_ymin, p_xmax, p_ymax = table_bbox
    p_ymax = p_ymin
    p_xmax = p_xmin
    for row in rows:
        xmin, ymin, xmax, ymax = row["bbox"]
        if not p_ymax == ymin:
            ymin = p_ymax
        row["bbox"] = xmin, ymin, xmax, ymax
        p_ymax = ymax
    #   column bottom borders has to overlap with last row's bottom border
    bottom_y = rows[-1]["bbox"][3]
    for col in cols:
        xmin, ymin, xmax, ymax = col["bbox"]

        if not p_xmax == xmin:
            xmin = p_xmax
        col["bbox"] = xmin, ymin, xmax, ymax
        p_xmax = xmax

    return objs


def filter_cols_and_rows(objs):
    """Filters anything else than columns and rows

    Args:
        objs (list): list of dictionaries representing predictions

    Returns:
        list of objects containing only predictions of type 'table column' and 'table row'
    """
    objs = [obj for obj in objs if obj["label"] in ["table column", "table row"]]
    return objs


def border_align(objs, table_bbox):
    """Aligns row and column objects to table border

    Args:
        objs (list): list of dictionaries representing predictions
        table_bbox (list): xmin, ymin, xmax, ymax coordinates of table bbox

    Returns:
        aligned table objects (predictions)
    """
    for obj in objs:
        bbox = obj["bbox"]
        if obj["label"] == "table row":
            bbox[0] = table_bbox[0]
            bbox[2] = table_bbox[2]
        elif obj["label"] == "table column":
            bbox[1] = table_bbox[1]
            bbox[3] = table_bbox[3]
        obj["bbox"] = bbox

    return objs


def predictions_to_objects(predictions, class_map, threshold=0.7):
    """Converts predictions to individual objects

    Args:
        predictions (dict): score, bboxes, and label for each prediction
        threshold(float, optional): score threshold, value between .99 and .0. Defaults to 0.7 (70%)
        class_map (dictionary): class or label map with index as key and class name as value

    Returns:
        list of objects in the following format:

            {
                "score" : prediction score,
                "label" : class name (not label index)
                "bbox" : bounding box coordinates
            }
    """
    objs = []
    labels = predictions["labels"].tolist()
    scores = enumerate(predictions["scores"].tolist())
    for idx, score in scores:
        if score > threshold:
            label = labels[idx]
            bbox = predictions["boxes"][idx].tolist()
            obj = {
                "score": score,
                "label": class_map[label],
                "bbox": list(map(int, bbox)),
            }
            objs.append(obj)
    return objs


def visualize_structure(image, objs):
    """Gets visualization of predictions

    Args:
        image (PIL.Image): Pillow image
        objs (list): list of dictionaries representing predictions

    Returns:
        original image with visualization of predictions
    """
    image = np.array(image)
    for obj in objs:
        xmin, ymin, xmax, ymax = obj["bbox"]
        cv2.rectangle(
                image, 
                (xmin, ymin), 
                (xmax, ymax), 
                (245, 105, 66), 
                2
                )
    image = image[:, :, ::-1].copy()
    return image


#   Source file: postprocess.py (align_columns & align_rows)


def add_padding(image, padding=50):
    """Adds padding to an image

    Args:
        image (PIL.Image): Pillow image
        padding (int): number of pixels to use for padding

    Returns:
        image with padding
    """

    w, h = image.size
    new_w = w + (padding*2)
    new_h = h + (padding*2)
    result = Image.new(image.mode, (new_w, new_h), (255, 255, 255))
    result.paste(image, (padding, padding))
    return result


def load_args(json_path):
    """Loads arguments from JSON file

    Args:
        json_path (string): path to json file containing default args
    Returns:
        data from the JSON file (default arguments)
    """
    data = None
    with open(json_path) as f:
        data = json.load(f)
    return data


def get_class_map(key="name"):
    """Gets label or class map

    Args:
        key (string): indicates what to use as dict key. Options are "name" and "index". Defaults to "name"

    Returns:
        class map in the form of a dictionary
    """
    assert key in ["index", "name"]
    if key == "name":
        return {
            "table": 0,
            "table column": 1,
            "table row": 2,
            "table column header": 3,
            "table projected row header": 4,
            "table spanning cell": 5,
            "no object": 6,
        }
    else:
        return {
            0: "table",
            1: "table column",
            2: "table row",
            3: "table column header",
            4: "table projected row header",
            5: "table spanning cell",
            6: "no object",
        }


# Source file: main.py
def get_model(args, device):
    """Loads DETR model on to the device specified.

    Args:
        args (Args): script arguments containing model data
        device (string): device to be used by model, e.g., "cuda" (GPU) or "cpu"

    Returns:
        the model, criterion, and postprocessors
            (see official table-transformer repository for more details)
    """
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    if args.model_load_path:
        logging.info(
            "Loading pre-trained table structure recognition model from checkpoint"
        )
        loaded_state_dict = torch.load(args.model_load_path, map_location=device)
        model_state_dict = model.state_dict()
        pretrained_dict = {
            k: v
            for k, v in loaded_state_dict.items()
            if k in model_state_dict and model_state_dict[k].shape == v.shape
        }
        model_state_dict.update(pretrained_dict)
        model.load_state_dict(model_state_dict, strict=True)
    return model, criterion, postprocessors


class TableExtractor:
    def __init__(self, weight_path):
        """Creates instance of TableExtractor object

        Args:
            weight_path (string): path to pre-trained weights
        """
        args = load_args(
            "./resources/structure_config.json"
        )  #   TODO: we might need to change ../ to ./
        args["model_load_path"] = weight_path
        args = type("Args", (object,), args)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"  #   TODO: remove this
        self.model, criterion, self.postprocessors = get_model(args, self.device)
        self.model.eval()

        self.normalize = R.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def predict(self, image):
        """Runs inference on given image

        Args:
            image (PIL.Image): image containing only table

        Returns:
            dictionary containing score, bboxes, and label of each prediction
        """

        w, h = image.size

        img_tensor = self.normalize(F.to_tensor(image))[0]
        img_tensor = torch.unsqueeze(img_tensor, 0).to(self.device)

        #   inference
        outputs = None
        with torch.no_grad():
            outputs = self.model(img_tensor)

        image_size = torch.unsqueeze(torch.as_tensor([int(h), int(w)]), 0).to(
            self.device
        )
        results = self.postprocessors["bbox"](outputs, image_size)[0]

        return results

    def get_table_data(self, image, lang="eng", debug=True, threshold=0.7):
        """Gets the table data as a pandas DataFrame

        Args:
            image (PIL.Image): image containing only table
            lang (string, optional): language used in the document. Defaults to "eng" or English (ISO 639-3 format)
            debug (bool): if True visualization of predictions is saved locally
            threshold(float): score threshold, value between .99 and .0

        Returns:
            df (pandas.DataFrame): extracted table data
        """

        #   pre-trained model uses "padded" tables
        padding = 50
        image = add_padding(image)
        results = self.predict(image)
        #   conversion to objects w/ score threshold
        objs = predictions_to_objects(results, get_class_map(key="index"), threshold)

        #   align columns and rows to table border
        w, h = image.size
        #   assuming we work with cropped out table
        xmin, ymin, xman, ymax = padding, padding, w - padding, h - padding 
        table_bbox = [xmin, ymin, xman, ymax]
        objs = border_align(objs, table_bbox)

        #   fix overlapping and align objects
        objs = structure_table(objs, table_bbox)

        #   keep only the columns and rows
        objs = filter_cols_and_rows(objs)

        #   get cells based on cols and rows
        cells = get_cells(objs)
        set_cell_text(cells, image, lang, clean=True)
        df = cells_to_dataframe(cells)

        if debug:
            visualization = visualize_structure(image, objs)
            out_path = "./visualization.jpg"
            cv2.imwrite(out_path, visualization)
            logging.info(
                f"[Debug] Table extraction visualization can be found at '{out_path}'."
            )

        return df
