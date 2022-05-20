#   Document class definition
import json
import logging
import os
import time
from io import StringIO
from pathlib import Path

import cv2
import langdetect
import layoutparser as lp
import numpy as np
import pandas as pd
import pycountry
from detectron2.utils.visualizer import ColorMode, Visualizer
from layoutparser.elements import Rectangle, TextBlock
from pdf2image import convert_from_path
from PIL import Image
from pytesseract import image_to_string
from tqdm import tqdm

from modules.exceptions import (
    DocumentFileFormatError,
    InputJsonStructureError,
    PageNumberError,
    UnsetAttributeError,
)
from modules.export import getLayoutHtml, getLayoutsHtml
from modules.sections import (
    getPageColumns,
    getTitleRatios,
    prioritizeLabels,
    sectionByChapterNums,
    sectionByRatio,
)


def overlapCheck(r1, r2):
    """Checks if two rectangles overlap
    Args:
        r1 (layoutparser.elements.Rectangle): rectangle with tl and br coordinates
        r2 (layoutparser.elements.Rectangle): rectangle with tl and br coordinates
    Returns:
        True if given rectangles overlap
    """
    #   rectangles next to each other?
    if r1.x_2 < r2.x_1 or r2.x_2 < r1.x_1:
        return False
    #   rectangles on top of each other?
    elif r1.y_1 > r2.y_2 or r2.y_1 > r1.y_2:
        return False
    else:
        return True


def filterOverlaps(rects, scores, classes):
    """Filters out overlapping rectangles

    Args:
        rects (list): layout parser Rectangle instances
        scores (list): prediction scores
        classes (list): prediction classes

    Returns:
        the given lists (rects, scores, classes) but without "big" overlappings
    """

    removal_idxs = []
    for r1_id, r1 in enumerate(rects):
        #   don't check for overlap with current rect
        rects2 = rects.copy()
        rects2[r1_id] = Rectangle(0, 0, 0, 0)  #   keep original length in list copy

        for r2_id, r2 in enumerate(rects2):
            overlap = overlapCheck(r1, r2)
            if overlap:
                r1_area = (r1.x_2 - r1.x_1) * (r1.y_2 - r1.y_1)
                r2_area = (r2.x_2 - r2.x_1) * (r2.y_2 - r2.y_1)

                #   calculate intersection area
                width = abs(min(r1.x_2, r2.x_2) - max(r1.x_1, r2.x_1))
                height = abs(max(r1.y_1, r2.y_1) - min(r1.y_2, r2.y_2))

                overlap_ratio = ((width * height) / min(r1_area, r2_area)) * 100
                if overlap_ratio > 60:
                    #   intersection covers more than 85% of smallest rect?
                    if r1_area > r2_area:
                        removal_idxs.append(r2_id)
                    else:
                        removal_idxs.append(r1_id)

    removal_idxs = np.unique(np.asarray(removal_idxs)).tolist()
    filtered_rects = [r for ri, r in enumerate(rects) if ri not in removal_idxs]
    filtered_scores = [s for si, s in enumerate(scores) if si not in removal_idxs]
    filtered_classes = [c for ci, c in enumerate(classes) if ci not in removal_idxs]

    return filtered_rects, filtered_scores, filtered_classes


class Document:
    @classmethod
    def __getLayoutJsonDims(cls, l, dims=0):
        """Recursive function to find number of dimensions (n of rows) in a list

        Args:
            l (list): list of which you want to know the number of rows
            dims (int): the current dimensionality or number of rows in the "list"

        Returns:
            the dimensionality (number of rows) in the "list"
        """

        if not type(l) == list:
            return dims

        return cls.__getLayoutJsonDims(l[0], (dims + 1))

    def __init__(
        self,
        source_path,
        output_path=None,
        predictor=None,
        metadata=None,
        table_predictor=None,
        lang="eng",
        lang_detect=False,
        langs=None,
        label_map=["text", "title", "list", "table", "figure"],
    ):
        """Creates instance of Document object

        Args:
            source_path (str): path to document
            output_path (str): path to output directory
            predictor (BatchPredictor, optional): configured predictor instance (see layoutdetection module)
            metadata (detectron2.data.Metadata, optional): dataset metadata
            lang (string, optional): language used in the document. Defaults to "eng" or English (ISO 639-3 format)
            lang_detect (bool, optional): if True language detection is used
            langs (list, optional): languages used in documents (ISO 639-3 format)
            label_map (list, optional): label map used by the model. Defaults to example label map
        """
        name = source_path.split("/")[-1]
        file_format = name.split(".")[-1]
        if file_format not in ["pdf"]:
            raise DocumentFileFormatError(name, file_format)

        self.name = ".".join(name.split(".")[:-1])
        self.file_format = file_format
        self.source_path = source_path

        if output_path is None:
            output_path = "output/" + self.name
        else:
            if output_path[-1] == "/":
                output_path += self.name
            else:
                output_path += "/" + self.name

        Path(output_path).mkdir(parents=True, exist_ok=True)
        self.output_path = output_path

        self.lang_detect = lang_detect
        self.lang = lang
        self.langs = langs
        self.metadata = metadata
        self.predictor = predictor
        self.table_predictor = table_predictor
        self.layouts = []
        self.label_map = label_map
        self.ordered = False

    def docToImages(self):
        """Converts each page of a document to images"""
        pil_imgs = convert_from_path(self.source_path)
        self.images = [
            cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB) for img in pil_imgs
        ]

    def __getTextFromImage(self, block, img):
        """Extracts text from image

        Args:
            block (layoutparser.elements.TextBlock): TextBlock obj containing document object data
            img (numpy.ndarray): image of document page

        Return:
            the extracted text as a string
        """
        #   TODO: maybe you can add an option to clean the OCR
        snippet = block.pad(left=5, right=5, top=5, bottom=5).crop_image(img)
        text = image_to_string(snippet, lang=self.lang)
        return text

    def __setExtractedData(self, block, img, page, idx):
        """Extracts and sets content of document object (block)

        Args:
            b_type (string): block type (e.g., figure, text, list)
            block (layoutparser.elements.TextBlock): TextBlock obj containing document object data
            img (numpy.ndarray): image of document page
            page (int): page number the block occurs on
            idx (int): block id for re-identification of images on same page
        """

        figure_dir = self.output_path + "/figures/"
        Path(figure_dir).mkdir(parents=True, exist_ok=True)
        b_type = block.type.lower()

        if b_type in ["text", "title", "list"]:
            text = self.__getTextFromImage(block, img)

            block.set(text=text, inplace=True)

        elif b_type in ["figure", "table"]:
            snippet = block.crop_image(img)
            img_name = str(page) + "-" + str(idx) + "_" + b_type + ".jpg"
            figure_path = figure_dir + img_name
            save_path = "../figures/" + img_name

            cv2.imwrite(figure_path, snippet)
            block.set(text=save_path, inplace=True)

            if b_type == "table" and self.table_predictor:
                snippet = cv2.cvtColor(snippet, cv2.COLOR_BGR2RGB)
                snippet = Image.fromarray(snippet)
                df = self.table_predictor.get_table_data(
                    snippet, lang=self.lang, debug=False, threshold=0.7
                )
                #   NOTE: when processing tables in the HTML you have to check whether you are
                #   working with the image path or with DataFrame
                #   NOTE: the DataFrame can be converted back from the string to a DF
                #   use: https://stackoverflow.com/questions/22604564/create-pandas-dataframe-from-a-string

                df_csv = df.to_csv(index=False)
                block.set(text=df_csv, inplace=True)

        else:
            logging.warning(f"Block of type '{b_type}' not supported.")

    def __detectLanguage(self, block, img):
        """Detects and sets language if detection is successful

        Args:
            block (layoutparser.elements.TextBlock): TextBlock obj containing document object data
            img (numpy.ndarray): image of document page
        """
        text = self.__getTextFromImage(block, img)
        if len(text.split(" ")) > 5:
            lang = langdetect.detect(text.strip())
            lang = pycountry.languages.get(alpha_2=lang)
            in_langs = self.langs and lang.alpha_3 in self.langs
            if in_langs or not self.langs:
                logging.info(
                    f"Language detection successful! Language is now set to {lang.name} ({lang.alpha_3})."
                )
                self.lang = lang.alpha_3
            else:
                logging.warning(
                    f"Language detection might have failed. {lang.name} ({lang.alpha_3}) not in list of languages ({self.langs})."
                )

            self.lang_detect = False

    def extractLayouts(self, segment_sections=False, visualize=False):
        """Extracts and sets layout from document images

        Args:
            segment_sections (bool): if True sections are segmented
            visualize (bool): if True detection visualizations are saved
        """

        if self.table_predictor:
            logging.info(
                "Processing will take longer if table extraction is enabled. You can disable it by setting the table_predictor to None"
            )

        if None in [self.predictor, self.metadata, self.images]:
            raise UnsetAttributeError(
                "extractLayout()", ["predictor", "metadata", "images"]
            )

        if len(self.layouts) > 0:
            self.layouts = []

        logging.info(f"Processing '{self.name}', starting layout detection now.")
        st = time.time()
        predictions = self.predictor(self.images)
        et = round((time.time() - st), 3)
        logging.info(f"Layout detection took {et}s.")
        logging.info(
            "Extracting content of " + str(len(predictions)) + " document objects."
        )
        for page, predicts in tqdm(
            enumerate(predictions), desc=(), total=len(predictions)
        ):

            boxes = predicts.pred_boxes if predicts.has("pred_boxes") else None
            np_boxes = boxes.tensor.cpu().numpy()
            classes = (
                predicts.pred_classes.tolist() if predicts.has("pred_classes") else None
            )
            scores = predicts.scores if predicts.has("scores") else None

            # rect format: x1, y1, x2, y2
            rects = [
                lp.Rectangle(int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                for b in np_boxes
            ]

            filtered_rects, filtered_scores, filtered_classes = filterOverlaps(
                rects=rects, scores=scores, classes=classes
            )
            img = self.images[page]
            blocks = []
            for j in range(len(filtered_rects)):
                block = lp.TextBlock(
                    block=filtered_rects[j],
                    type=self.label_map[filtered_classes[j]],
                    score=filtered_scores[j],
                )
                blocks.append(block)

                #   detect language used in document
                is_text = block.type.lower() == "text"
                if self.lang_detect and is_text:
                    self.__detectLanguage(block, img)

            for j, b in enumerate(blocks):
                self.__setExtractedData(b, img, page, j)

            self.layouts.append(lp.Layout(blocks=blocks))

            if visualize:
                self.__visualizePredictions(predicts, img, page)

        if segment_sections:
            if not self.ordered:
                logging.info("Ordering layout for section segmentation.")
                self.orderLayouts()

            self.__segmentSections()

    def orderLayouts(self):
        """Orders each page's layout based object bounding boxes"""

        if not self.ordered:
            width = self.images[0].shape[1]

            for page, layout in enumerate(self.layouts):
                #   layout support up to 3 columns
                #   split layout based on block center (x-axis)
                cols = getPageColumns(layout)
                blocks = layout._blocks
                if cols == 1:
                    left_blocks = sorted(blocks, key=lambda b: b.block.y_1)
                if cols == 2:
                    #   get blocks and filter per column
                    left_blocks = list(
                        filter(
                            lambda b: b.block.x_1 + ((b.block.x_2 - b.block.x_1) / 2)
                            < (width / 2),
                            blocks,
                        )
                    )
                    right_blocks = list(
                        filter(
                            lambda b: b.block.x_1 + ((b.block.x_2 - b.block.x_1) / 2)
                            >= (width / 2),
                            blocks,
                        )
                    )

                    #   filter on y-axis page location
                    left_blocks = sorted(left_blocks, key=lambda b: b.block.y_1)
                    right_blocks = sorted(right_blocks, key=lambda b: b.block.y_1)

                    #   recompose layout
                    left_blocks.extend(right_blocks)

                elif cols == 3:
                    cols = width / 3
                    break1, break2 = cols, cols * 2

                    #   get blocks and filter per column
                    left_blocks = list(
                        filter(
                            lambda b: b.block.x_1 + ((b.block.x_2 - b.block.x_1) / 2)
                            <= break1,
                            blocks,
                        )
                    )
                    center_blocks = list(
                        filter(
                            lambda b: break1
                            < b.block.x_1 + ((b.block.x_2 - b.block.x_1) / 2)
                            < break2,
                            blocks,
                        )
                    )
                    right_blocks = list(
                        filter(
                            lambda b: break2
                            <= b.block.x_1 + ((b.block.x_2 - b.block.x_1) / 2),
                            blocks,
                        )
                    )

                    #   filter on y-axis page location
                    left_blocks = sorted(left_blocks, key=lambda b: b.block.y_1)
                    center_blocks = sorted(center_blocks, key=lambda b: b.block.y_1)
                    right_blocks = sorted(right_blocks, key=lambda b: b.block.y_1)

                    #   recompose layout
                    center_blocks.extend(right_blocks)
                    left_blocks.extend(center_blocks)

                self.layouts[page] = lp.Layout(
                    blocks=[b.set(id=idx) for idx, b in enumerate(left_blocks)]
                )

                self.ordered = True

        else:
            logging.info("Not re-ordering layout.")

    def __visualizePredictions(self, predicts, img, index):
        """Saves prediction visualizations

        Args:
            predicts (detectron2.structures.instances.Instances): inference output data
            img (numpy.ndarray): original document image used for inference
            index (int): index for file naming, usually the page number
        """

        v = Visualizer(
            img[:, :, ::-1],
            self.metadata,
            scale=1.0,
            instance_mode=ColorMode.SEGMENTATION,
        )

        visual = v.draw_instance_predictions(predicts.to("cpu"))
        vis_img = visual.get_image()[:, :, ::-1]

        visual_dir = self.output_path + "/visuals/"
        Path(visual_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(visual_dir + str(index) + ".jpg", vis_img)

    def pageCheck(self, page):
        """Checks if the page is within range

        Args:
            page (int): page number to check
        """

        pages = len(self.layouts)
        if not 0 <= page < pages:
            raise PageNumberError(page, pages)

    def __getLayoutJson(self, page):
        """Gets JSON object representing a single document page layout

        Args:
            page (int): page number of document

        Returns:
            JSON object representing a document single page layout
        """

        layout_json = []
        for block in self.layouts[page]:
            text = block.text
            if block.type.lower() == "table" and not text.endswith("jpg"):
                table_str = StringIO(text)
                df = pd.read_csv(table_str)
                text = df.to_dict()
                print("type:", type(text))
                print("content:", text)

            el = {
                "id": block.id,
                "type": block.type,
                "content": text,
                "box": block.coordinates,
                "page": page,
            }
            try:
                if block.section:
                    el["section"] = block.section
            except AttributeError:
                pass
            layout_json.append(el)

        return layout_json

    def __getLayoutsJson(self):
        """Gets JSON object representing the whole document layout

        Returns:
            JSON objects (list of dictionaries) containing whole document layout
        """

        layouts_json = []
        for page in range(len(self.layouts)):
            layouts_json.append(self.__getLayoutJson(page))

        return layouts_json

    def saveLayoutAsJson(self, page):
        """Saves JSON representation of single document page layout

        Args:
            page (int): page number of document
        """

        self.pageCheck(page)

        json_dir = self.output_path + "/jsons/"
        Path(json_dir).mkdir(parents=True, exist_ok=True)
        json_path = json_dir + str(page) + ".json"

        layout_json = self.__getLayoutJson(page)
        with open(json_path, "w") as f:
            f.write(json.dumps(layout_json))

    def saveLayoutsAsJson(self):
        """Saves JSON representation of whole document layout"""

        pages = len(self.layouts)
        if pages < 1:
            logging.warning(
                "You have no document layouts to export. Output directory is created regardless."
            )
        json_dir = self.output_path + "/jsons/"
        Path(json_dir).mkdir(parents=True, exist_ok=True)

        for page in range(pages):
            self.saveLayoutAsJson(page)

        json_path = json_dir + self.name + ".json"
        layouts_json = self.__getLayoutsJson()
        with open(json_path, "w") as f:
            f.write(json.dumps(layouts_json))

    def saveLayoutAsHtml(self, page):
        """Saves JSON representation of single document page layout

        Args:
            page (int): page number of document
        """

        self.pageCheck(page)
        html_dir = self.output_path + "/htmls/"
        Path(html_dir).mkdir(parents=True, exist_ok=True)
        html_path = html_dir + str(page) + ".html"
        layout_html = getLayoutHtml(self.layouts[page])

        html_path = html_dir + str(page) + ".html"
        with open(html_path, "w") as f:
            f.write(layout_html)

    def saveLayoutsAsHtml(self):
        """Saves HTML representation of whole document layout"""

        html_dir = self.output_path + "/htmls/"
        Path(html_dir).mkdir(parents=True, exist_ok=True)

        for page in range(len(self.layouts)):
            self.saveLayoutAsHtml(page)

        html_path = html_dir + self.name + ".html"
        layouts_html = getLayoutsHtml(self.layouts)

        with open(html_path, "w") as f:
            f.write(layouts_html)

    def __jsonToLayout(self, layout_json):
        """Gets a Layout object from a JSON layout representation of a single page

        Args:
            layout_json (list): JSON layout representation of single page
        """

        blocks = []
        for b in layout_json:
            x1, y1, x2, y2 = b["box"]
            rect = Rectangle(x1, y1, x2, y2)
            block = TextBlock(
                block=rect,
                text=b["content"],
                id=b["id"],
                type=self.label_map.index(b["type"]),
            )
            blocks.append(block)

        self.layouts.append(lp.Layout(blocks=blocks))

    def loadLayoutFromJson(self, filename):
        """Loads layouts from JSON file

        Args:
            filename (str): JSON filename
        """

        if not filename.split(".")[-1] == "json":
            filename += ".json"
        json_path = self.output_path + "/jsons/" + filename
        with open(json_path, "r") as f:
            layout_json = json.load(f)

        #   single or joined layouts file check based on list dimensionality
        dims = self.__getLayoutJsonDims(l=layout_json)
        if dims == 1:
            self.__jsonToLayout(layout_json)
        elif dims == 2:
            for layout in layout_json:
                self.__jsonToLayout(layout)
        else:
            raise InputJsonStructureError(filename)

    def __setSections(self, labels):
        """Sets section to which each document object belongs

        Args:
            labels (list): a list where each label corresponds to a title object
        """

        section = 0
        title_id = 0
        for layout in self.layouts:
            for b in layout:
                #   Check if its section heading
                if b.type.lower() == "title":
                    if labels[title_id] == "heading":
                        section += 1
                    title_id += 1
                b.section = section

    def __segmentSections(self):
        """Segments sections based on numbering and natural breaks"""
        cn_labels = sectionByChapterNums(self.layouts)
        ratios = getTitleRatios(self.layouts)
        r_labels = sectionByRatio(ratios, self.name)
        labels = prioritizeLabels(self.layouts, cn_labels, r_labels)
        self.__setSections(labels)
