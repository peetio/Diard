#   Document class definition
import os
import cv2
import logging
import json
import layoutparser as lp
import numpy as np
from pytesseract import image_to_string

from tqdm import tqdm
from pathlib import Path
from pdf2image import convert_from_path
from detectron2.utils.visualizer import ColorMode, Visualizer
from layoutparser.elements import TextBlock, Rectangle

from modules.exceptions import DocumentFileFormatError, UnsetAttributeError, PageNumberError, InputJsonStructureError



class Document():
    @staticmethod
    def setExtractedData(b_type, block, img):
        """Extracts and set content of document object (block)
        """

        if b_type in ['text', 'title', 'list']:
            snippet = (block
                        .pad(left=5, right=5, top=5, bottom=5)
                        .crop_image(img))

            text = image_to_string(snippet, lang='deu')
            block.set(text=text, inplace=True)

        elif b_type in ['table', 'figure']:
            #   TODO: add table support
            if b_type == 'table':
                logging.warning("Tables are currently not supported and are therefore processed like figures.")
            snippet = block.crop_image(img)
            img_name = page + ".jpg"
            figure_path = self.output_path + "/figures/" + img_name
            save_path = os.path.abspath(figure_path)    #   NOTE: is using the abs path correct?
            cv2.imwrite(figure_path, snippet)
            block.set(text=save_path, inplace=True)
        else:
            logging.warning(f'Block of type \'{b_type}\' not supported.')



    @classmethod
    def getLayoutJsonDims(cls, l, dims=0):
        """Recursive function to find number of dimensions (n of rows) in a list

        Args:
            l (list): list of which you want to know the number of rows
            dims (int): the current dimensionality or number of rows in the "list"

        Returns:
            the dimensionality or number of rows in the "list"
        """

        if not type(l) == list:
            return dims

        return cls.getLayoutJsonDims(l[0], (dims+1))


 
    def __init__(self, source_path, output_path=None, predictor=None, metadata=None,
                    label_map=["text","title","list","table","figure"]):
        """Creates instance of Document object
        
        Args:
            source_path (str): path to document
            output_path (str): path to output directory
            predictor (detectron2.engine.defaults.DefaultPredictor, optional): configured default predictor instance
            metadata (detectron2.data.Metadata, optional): dataset metadata
            label_map (list): label map used by the model. Defaults to example label map
        """

        name = source_path.split('/')[-1]
        file_format = name.split('.')[-1]
        if file_format not in ['pdf', 'docx']:  #   TODO: Apr 26 - test if other variations of pdf and docx can be used with image conversion
            raise DocumentFileFormatError(name, file_format)

        self.name = '.'.join(name.split('.')[:-1]) 
        self.file_format = file_format
        self.source_path = source_path 

        if output_path is None:
            output_path = 'output/'+self.name
        else:
            if output_path[-1] == '/':
                output_path+=self.name
            else: output_path+='/'+self.name

        Path(output_path).mkdir(parents=True, exist_ok=True)
        self.output_path = output_path

        self.metadata = metadata
        self.predictor = predictor
        self.layouts = []   #   TODO: 26 Apr - do we allow initialization of layouts, or do we do this directly through a method where one can load the data from a JSON file for example?
        self.label_map = label_map
    


    def docToImages(self):
        """Converts each page of a document to images"""
        #   TODO: add word document support
        pil_imgs = convert_from_path(self.source_path) 
        self.images = [np.asarray(img) for img in pil_imgs] #   TODO: make sure they are in the right format



    def extractLayouts(self, segment_sections=False, visualize=False):
        """Extracts and sets layout from document images
        
        Args:
            segment_sections (bool): if True sections are segmented
            visualize (bool): if True detection visualizations are saved to self.output_path
        """

        if None in [self.predictor, self.metadata, self.images]:
            raise UnsetAttributeError("extractLayout()", ["predictor", "metadata", "images"])

        if len(self.layouts) > 0:
            self.layouts = []

        for page, img in tqdm(enumerate(self.images),
                            desc=("Processing \'"+self.name+"\'"),
                            total=len(self.images)):

            predicts = self.predictor(img)["instances"]
            boxes = predicts.pred_boxes if predicts.has("pred_boxes") else None
            np_boxes = boxes.tensor.cpu().numpy()
            classes = predicts.pred_classes.tolist() if predicts.has("pred_classes") else None
            scores = predicts.scores if predicts.has("scores") else None

            # rect format: x1, y1, x2, y2
            rects = [lp.Rectangle(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in np_boxes]
            blocks= []
            for j in range(len(rects)):
                block = lp.TextBlock(block=rects[j], 
                                            type=self.label_map[classes[j]], 
                                            score=scores[j])

                b_type = block.type.lower()
                self.setExtractedData(b_type, block, img)    
                blocks.append(block)

            self.layouts.append(lp.Layout(blocks=blocks))

            if visualize:
                self.visualizePredictions(predicts, img, page)

            if segment_sections:
                #   TODO: 26 Apr - add section segmentation (in new function in other module)
                pass



    #   TODO: maybe add function to order only one layout if this is useful
    def orderLayouts(self):
        """Orders each page's layout based object bounding boxes"""

        #   TODO: add support for Manhattan (page split), parameter of automatic detection of layout type?
        for page, layout in enumerate(self.layouts):
            layout = sorted(layout, key=lambda b:b.block.y_1)
            self.layouts[page] = lp.Layout([b.set(id = idx) for idx, b in enumerate(layout)])
        


    def visualizePredictions(self, predicts, img, index): 
        """Saves prediction visualizations

        Args:
            predicts (detectron2.structures.instances.Instances): inference output data
            img (numpy.ndarray): original document image used for inference
            index (int): index for file naming, usually the page number
        """

        v = Visualizer(img[:, :, ::-1],
                self.metadata,
                scale=1.0,
                instance_mode=ColorMode.SEGMENTATION)

        visual = v.draw_instance_predictions(predicts.to("cpu"))
        vis_img = visual.get_image()[:, :, ::-1]
        
        visual_dir = self.output_path+"/visuals/"
        Path(visual_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(visual_dir + str(index)+'.jpg', vis_img)



    def getLayoutJson(self, page):
        """Gets JSON object representing a single document page layout

        Args:
            page (int): page number of document

        Returns:
            JSON object representing a document page layout
        """

        layout_json = []
        for block in self.layouts[page]:
            layout_json.append({"id" : block.id,
                                "type": block.type,
                                "content" : block.text,
                                "box": block.coordinates,
                                "page": page})

        return layout_json 


        
    def getLayoutsJson(self):
        """Gets JSON object representing the whole document layout"""
        
        layouts_json = []
        for page in range(len(self.layouts)):
            layouts_json.append(self.getLayoutJson(page))
        
        return layouts_json



    def saveLayoutAsJson(self, page):
        """Saves JSON representation of single document page layout

        Args:
            page (int): page number of document
        """

        #   TODO: check if page number is present
        pages = len(self.layouts)
        if not 0 <= page <= pages-1: 
            raise PageNumberError(page, pages)

        json_dir = self.output_path+"/jsons/"
        Path(json_dir).mkdir(parents=True, exist_ok=True)
        json_path = json_dir+str(page)+".json"

        layout_json = self.getLayoutJson(page)
        with open(json_path, 'w') as f:
            f.write(json.dumps(layout_json))



    def saveLayoutsAsJson(self):
        """Saves JSON representation of whole document layout"""

        json_dir = self.output_path+"/jsons/"
        Path(json_dir).mkdir(parents=True, exist_ok=True)

        for page in range(len(self.layouts)):
            self.saveLayoutAsJson(page)
        
        json_path = json_dir+self.name+".json"
        layouts_json = self.getLayoutsJson()
        with open(json_path, 'w') as f:
            f.write(json.dumps(layouts_json))


    def jsonToLayout(self, layout_json):
        """Gets a Layout object from a JSON layout representation of a single page

        Args:
            layout_json (list): JSON layout representation of single page
        """
        blocks = []
        for b in layout_json:
            x1, y1, x2, y2 = b['box']
            rect = Rectangle(x1, y1, x2, y2)
            block = TextBlock(block=rect,
                            text=b['content'],
                            id=b['id'],
                            type=self.label_map.index(b['type']))
            blocks.append(block)
        #   TODO: create layout object from the blocks
        self.layouts.append(lp.Layout(blocks=blocks))



    def loadLayoutFromJson(self, filename):
        """Loads layouts from JSON file
        
        Args:
            filename (str): JSON filename
        """

        if not filename.split('.')[-1] == "json":
            filename+=".json"
        json_path = self.output_path+"/jsons/"+filename
        with open(json_path, 'r') as f:
            layout_json = json.load(f)

        #   TODO: here you have to check if the json file that is being loaded is a single JSON file, or a JSON file containing a whole document
        #if len(layout_json) > 1:

        dims = self.getLayoutJsonDims(l=layout_json)

        if dims == 1:
            self.jsonToLayout(layout_json)
        elif dims == 2:
            #   TODO: add support for whole document layout JSONs
            for layout in layout_json:
                self.jsonToLayout(layout)
        else:
            raise InputJsonStructureError(filename)
    #   TODO: add function to load layout objects from JSON files


    #   TODO: add advanced HTML exportation (w/ CSS file this time for additional information and styling)

    #   TODO: add section segmentation method
    #           Note that you add a parameter to the saveLayoutsAsJson(self, page, +sections=Treu) where if sections True the output json contains of a list of lists. This is the same when exporting the joined json file. But when exporting the sections, the page numbering is not used to split the different objects. In other words, instead of having a list of pages which each contain a list of objects belonging to that page, we will then have a list of sections which each contain a list of objects belonging to that section. But for this you first need to implement section clustering. Note that the section clustering should just set a new attribute (section=section_number) and not change the structure of the self.layouts attribute! Handling which object belongs to which section has to be done when exporting the data.
