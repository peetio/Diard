#   Document class definition
import os
import cv2
import logging
import layoutparser as lp
import numpy as np
from pytesseract import image_to_string

from tqdm import tqdm
from pathlib import Path
from pdf2image import convert_from_path
from detectron2.utils.visualizer import ColorMode, Visualizer

from modules.exceptions import DocumentFileFormatError, UnsetAttributeError



class Document():
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
        self.source_path= source_path 

        if output_path is None:
            output_path = 'output/'+self.name
        else:
            if output_path[-1] == '/':
                output_path+=self.name
            else: output_path+='/'+self.name

        Path(output_path).mkdir(parents=True, exist_ok=True)
        #if not os.path.isdir(output_path):
        #    os.makedirs(output_path)
        self.output_path = output_path

        self.metadata = metadata
        self.predictor = predictor
        self.layouts = []   #   TODO: 26 Apr - do we allow initialization of layouts, or do we do this directly through a method where one can load the data from a JSON file for example?
        self.label_map = label_map
    


    def docToImages(self):
        """Converts each page of a document to images"""
        pil_imgs = convert_from_path(self.source_path) 
        self.images = [np.asarray(img) for img in pil_imgs] #   TODO: make sure they are in the right format



    def extractLayouts(self, segment_sections=False, visualize=False):
        """Extracts and sets layout from document images
        
        Args:
            segment_sections (bool): if True sections are segmented
            visualize (bool): if True detection visualizations are saved to self.output_path
        """

        ocr_agent = lp.TesseractAgent(languages='deu')

        if None in [self.predictor, self.metadata]:
            raise UnsetAttributeError("extractLayout()", ["predictor", "metadata"])

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

                #   TODO: put this in a function in another module
                b_type = block.type.lower()
                if b_type in ['text', 'title', 'list']:
                    snippet = (block
                            .pad(left=5, right=5, top=5, bottom=5)
                            .crop_image(img))

                    #text = ocr_agent.detect(snippet)
                    text = image_to_string(snippet, lang='deu')
                    print(text)
                    block.set(text=text, inplace=True)

                elif b_type in ['table', 'figure']:
                    #   TODO: add table support
                    if b_type == 'Table':
                        logging.warning("Tables are currently not supported and are therefore processed like figures.")
                    snippet = block.crop_image(img)
                    img_name = page + ".jpg"
                    figure_path = self.output_path + "/figures/" + img_name
                    save_path = os.path.abspath(figure_path)    #   NOTE: is using the abs path correct?
                    cv2.imwrite(figure_path, snippet)
                    block.set(text=save_path, inplace=True)
                else:
                    logging.warning(f'Block of type \'{b_type}\' not supported.')
                    
                blocks.append(block)

            self.layouts.append(lp.Layout(blocks=blocks))

            #   TODO: set the data for each object type
            if visualize:
                #   TODO: 26 Apr - Move this to new function in other module
                v = Visualizer(img[:, :, ::-1],
                        self.metadata,
                        scale=1.0,
                        instance_mode=ColorMode.SEGMENTATION)

                visual = v.draw_instance_predictions(predicts.to("cpu"))
                vis_img = visual.get_image()[:, :, ::-1]
                
                visual_dir = self.output_path+"/visuals/"
                Path(visual_dir).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(visual_dir + str(page)+'.jpg', vis_img)

            if segment_sections:
                #   TODO: 26 Apr - add section segmentation (in new function in other module)
                pass

            #   TODO: 26 Apr - define what the "layout detection object" will look like!



    #   TODO: maybe add function to order only one layout if this is useful
    def orderLayouts(self):
        """Orders each page's layout based object bounding boxes"""
        for page, layout in enumerate(self.layouts):
            self.layouts[page] = layout.sort(key=lambda b:b.block.y_1)
            self.layouts[page] = lp.Layout([b.set(id = idx) for idx, b in enumerate(layout)])
        #   TODO: add support for Manhattan (page split), NOTE: do we need a specification of whether it's Manhattan or not or do we figure this out ourselves... When using a lot of documents like in our case this would be very handy

