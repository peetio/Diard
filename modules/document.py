#   Document class definition
import os
import cv2
import layoutparser as lp
import numpy as np
from tqdm import tqdm

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
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
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

        if None in [self.predictor, self.metadata]:
            raise UnsetAttributeError("extractLayout()", ["predictor", "metadata"])

        for i, img in tqdm(enumerate(self.images),
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
                if block.type == 
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

                visual_path = self.output_path+'/'+str(i)+'.jpg'
                cv2.imwrite(visual_path, vis_img)

            if segment_sections:
                #   TODO: 26 Apr - add section segmentation (in new function in other module)
                pass

            #   TODO: 26 Apr - define what the "layout detection object" will look like!



    def orderLayouts(self):
        """Orders each page's layout based object bounding boxes"""
        for i, layout in enumerate(self.layouts):
            self.layouts[i] = layout.sort(key=lambda b:b.block.y_1)
            self.layouts[i] = lp.Layout([b.set(id = idx) for idx, b in enumerate(layout)]) #   NOTE: not sure if this changes the original item in the list
        #   TODO: add support for Manhattan, NOTE: do we need a specification of whether it's Manhattan or not or do we figure this out ourselves... When using a lot of documents like in our case this would be very handy

