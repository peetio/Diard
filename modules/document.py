#   Document class definition
import os
import cv2
import logging
import json
import jenkspy
import layoutparser as lp
import numpy as np
import pandas as pd
from pytesseract import image_to_string

from tqdm import tqdm
from pathlib import Path
from pdf2image import convert_from_path
from detectron2.utils.visualizer import ColorMode, Visualizer
from layoutparser.elements import TextBlock, Rectangle

from modules.exceptions import DocumentFileFormatError, UnsetAttributeError, PageNumberError, InputJsonStructureError



class Document():
    @staticmethod
    def getRatio(coords, text):
        """Gets the surface over char count ratio

        Args:
            coords (tuple): top left and bottom right bounding box coordinates (x1, y1, x2, y2)
            text (string): extracted text from title object

        Returns:
            an integer representing surface / char count (ratio) of a title
        """ 

        #   set content & get first line of title
        text = text.strip()
        split_text = text.split("\n")
        first_line = split_text[0].strip()

        char_count = len(first_line)

        x1, y1, x2, y2 = coords  #   TODO: check if this extraction works
        width = x2 - x1
        height = y2 - y1

        #   get surface covering only the first line of the title
        surface = width * (height / len(split_text))

        if char_count > 0:
            ratio = surface / char_count
        else: ratio = 0

        return ratio



    @staticmethod
    def applyJenks(ratios, n_classes=3):
        """Applies Jenks Natural Breaks Optimization to find similar titles

        Args:
            ratios (list): list containing surface/ char count ratio for each title
            n_classes (int): number of classes used, should be 3 for heading, sub heading, and outliers

        Returns:
            list with labels for each title at corresponding index
        """

        #   add id for re-identification, [[id, ratio value], ...]
        ratios_id = [[i,ratio] for i, ratio in enumerate(ratios)]

        #   sort by ratio
        ratios_id = sorted(ratios_id, key=lambda ratio:ratio[1])

        values = [ratio[1] for ratio in ratios_id]
        breaks = jenkspy.jenks_breaks(values, nb_class=n_classes)
        labels = pd.cut(values,
                              bins=breaks, 
                              labels=[0, 1, 2],
                              include_lowest=True)

        #   reorder title ratios with re-identification
        reordered_labels = labels.copy()
        for i, label in enumerate(labels):
            reordered_labels[ratios_id[i][0]] = label
            
        return reordered_labels



    @staticmethod
    def mapJenksLabels(ratios, labels, label_map = {0 : 'heading', 
                                    1 : 'sub', 
                                    2 : 'random'}):
        """Maps Jenks Algorithm label output to title categories 

        Args:
            ratios (list): ratio value for each title
            labels (list): a list where each label corresponds to a title object
            label_map (dict): label map containing indexed title categories

        Returns:
            a title category label list
        """
 
        original_map = {0 : [], 1 : [], 2 : []} 

        for i, label in enumerate(labels):
            #   save original indexes of labels
            original_map[label].append(i)

        #   find outlier label by min number of samples
        outlier_id = 0
        smallest_len = len(original_map[0])
        for i in range(len(original_map)):
            if(len(original_map[i]) <= smallest_len):
                label_map[i] = 'outliers' 
                outlier_id = i

        #   start with random outlier ratio
        random_outlier_ratio = ratios[original_map[outlier_id][0]]

        #   gets index that's not outlier id
        if (outlier_id in [0, 1]):
            random_id = [0, 1]
            random_id.remove(outlier_id)
            random_id = random_id[0]  

        else: random_id = 0

        min_diff = abs(ratios[original_map[random_id][0]] - random_outlier_ratio)
        biggest_ratio = 0
        min_diff_id = 0

        #   find 'heading' & 'sub' labels indexes
        for i in range(len(original_map)):
            if (i != outlier_id):
                random_label_ratio = ratios[original_map[i][0]]
                ratio_diff = abs(random_label_ratio - random_outlier_ratio)
                
                if(random_label_ratio >= biggest_ratio):
                    biggest_ratio = random_label_ratio

                    #   map remaining labels
                    label_map[i] = 'heading' 
                    for j in range(len(original_map)):
                        if j not in [outlier_id, i]:
                            label_map[j] = 'sub' 

                if(ratio_diff <= min_diff):
                    min_diff = ratio_diff
                    min_diff_id = i
            
            #   merge outliers with closest neighbouring category
            if (i == len(original_map)-1):
                label_map[outlier_id] = label_map[min_diff_id]

        #   map every label in original label list 
        mapped_labels = []
        for label in labels:
            mapped_labels.append(label_map[label])

        return mapped_labels



    @classmethod
    def getLayoutJsonDims(cls, l, dims=0):
        """Recursive function to find number of dimensions (n of rows) in a list

        Args:
            l (list): list of which you want to know the number of rows
            dims (int): the current dimensionality or number of rows in the "list"

        Returns:
            the dimensionality (number of rows) in the "list"
        """

        if not type(l) == list:
            return dims

        return cls.getLayoutJsonDims(l[0], (dims+1))



    @staticmethod
    def getListSpan(text, section):
        """Gets HTML code representing a list

        Args:
            text (string): content (text) of document object
            section (int): the current document section
        
        Returns:
            string: a string containing an HTML representation of a list
        """

        split_text = (text
                          .replace("\f", '')
                          .split("\n"))

        #   Remove empty lines (trailing '\n')
        split_text = [line for line in split_text if line != '']

        html_span = '<ul class="'+str(section)+'">'

        for list_item in split_text:
            html_span += "<li>"+list_item+"</li>"

        html_span += "</ul>"

        return html_span



    @staticmethod
    def getTextSpan(text, filetype, section):
        """Gets HTML code representing a paragraph or title depending on the type

        Args:
            text (string): content (text) of document object
            filetype (string): type of the document object
            section (int): the current document section
        
        Returns:
            HTML code string representing text
        """

        # TODO: postprocessing of OCR (i.e., replace '-',...)
        text = text .replace("\f", '')

        if filetype == 'text':
            html_span = '<p class="'+str(section)+'">'+text+"</p>"

        elif filetype == 'title':
            html_span = '<h2 class="'+str(section)+'">'+text+"</h2>"
        
        return html_span



    @staticmethod
    def getImageSpan(path, coords, section):
        """Gets HTML code representing an image

        Args:
            abs_path (string): absolute path to the image
            coords (tuple): top left and bottom right bounding box coordinates (x1, y1, x2, y2)
            section (int): the current document section
        
        Returns:
            HTML code string representing a figure
        """

        x1, y1, x2, y2 = coords
        width = str((x2-x1)//2)
        height = str((y2-y1)//2)
        html_span = '<img src="'+path+'" class="'+str(section)+'" ' + 'alt="document figure" width="'+width+'" height="'+height+'">'

        return html_span



    @staticmethod
    def getHtmlSpanByType(block, section):
        """Gets HTML code representing a document objects content based on its type

        Args:
            block (layoutparser.elements.TextBlock): TextBlock obj containing document object data
            section (int): the current document section
        
        Returns:
            a string containing an HTML representation of document object
        """
        #   TODO: now it only works with section, make it also useable without section segmentation

        filetype = block.type.lower()
        text = str(block.text)

        if(filetype in ['text', 'title']):
            html_span = Document.getTextSpan(text, filetype, section)
                    
        #   TODO: add table support
        elif(filetype in ['figure', 'table']):
            coords = block.block.coordinates
            html_span = Document.getImageSpan(text, coords, section)
                    
        elif(filetype == 'list'):
            html_span = Document.getListSpan(text, section)
               
        return html_span



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
        if file_format not in ['pdf']:  #   TODO: test if other variations of pdf and docx can be used with image conversion (add docx when support added)
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
        self.layouts = []
        self.label_map = label_map



    def docToImages(self):
        """Converts each page of a document to images"""
        #   TODO: add word document support
        pil_imgs = convert_from_path(self.source_path) 
        self.images = [np.asarray(img) for img in pil_imgs]



    def setExtractedData(self, b_type, block, img, page):
        """Extracts and sets content of document object (block)"""

        figure_dir = self.output_path + "/figures/"
        Path(figure_dir).mkdir(parents=True, exist_ok=True)

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
            img_name = str(page) + ".jpg"
            figure_path = figure_dir + img_name
            save_path = "../figures/" + img_name

            cv2.imwrite(figure_path, snippet)
            block.set(text=save_path, inplace=True)
        else:
            logging.warning(f'Block of type \'{b_type}\' not supported.')



    def extractLayouts(self, segment_sections=False, visualize=False):
        """Extracts and sets layout from document images
        
        Args:
            segment_sections (bool): if True sections are segmented
            visualize (bool): if True detection visualizations are saved
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
                self.setExtractedData(b_type, block, img, page)    
                blocks.append(block)

            self.layouts.append(lp.Layout(blocks=blocks))

            if visualize:
                self.visualizePredictions(predicts, img, page)

        if segment_sections:
            self.orderLayouts()
            logging.info('Ordering layout for section segmentation.')
            self.segmentSections()



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



    def pageCheck(self, page):
        """Checks if the page is within range
        
        Args:
            page (int): page number to check
        """

        pages = len(self.layouts)
        if not 0 <= page <= pages-1: 
            raise PageNumberError(page, pages)



    def getLayoutJson(self, page):
        """Gets JSON object representing a single document page layout

        Args:
            page (int): page number of document

        Returns:
            JSON object representing a document single page layout
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
        """Gets JSON object representing the whole document layout

        Returns:
            JSON objects (list of dictionaries) containing whole document layout
        """
        
        layouts_json = []
        for page in range(len(self.layouts)):
            layouts_json.append(self.getLayoutJson(page))
        
        return layouts_json



    def saveLayoutAsJson(self, page):
        """Saves JSON representation of single document page layout

        Args:
            page (int): page number of document
        """

        self.pageCheck(page)

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



    def getLayoutHtml(self, page, section=None):
        """Saves HTML representation of single document page layout

        Args:
            page (int): page number of document
            section (int): section to start from if processing multiple pages
        
        Returns:
            HTML representation of a single document page layout
        """

        if section is None:
            html = ""
            single = True
            section=0
        else:
            html = '<link rel="stylesheet" href="../../../resources/stylesheet.css">' 
            html += '<script src="../../../resources/stylescript.js"></script>'
            html += '<body>'
            single = False

        for b in self.layouts[page]:
            try:
                new_section = not b.section == section
                is_title = b.type == 'title'
                if new_section and is_title:
                    section = b.section
                    html+='<hr class="'+str(section)+'">'
            except AttributeError:
                #   no section segmentation used
                pass

            html_span = self.getHtmlSpanByType(b, section)
            html+=html_span

        if single: 
            return html+'</body'
        else: 
            return (html, section)



    def getLayoutsHtml(self):
        """Gets HTML representation of a whole document layout

        Returns:
            HTML representation of whole document layout
        """

        html = '<link rel="stylesheet" href="../../../resources/stylesheet.css">' 
        html += '<script src="../../../resources/stylescript.js"></script>'
        html += '<body>'
        section=0
        for page in range(len(self.layouts)):
            html_span, section = self.getLayoutHtml(page, section)
            html+=html_span

        html += '</body>'
        return html



    def saveLayoutAsHtml(self, page):
        """Saves JSON representation of single document page layout

        Args:
            page (int): page number of document
        """
        #   TODO: add visualization (javascript + css)

        self.pageCheck(page)
        html_dir = self.output_path+"/htmls/" 
        Path(html_dir).mkdir(parents=True, exist_ok=True)
        html_path = html_dir+str(page)+".html"
        layout_html = self.getLayoutHtml(page)

        html_path = html_dir+str(page)+".html"
        with open(html_path, 'w') as f:
            f.write(layout_html)

   

    def saveLayoutsAsHtml(self):
        """Saves HTML representation of whole document layout"""
        
        html_dir = self.output_path+"/htmls/" 
        Path(html_dir).mkdir(parents=True, exist_ok=True)

        for page in range(len(self.layouts)):
            self.saveLayoutAsHtml(page)

        html_path = html_dir+self.name+".html"
        layouts_html = self.getLayoutsHtml()

        with open(html_path, 'w') as f:
            f.write(layouts_html)



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

        #   single or joined layouts file check based on list dimensionality
        dims = self.getLayoutJsonDims(l=layout_json)
        if dims == 1:
            self.jsonToLayout(layout_json)
        elif dims == 2:
            for layout in layout_json:
                self.jsonToLayout(layout)
        else:
            raise InputJsonStructureError(filename)



    def sectionByChapterNums(self):
        """Finds sections based on the chapter numbering

        Returns:
            a label list where each item corresponds to a title object
        """

        curr_chapter = None
        labels = []
        for layout in self.layouts:
            for b in layout:
                if b.type.lower() == 'title':
                    chapter= ""
                    text = b.text.strip()
                    for t in text:
                        if t.isdigit():
                            chapter+=(t)
                        else: break

                    #   compare first digit of title w/ previous
                    if (len(chapter) > 0 and 
                                chapter != curr_chapter):

                        curr_chapter = chapter 
                        labels.append('heading')
                    else:
                        labels.append('sub')

        return labels 



    def getTitleRatios(self):
        """Gets the ratio (bounding box surface / char count) for each title

        Returns:
            list containing ratio for each title
        """

        ratios = [] 
        for layout in self.layouts:
            for b in layout:
                if b.type.lower() == 'title':
                    t = b.text
                    ratio = self.getRatio(b.block.coordinates, t)
                    ratios.append(ratio)
                        
        return ratios 



    def prioritizeLabels(self, cn_labels, r_labels):
        """Compares segmentation method outputs to get best of both worlds

        Args:
            cn_labels (list): chapter numbering section segmentation output labels
            r_labels (list): natural breaks section segmentation output labels

        Returns:
            a list containing the combined labels
        """

        #   TODO: fix chapter numbering failures
                    #   might have something to do with the reordering...
        labels =  []
        title_id = 0

        for layout in self.layouts:
            for b in layout:
                if b.type.lower() == "title":
                    #   prioritize numbered chapter headings
                    text = b.text.strip()
                    c = text[0]
                    is_digit = c.isdigit()
                    isnt_empty = len(r_labels) > 0
                    is_heading = r_labels[title_id] == 'heading'

                    if cn_labels[title_id] == 'heading':
                        labels.append('heading')
                    
                    elif (not is_digit and isnt_empty and is_heading):
                        labels.append('heading')
                    else:
                        labels.append('sub')

                    title_id+=1

        return labels



    def sectionByRatio(self):
        """Finds sections based on ratio (bounding box surface / char count)

        Returns:
            a label list where each item corresponds to a title object
        """

        n_classes = 3   #   heading, sub heading, outliers
        mapped_labels = []
        ratios = self.getTitleRatios()

        if len(ratios) > n_classes:
            labels = self.applyJenks(ratios, n_classes)
            mapped_labels = self.mapJenksLabels(ratios, labels)
        else:
            logging.warning(f'Not enough titles detected in \'{self.name}\' to find natural breaks, using only chapter numbering. Minimum number of titles is {n_classes}')

        return mapped_labels 



    def setSections(self, labels):
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
                    if(labels[title_id] == 'heading'):
                        section += 1
                    title_id+=1
                b.section = section 



    def segmentSections(self):
        """Segments sections based on numbering and natural breaks"""
        cn_labels = self.sectionByChapterNums()
        r_labels = self.sectionByRatio()
        labels = self.prioritizeLabels(cn_labels, r_labels)
        self.setSections(labels)

