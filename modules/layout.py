import cv2
import json
import layoutparser as lp


def sortBlocks(layout):
    """Sorts bounding boxes based on top left y coordinate

    Args:
        layout (layoutparser.elements.Layout): a Layout obj containing TextBlock objects of any type

    Returns:
        layoutparser.elements.Layout: returns a sorted copy of the Layout object
    """

    return layout.sort(key=lambda b:b.block.y_1) # filter on y1 coordinate


def setOrderedIds(layout):
    """Sets IDs for each TextBlock in a Layout object's list of blocks

    Args:
        layout (layoutparser.elements.Layout): a Layout obj containing TextBlock objects of any type

    Returns:
        layoutparser.elements.Layout: indexed version of the object given as parameter
    """

    return lp.Layout([b.set(id = idx) for idx, b in enumerate(layout)])


def setExtractedText(image, text_blocks):
    """Sets the text for blocks of the type ["Text", "Title", "List"]

    Args:
        image (numpy.ndarray): a cv2 RGB document image
        text_blocks (layoutparser.elements.Layout): a Layout obj containing TextBlock objects of types "Text", "Title", and "List"
    """

    ocr_agent = lp.TesseractAgent(languages='deu')

    # text recognition for each document text object
    for block in text_blocks:
        segment_image = (block
                            .pad(left=5, right=5, top=5, bottom=5)
                            .crop_image(image))

        text = ocr_agent.detect(segment_image)
        block.set(text=text, inplace=True)


def setExtractedImage(image, figure_blocks, filename, page_n):
    """Saves an image of segment and sets path to image as text for blocks of the type "Figure"

    Args:
        image (numpy.ndarray): a cv2 RGB document image
        figure_blocks (layoutparser.elements.Layout): a Layout obj containing only TextBlock objects of type "Figure"
    """

    inter_i = 0
    for block in figure_blocks:
        segment_image = block.crop_image(image)

        img_filename = filename + '_' + str(inter_i) + str(page_n) + ".jpg"
        figure_dir= "jsons/" + filename + "/figures/" + img_filename
        save_path = "./output/" + figure_dir 
        block_filename = "../../" + figure_dir

        cv2.imwrite(save_path, segment_image)
        block.set(text=block_filename, inplace=True)

        inter_i+=1


def filterBlocksByType(layout, block_type):
    """Filters TextBlocks in a Layout obj based on their type

    Args:
        layout (layoutparser.elements.Layout): a Layout obj containing TextBlock objects of any type
        block_type (str): type of block you want to keep in the list of blocks

    Returns:
        layoutparser.elements.Layout: layout object containing TextBlocks of only the specified type
    """

    filtered_blocks = lp.Layout([b for b in layout if (b.type==block_type
                                                                    .lower()
                                                                    .capitalize())])

    return filtered_blocks 


def filterOutBlocksByType(layout, type_list):
    """Filters out TextBlocks in a Layout obj based on their type

    Args:
        layout (layoutparser.elements.Layout): a Layout obj containing TextBlock objects of any type
        block_type (str): type of block you want to keep in the list of blocks

    Returns:
        layoutparser.elements.Layout: layout object containing TextBlocks that don't have the specified type
    """

    # Input: Layout that uses label_map, list of block type strings
    # Return: Layout object with only blocks of type not in 'block_type'
    type_list = [block_type.lower().capitalize() for block_type in type_list]
    filtered_blocks = lp.Layout([b for b in layout if (b.type not in type_list)])

    return filtered_blocks 


def setExtractedData(image, layout, filename, page_n):
    """Sets extracted data of any block type as text for each TextBlock in the Layout obj

    Args:
        image (numpy.ndarray): a cv2 RGB document image
        layout (layoutparser.elements.Layout): a Layout obj containing TextBlock objects of any type
        page_n (int): the page number of PDF on which the objects were detected
    """

    text_like_blocks = filterOutBlocksByType(layout, ['Figure', 'Table'])
    figure_blocks = filterOutBlocksByType(layout, ['Text', 'Title', 'List'])

    # set text for TextBlocks of each type
    setExtractedText(image, text_like_blocks)
    setExtractedImage(image, figure_blocks, filename, page_n)


def getOrderedLayout(boxes, classes, scores, 
        label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"}):
    """Indexes and orders (sorts) detections on the document

    Args:
        boxes (detectron2.structures.Boxes): a torch.Tensor containing bounding boxes (x1, y1, x2, y2)
        classes (list): a list of class indexes that correspond to a class in the label_map
        scores (torch.Tensor): a tensor containing detection scores
        label_map (dictionary): a dictionary to find the corresponding class for a given index

    Returns:
        layoutparser.elements.Layout: Layout obj containing indexed and sorted TextBlocks of any type
    """

    # params: Boxes, Classes and Scores objects from Detectron2 toolkit
    # returns: indexed and ordered Layout Parser Layout object instance

    np_boxes = boxes.tensor.cpu().numpy()

    # rectangle = x1, y1, x2, y2
    rectangles = [lp.Rectangle(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in np_boxes]

    textBlocks = []
    for i in range(len(rectangles)):
        textBlocks.append(lp.TextBlock(block=rectangles[i], 
                                        type=label_map[classes[i]], 
                                        score=scores[i]))

    layout = lp.Layout(blocks=textBlocks)
    layout = sortBlocks(layout)
    layout = setOrderedIds(layout)

    return layout

   
def getPredictionData(output):
    """Get the data from the inference output (predictions)

    Args:
        output (detectron2.structures.instances.Instances): an Instance obj containing the different fields (incl. bounding boxes, prediction scores, and class indexes)

    Returns:
        list: a list containing a list of bounding boxes, a list of classes, and a list of prediction scores
    """

    boxes = output.pred_boxes if output.has("pred_boxes") else None
    classes = output.pred_classes.tolist() if output.has("pred_classes") else None
    scores = output.scores if output.has("scores") else None
    
    return boxes, classes, scores


def getJsonObjectsFromLayout(layout, page_n):
    """Get a JSON object from a Layout object

    Args:
        layout (layoutparser.elements.Layout): a Layout obj containing TextBlock objects of any type
        page_n (int): the page number of PDF on which the objects were detected

    Returns:
        string: a string representing a JSON object
    """

    json_blocks = []

    for block in layout:
        json_blocks.append({"id" : block.id,
                                "type": block.type,
                                "content" : block.text,
                                "box": block.coordinates,
                                "page": page_n})

    return json.dumps(json_blocks)

