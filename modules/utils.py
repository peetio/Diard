import json
import os

import numpy as np
from pdf2image import convert_from_path


def getImagesFromPdf(pdf_path):
    """Extracts images for every page in pdf at specified path

    Args:
        pdf_path (str): The absolute or relative path to a single PDF file
    
    Returns:
        list: a list of images representing the pdf pages
    """

    pdf_images = convert_from_path(pdf_path)

    return pdf_images


def exportAsJson(json_obj, json_filename):
    """Saves a JSON object with the given filename

    Args:
        json_obj (string): a string representing a JSON object
    """

    with open(json_filename, 'w') as json_file:
        json_file.write(json_obj)


def makeDir(path):
    """Makes a directory to store output data if it doesn't exist already

    Args:
        path (string): path of to be created directory
    """

    if (not os.path.exists(path)):
        os.mkdir(path)    # create directory per PDF 


def makeDirs(paths):
    """Makes directories to store visualization and JSON output

    Args:
        paths (list): list of paths of to be created directories
    """

    for path in paths:
        makeDir(path)


def loadFromJson(json_path):
    """Gets JSON data from specified path

    Args:
        json_path (string): path to the a single JSON file

    Returns:
        list: a list containing JSON objects
    """
    with open(json_path, 'r') as json_file:
        json_content = json.load(json_file)

    return json_content


def filterLayoutJsons(jsons):
    """Filters out non-JSON files

    Args:
        jsons (list): a list of paths to JSON files

    Returns:
        list: all json files containing extracted layout data
    """

    temp_jsons = []
    for filename in jsons:
        if ((filename[len(filename)-5:] == ".json") and
                    (filename[len(filename)-6].isdigit())):
            temp_jsons.append(filename)

    # Sort by page number index in filename
    temp_jsons.sort(key = lambda filename: int(filename.split('.')[0]))

    return temp_jsons


def joinJsons(json_dir):
    """Exports a single JSON that recomposes the PDF with extracted data

    Args:
        json_dir (string): path to the directory containing JSONs for each page in PDF
    """

    jsons = os.listdir(json_dir)
    jsons = filterLayoutJsons(jsons)

    # Recomposition of PDF with JSON files in json_dir
    json_object = []
    for json_filename in jsons:
        json_content = loadFromJson(json_dir + '/' + json_filename)
        for block in json_content:
            json_object.append(block)

    joined_json_path = json_dir + '/' + "joined.json"
    exportAsJson(json.dumps(json_object), joined_json_path) # json_obj, json_filename
    

def exportJsonObj(json_obj, output_path):
    """Exports JSON object to given path

    Args:
        json_obj (dict): dictionary containing layout information with section
                            numbering for each document obj
        output_path (string): path to output modified JSON
    """

    with open(output_path, 'w') as json_file:
        json_file.write(json.dumps(json_obj))


def getJsonContent(json_path):
    """Loads the JSON content form specified file

    Args:
        json_path (str): path to a JSON file

    Returns:
        list: a list containing the JSON content
    """

    with open(json_path, 'r') as json_file:
        json_content = json.load(json_file)

    return json_content


