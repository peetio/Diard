import json
import os

import jenkspy
import numpy as np
import pandas as pd
from pdf2image import convert_from_path

from modules.utils import exportJsonObj, getJsonContent

#   Data loading and exportation

#   One-dimensional Section Segmentation

def applyJenks(titles, feature_type, nb_class):
    """Applies Jenks Natural Breaks Optimization on specified title feature

    Args:
        titles (dict): dictionary containing the title content and feature value
        feature_type (string): key to access feature value in title dictionary

    Returns:
        list: a list with labels for each title at the same index but without certainty
                about the corresponding category for each label index
    """

    #   Apply Jenks Natural Breaks Optimization
    df = pd.DataFrame(titles)

    #   Add original indexes
    original_idxs = [idx for idx in range(len(df['title_content']))]
    df['original_idxs'] = original_idxs

    df = df.sort_values(by=feature_type)

    breaks = jenkspy.jenks_breaks(df[feature_type], nb_class=nb_class)
    df['labels'] = pd.cut(df[feature_type], 
                                    bins=breaks, 
                                    labels=['0', '1', '2'],
                                    include_lowest=True)

    labels = dfToLabels(df)

    return labels


def dfToLabels(df):
    """Converts DataFrame to a list of labels

    Args:
        df (pandas.DataFrame): DataFrame containing Jenks Optimization output

    Returns:
        list: a list with labels for each title at the same index
    """ 

    labels = []
    for idx in range(len(df['labels'])):
        label_at_idx = df.loc[df['original_idxs'] == idx]['labels']
        labels.append(int(label_at_idx))

    return labels 


def calcRatio(block, title_content):
    """Calculates the surface over char count ratio

    Args:
        block (layoutparser.elements.TextBlock): TextBlock obj containing bounding box coordinates 
        title_content (string): title document object extracted text

    Returns:
        int: an integer representing surface / char count (ratio)
    """ 

    #   Set content & Get first line of title
    title_content = title_content.strip()
    title_content_split = title_content.split("\n")
    first_line = title_content_split[0]

    char_count = len(first_line)

    x1, y1, x2, y2 = block['box']
    box_width = x2 - x1
    box_height = y2 - y1

    #   Get surface covering only the first line of the title
    surface = box_width * (box_height / len(title_content_split))

    #   Surface over Character Count
    if char_count > 0:
        ratio = surface / char_count
    else: ratio = 0

    return ratio


def setTitleSurfaceCCRatio(json_content):
    """Sets the surface over char count ratio for each title

    Args:
        json_content (dict): a list containing JSON objects representing layout objects 

    Returns:
        dict: dictionary containing the title content and feature value
    """ 

    titles = {
            'title_content' : [],
            'ratio': []
            }

    for block in json_content:
        if block['type'] == "Title":
            title_content = block['content']
            titles['title_content'].append(title_content)
            ratio = calcRatio(block, title_content)
            titles['ratio'].append(ratio)
                
    return titles


def setSections(json_content, mapped_labels):
    """Sets the section numbers for each layout object

    Args:
        json_content (dict): a list containing JSON objects representing layout objects 
        mapped_labels (list): a list containing the category index for each title at the same index
    """
    
    section_n= 0
    title_idx = 0

    #   Set section for each block & Export as Json
    for block in json_content:
        #   Check if its section heading
        if block['type'] == "Title":
            if(mapped_labels[title_idx] == 'heading'):
                section_n += 1

            title_idx+=1
        block['section'] = section_n 


def sectionByRatio(json_content, filename, nb_class=3):
    """Segments the sections based on the ratio of each title

    Args:
        json_content (dict): a list containing JSON objects representing layout objects
    """
    
    #   Use surface over char count ratio clustering
    mapped_labels = []
    titles = setTitleSurfaceCCRatio(json_content)
    if(len(titles['ratio']) > nb_class):
        labels = applyJenks(titles, 'ratio', nb_class)
        mapped_labels = classifyLabels(titles, labels)
    else:
        print("Not enough titles detected in '" + filename + "' to find natural breaks in ratio, returning empty list. Minimum number of titles is " + str(nb_class) + '.')

    return mapped_labels


def addOriginalIndexes(labels, label_map):
    """Gets split label list containing original title indexes and corresponding label index

    Args:
        labels (list): a list with labels for each title at the same index
        label_map (dict): a dictionary with correct mapping from label index to category string

    Returns:
        dict: a dictionary containing a list for each label index
    """
 
    split_labels = {0 : [],
                    1 : [],
                    2 : []} 

    for i in range(len(labels)):
        label = labels[i]
        split_labels[label].append(i)   #   Add original index to list of labels

    return split_labels


def getOutliersIndex(split_labels, label_map):
    """Gets the index of the list containing the outlier title indexes

    Args:
        split_labels (dict): a dictionary containing a list for each label index
        label_map (dict): a dictionary with correct mapping from label index to category string
    """
 
    outlier_idx = 0
    smallest_len = len(split_labels[0])
    for i in range(len(split_labels)):
        if(len(split_labels[i]) <= smallest_len):
            label_map[i] = 'outliers' 
            outlier_idx = i
    
    return outlier_idx


def mergeAndCategorize(ratios, outlier_idx, split_labels, label_map):
    """Merges the outlier titles with the closest neighbouring category (sub or normal headings)
        after figuring out which label index represents the other categories

    Args:
        ratios (list): a list containing the ratio of each title at the same index
        outlier_idx (int): label map index of the outlier titles
        split_labels (dict): a dictionary containing a list for each label index
        label_map (dict): a dictionary with correct mapping from label index to category string
    """
 
    #   Merges the outliers with either normal or sub headings
    random_outlier_ratio = ratios[split_labels[outlier_idx][0]]

    #   Gets index that is not outlier idx
    if (outlier_idx in [0, 1]):
        random_idx = [0, 1]
        random_idx.remove(outlier_idx)
        random_idx = random_idx[0]  

    else: random_idx = 0

    smallest_diff = abs(ratios[split_labels[random_idx][0]] - random_outlier_ratio)

    biggest_ratio = 0
    smallest_diff_idx = 0

    #   Find 'heading' & 'sub' labels indexes
    for i in range(len(split_labels)):
        if (i != outlier_idx):
            random_label_ratio = ratios[split_labels[i][0]]
            ratio_diff = abs(random_label_ratio - random_outlier_ratio)
            
            if(random_label_ratio >= biggest_ratio):
                biggest_ratio = random_label_ratio

                #   Map 'heading' and 'sub heading' indexes
                label_map[i] = 'heading' 
                for j in range(len(split_labels)):
                    if j not in [outlier_idx, i]:
                        label_map[j] = 'sub' 

            if(ratio_diff <= smallest_diff):
                smallest_diff = ratio_diff
                smallest_diff_idx = i
        
        #   Merge/ Map outliers to closest neighbouring category
        if (i == len(split_labels)-1):
            label_map[outlier_idx] = label_map[smallest_diff_idx]


def mapLabels(labels, label_map):
    """Maps each label outputted by the optimization method to the correct category in specified label map

    Args:
        labels (list): a list with labels for each title at the same index
        label_map (dict): a dictionary with correct mapping from label index to category string

    Returns:
        list: a list containing the category index for each title at the same index
    """
 
    #   Map every label in original label list 
    mapped_labels = []
    for label in labels:
        mapped_labels.append(label_map[label])

    return mapped_labels


def classifyLabels(titles, labels):
    """Classifies the labels outputted by the optimization method as the corresponding the title category

    Args:
        titles (dict): dictionary containing the title content and feature value
        labels (list): a list with labels for each title at the same index

    Returns:
        list: a list containing the category index for each title at the same index
    """
 
    label_map = {0 : 'heading', 1 : 'sub', 2 : 'random'}

    #   Segment/ split buckets
    split_labels = addOriginalIndexes(labels, label_map)
    
    outlier_idx = getOutliersIndex(split_labels, label_map)

    #   Merge outliers with closest neighbour
    ratios = titles['ratio']
    mergeAndCategorize(ratios, outlier_idx, split_labels, label_map)

    mapped_labels = mapLabels(labels, label_map)

    return mapped_labels



#   Chapter Numbering Section Segmentation
def sectionByChapterNumbering(json_content):
    """Segments the sections based on the chapter numbering

    Args:
        json_content (dict): a list containing JSON objects representing layout objects
    """

    chapter_n = 0
    curr_chapter = None
    labeled_titles = []

    for block in json_content:
        if block['type'] == "Title":
            extracted_chapter = ""
            for c in block['content']:
                if c.isdigit():
                    extracted_chapter+=(c)
                else: break

            if (len(extracted_chapter) > 0 and 
                        extracted_chapter != curr_chapter):

                curr_chapter = extracted_chapter
                labeled_titles.append('heading')

            else:
                labeled_titles.append('sub')


    return labeled_titles


def combineSegmentationMethods(json_content, chapter_numbered_labels, ratio_labels):
    """Combines two segmentation results the based on a number of criteria

    Args:
        json_content (dict): a list containing JSON objects representing layout objects
        chapter_numbered_labels (list): a list containing labeled title output from chapter numbering segmentation
        ratio_labels (list): a list containing labeled title output from Jenks Optimization segmentation
    """


    labeled_titles =  []
    title_idx = 0

    for block in json_content:
        if block['type'] == "Title":
            #   Prioritize numbered chapter headings
            if (chapter_numbered_labels[title_idx] == 'heading'):
                labeled_titles.append('heading')

            elif (not block['content'][0].isdigit()
                                    and len(ratio_labels) > 0
                                    and ratio_labels[title_idx] == 'heading'
                                    ):
                labeled_titles.append('heading')

            else:
                labeled_titles.append('sub')

            title_idx+=1

    return labeled_titles



def segmentSections(json_dir, pdfs):
    """Main function to segment sections for multiple PDFs

    Args:
        json_dir (string): path to the directory containing JSON directories for each PDF
    """

    for pdf_filename in pdfs:
        #   Step 1. Define path to JSON containing layout objects & output file

        filename = pdf_filename[:len(pdf_filename)-4]
        json_path = json_dir + filename + '/' + "joined.json"
        json_content = getJsonContent(json_path)
        output_path = json_dir + filename + '/' + "sectioned.json"

        #   Step 2. Perform section segmentation
        cn_labeled_titles = sectionByChapterNumbering(json_content)
        r_labeled_titles = sectionByRatio(json_content, pdf_filename)

        #   Step 3. Combine the results of each methods to get best of both worlds
        labeled_titles = combineSegmentationMethods(json_content, 
                                                        cn_labeled_titles, 
                                                        r_labeled_titles)
        
        #   Step 4. Update & Export the JSON object
        setSections(json_content, labeled_titles)
        exportJsonObj(json_content, output_path)

