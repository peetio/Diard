import json
import os

from modules.utils import makeDir


def getBlocksFromJson(json_path):
    """Gets layout and content information from specified JSON file

    Args:
        json_path (str): path to a JSON file
    
    Returns:
        list: a list containing the JSON content
    """


    with open(json_path, 'r') as json_file:
        blocks = json.load(json_file)

    return blocks


def getTextSpan(content, filetype):
    """Gets HTML code representing a paragraph or title depending on the type

    Args:
        content (string): content (text) of document object
        filetype (string): type of the document object
    
    Returns:
        string: a string containing an HTML representation
    """

    # TODO: postprocessing of OCR (i.e., replace '-')
    content = content.replace("\f", '')

    if filetype == 'Text':
        html_span = "<p>"+content+"</p>"

    elif filetype == 'Title':
        html_span = "<h1>"+content+"</h1>"
    
    return html_span


def getImageSpan(content, block):
    """Gets HTML code representing an image

    Args:
        content (string): path to the image
        block (layoutparser.elements.TextBlock): TextBlock obj containing bounding box coordinates 
    
    Returns:
        string: a string containing an HTML representation
    """

    x1, y1, x2, y2 = block['box']
    html_span = "<img src=\""+content+"\" alt=\"document figure\" width=\""+str((x2-x1)//2)+"\" height=\""+str((y2-y1)//2)+"\">"

    return html_span


def getListSpan(content):
    """Gets HTML code representing a list

    Args:
        content (string): content (text) of document object
    
    Returns:
        string: a string containing an HTML representation
    """


    split_content = (content
                        .replace("\f", '')
                        .split("\n"))

    #   Remove empty lines (trailing '\n')
    split_content = [line for line in split_content if line != '']

    html_span = "<ul>"

    for list_item in split_content:
        html_span += "<li>"+list_item+"</li>"

    html_span += "</ul>"

    return html_span


def getHtmlContent(json_content):
    """Gets the HTML code representing a document's extracted data

    Args:
        json_content (dict): a dictionary containing JSON objects 
    
    Returns:
        string: a string containing an HTML representation
    """

    # TODO: filter out the double newlines
    html_content = ""
    section = -1 

    for block in json_content:
        html_content += getHtmlSpanByType(block)
        
        #   New section check & Heading marking
        if('section' in block 
                            and block['section'] != section
                            and block['type'] == "Title"):

            section = block['section']
            html_content += '<hr>'

    return html_content


def getHtmlSpanByType(block):
    """Gets HTML code representing a document objects content based on its type

    Args:
        block (layoutparser.elements.TextBlock): TextBlock obj containing document object data
    
    Returns:
        string: a string containing an HTML representation of document object
    """

    filetype = block['type']
    content = str(block['content'])

    if(filetype in ['Text', 'Title']):
        html_span = getTextSpan(content, filetype)
                
    elif(filetype in ['Figure', 'Table']):
        html_span = getImageSpan(content, block)
                
    elif(filetype == 'List'):
        html_span = getListSpan(content)
           
    return html_span


def writeToHtml(json_content, html_dir, json_filename):
    """Creates & saves an HTML file representing a document's content

    Args:
        json_content (dict): a dictionary containing JSON objects 
        html_dir (string): path to the output directory
        json_filename (string): name of the json file being processed, normally this is a page number
    """
   
    #   Make output dir
    makeDir(html_dir)

    #   Make, update, and save output file
    html_filename = json_filename[:len(json_filename)-5] + ".html"
    html_path = html_dir + '/' + html_filename 
    html_file = open(html_path, 'w')

    html_content = getHtmlContent(json_content)

    html_file.write(html_content)
    html_file.close()

