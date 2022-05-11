def getHtmlSpanByType(block, section):
        """Gets HTML code representing a document objects content based on its type

        Args:
            block (layoutparser.elements.TextBlock): TextBlock obj containing document object data
            section (int): the current document section

        Returns:
            a string containing an HTML representation of document object
        """

        filetype = block.type.lower()
        text = str(block.text)

        if filetype in ["text", "title"]:
            html_span = getTextSpan(text, filetype, section)

        #   TODO: add table support
        elif filetype in ["figure", "table"]:
            coords = block.block.coordinates
            html_span = getImageSpan(text, coords, section)

        elif filetype == "list":
            html_span = getListSpan(text, section)

        return html_span

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
    width = str((x2 - x1) // 2)
    height = str((y2 - y1) // 2)
    html_span = (
        '<img src="'
        + path
        + '" class="'
        + str(section)
        + ' '
        + '" '
        + 'alt="document figure" width="'
        + width
        + '" height="'
        + height
        + '">'
    )

    return html_span

def getListSpan(text, section):
    """Gets HTML code representing a list

    Args:
        text (string): content (text) of document object
        section (int): the current document section

    Returns:
        string: a string containing an HTML representation of a list
    """

    split_text = text.replace("\f", "").split("\n")

    #   Remove empty lines (trailing '\n')
    split_text = [line for line in split_text if line != ""]

    html_span = '<ul class="' + str(section) + '">'

    for list_item in split_text:
        html_span += "<li>" + list_item + "</li>"

    html_span += "</ul>"

    return html_span

def getTextSpan(text, filetype, section):
    """Gets HTML code representing a paragraph or title depending on the type

    Args:
        text (string): content (text) of document object
        filetype (string): type of the document object
        section (int): the current document section

    Returns:
        HTML code string representing text
    """

    text = text.replace("\f", "")

    if filetype == "text":
        html_span = '<p class="' + str(section) + '">' + text + "</p>"

    elif filetype == "title":
        title_id = str(section) + text[0] + '-' + str(len(text))
        html_span = '<h2 class="' + str(section) + '" id="' + title_id + '">' + text + "</h2>"

    return html_span

