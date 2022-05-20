from io import StringIO

import pandas as pd


def getLayoutHtml(layout, section=None):
    """Saves HTML representation of single document page layout

    Args:
        layout (layoutparser.Layout): document objects
        page (int): page number of document
        section (int): section to start from if processing multiple pages

    Returns:
        HTML representation of a single document page layout
    """

    html = ""
    if section is None:
        single = True
        section = 0
    else:
        single = False

    for b in layout:
        try:
            new_section = not b.section == section
            is_title = b.type == "title"
            if new_section and is_title:
                section = b.section
        except AttributeError:
            #   no section segmentation used
            pass

        html_span = getHtmlSpanByType(b, section)
        html += html_span

    if single:
        return html + "</body>"
    else:
        return (html, section)


def getLayoutsHtml(layouts):
    """Gets HTML representation of a whole document layout

    Args:
        layouts (list): list of Layout instances

    Returns:
        HTML representation of whole document layout
    """

    html = (
        '<link rel="stylesheet" href="../../../resources/stylesheet.css">'
        + '<link rel="preconnect" href="https://fonts.googleapis.com">'
        + '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        + '<link href="https://fonts.googleapis.com/css2?family=Roboto+Condensed&display=swap" rel="stylesheet">'
        + '<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">'
        + '<script src="../../../resources/stylescript.js"></script>'
        + '<div class="toc_container">'
        + '<div class="toc">'
        + "<h4>TABLE OF CONTENTS</h4>"
        + '<div class="table"></div></div></div>'
        + '<div id="layout">'
        + "<body>"
    )

    section = 0
    for page in range(len(layouts)):
        html_span, section = getLayoutHtml(layouts[page], section)
        html += html_span

    html += "</div></body>"
    return html


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
        if filetype == "table" and not text.endswith("jpg"):
            html_span = getTableSpan(text, section)
        else:
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
        + " "
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


def getTableSpan(text, section):
    """Gets HTML code representing a table

    Args:
        text (string): string containing the dataframe after .to_string()
        section (int): the current document section

    Returns:
        string: a string containing an HTML representation of a table
    """
    #   reconstruct dataframe from string
    table_str = StringIO(text)
    df = pd.read_csv(table_str)

    #   TODO: add support for more complex tables
    html_span = "<table class='" + str(section) + "'>"
    for row_idx in range(len(df)):
        html_span += "<tr>"
        for (col_idx, col_rows) in df.iteritems():
            html_span += "<td>" + col_rows[row_idx] + "</td>"
        html_span += "</tr>"

    html_span += "</table>"
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
        title_id = str(section) + text[0] + "-" + str(len(text))
        html_span = (
            '<h2 class="' + str(section) + '" id="' + title_id + '">' + text + "</h2>"
        )

    return html_span
