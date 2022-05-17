import logging

import jenkspy
import pandas as pd


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

    x1, y1, x2, y2 = coords
    width = x2 - x1
    height = y2 - y1

    #   get surface covering only the first line of the title
    surface = width * (height / len(split_text))

    if char_count > 0:
        ratio = surface / char_count
    else:
        ratio = 0

    return ratio


def applyJenks(ratios, n_classes=3):
    """Applies Jenks Natural Breaks Optimization to find similar titles

    Args:
        ratios (list): list containing surface/ char count ratio for each title
        n_classes (int): number of classes used, should be 3 for heading, sub heading, and outliers

    Returns:
        list with labels for each title at corresponding index
    """

    #   add id for re-identification, [[id, ratio value], ...]
    ratios_id = [[i, ratio] for i, ratio in enumerate(ratios)]

    #   sort by ratio
    ratios_id = sorted(ratios_id, key=lambda ratio: ratio[1])

    values = [ratio[1] for ratio in ratios_id]
    breaks = jenkspy.jenks_breaks(values, nb_class=n_classes)
    labels = pd.cut(values, bins=breaks, labels=[0, 1, 2], include_lowest=True)

    #   reorder title ratios with re-identification
    reordered_labels = labels.copy()
    for i, label in enumerate(labels):
        reordered_labels[ratios_id[i][0]] = label

    return reordered_labels


def mapJenksLabels(ratios, labels, label_map={0: "heading", 1: "sub", 2: "random"}):
    """Maps Jenks Algorithm label output to title categories

    Args:
        ratios (list): ratio value for each title
        labels (list): a list where each label corresponds to a title object
        label_map (dict): label map containing indexed title categories

    Returns:
        a title category label list
    """

    original_map = {0: [], 1: [], 2: []}

    for i, label in enumerate(labels):
        #   save original indexes of labels
        original_map[label].append(i)

    #   find outlier label by min number of samples
    outlier_id = 0
    smallest_len = len(original_map[0])
    for i in range(len(original_map)):
        if len(original_map[i]) <= smallest_len:
            label_map[i] = "outliers"
            outlier_id = i

    #   start with random outlier ratio
    random_outlier_ratio = ratios[original_map[outlier_id][0]]

    #   gets index that's not outlier id
    if outlier_id in [0, 1]:
        random_id = [0, 1]
        random_id.remove(outlier_id)
        random_id = random_id[0]

    else:
        random_id = 0

    min_diff = abs(ratios[original_map[random_id][0]] - random_outlier_ratio)
    biggest_ratio = 0
    min_diff_id = 0

    #   find 'heading' & 'sub' labels indexes
    for i in range(len(original_map)):
        if i != outlier_id:
            random_label_ratio = ratios[original_map[i][0]]
            ratio_diff = abs(random_label_ratio - random_outlier_ratio)

            if random_label_ratio >= biggest_ratio:
                biggest_ratio = random_label_ratio

                #   map remaining labels
                label_map[i] = "heading"
                for j in range(len(original_map)):
                    if j not in [outlier_id, i]:
                        label_map[j] = "sub"

            if ratio_diff <= min_diff:
                min_diff = ratio_diff
                min_diff_id = i

        #   merge outliers with closest neighbouring category
        if i == len(original_map) - 1:
            label_map[outlier_id] = label_map[min_diff_id]

    #   map every label in original label list
    mapped_labels = []
    for label in labels:
        mapped_labels.append(label_map[label])

    return mapped_labels


def sectionByRatio(ratios, filename):
    """Finds sections based on ratio (bounding box surface / char count)

    Args:
        ratios (list): char count over surface ratio for each title
        filename (string): document filename for error message

    Returns:
        a label list where each item corresponds to a title object
    """

    n_classes = 3  #   heading, sub heading, outliers
    mapped_labels = []

    if len(ratios) > n_classes:
        labels = applyJenks(ratios, n_classes)
        mapped_labels = mapJenksLabels(ratios, labels)
    else:
        logging.warning(
            f"Not enough titles detected in '{filename}' to find natural breaks, using only chapter numbering. Minimum number of titles is {n_classes}"
        )

    return mapped_labels


def getPageColumns(layout):
    """Gets the number of columns used in the layout

    Args:
        layout (layoutparser.Layout): document objects

    Returns:
        the number of columns in given layout
    """

    cols = 1
    for i, b1 in enumerate(layout):
        b1_x1, b1_y1, b1_x2, b1_y2 = b1.block.coordinates
        # cx1 =  b1_x1 + ((b1_x2 - b1_x1) / 2)   #   TODO: remove this

        l2 = layout.copy()
        l2.pop(i)  #  exclude current block
        for j, b2 in enumerate(l2):
            b2_x1, b2_y1, b2_x2, b2_y2 = b2.block.coordinates
            overlap = not (b2_y1 > b1_y2 or b1_y1 > b2_y2)
            neighbours = b1_x2 < b2_x1
            if overlap and neighbours:
                if cols < 2:
                    cols = 2

                l3 = l2.copy()
                l3.pop(j)
                for y, b3 in enumerate(l3):
                    b3_x1, b3_y1, b3_x2, b3_y2 = b3.block.coordinates
                    overlap = not (b3_y1 > b2_y2 or b2_y1 > b3_y2)
                    neighbours = b2_x2 < b3_x1
                    if overlap and neighbours:
                        #   max number of cols is 3
                        return 3
    return cols


def prioritizeLabels(layouts, cn_labels, r_labels):
    """Compares segmentation method outputs to get best of both worlds

    Args:
        layouts (list): list of Layout instances
        cn_labels (list): chapter numbering section segmentation output labels
        r_labels (list): natural breaks section segmentation output labels

    Returns:
        a list containing the combined labels
    """

    labels = []
    title_id = 0

    for layout in layouts:
        for b in layout:
            if b.type.lower() == "title":
                #   prioritize numbered chapter headings
                text = b.text.strip()
                is_digit = False
                if len(text) > 0:
                    c = text[0]
                    is_digit = c.isdigit()

                isnt_empty = len(r_labels) > 0
                is_heading = False
                if isnt_empty:
                    is_heading = r_labels[title_id] == "heading"

                if cn_labels[title_id] == "heading":
                    labels.append("heading")

                elif not is_digit and isnt_empty and is_heading:
                    labels.append("heading")
                else:
                    labels.append("sub")

                title_id += 1

    return labels


def sectionByChapterNums(layouts):
    """Finds sections based on the chapter numbering

    Args:
        layouts (list): list of Layout instances

    Returns:
        a label list where each item corresponds to a title object
    """

    curr_chapter = None
    labels = []
    for layout in layouts:
        for b in layout:
            if b.type.lower() == "title":
                chapter = ""
                text = b.text.strip()
                for t in text:
                    if t.isdigit():
                        chapter += t
                    else:
                        break

                #   compare first digit of title w/ previous
                if len(chapter) > 0 and chapter != curr_chapter:

                    curr_chapter = chapter
                    labels.append("heading")
                else:
                    labels.append("sub")

    return labels


def getTitleRatios(layouts):
    """Gets the ratio (bounding box surface / char count) for each title

    Args:
        layouts (list): list of Layout instances

    Returns:
        list containing ratio for each title
    """

    ratios = []
    for layout in layouts:
        for b in layout:
            if b.type.lower() == "title":
                t = b.text
                ratio = getRatio(b.block.coordinates, t)
                ratios.append(ratio)

    return ratios
