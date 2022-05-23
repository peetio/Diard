import argparse
import json
import pandas as pd
import logging
from modules.exceptions import InputJsonStructureError

def get_layout_json_dims(l, dims=0):
        """Recursive function to find number of dimensions (n of rows) in a list

        Args:
            l (list): list of which you want to know the number of rows
            dims (int): the current dimensionality or number of rows in the "list"

        Returns:
            the dimensionality (number of rows) in the "list"
        """

        if not type(l) == list:
            return dims

        return get_layout_json_dims(l[0], (dims + 1))

def load_layout_from_json(json_path):
    """Loads layout blocks (document objects) from JSON file
        variation of Document 'load_layout_from_json' method

    Args:
        json_path (str): JSON filename
    """
    with open(json_path, "r") as f:
        layout_jsons = json.load(f)

    #   single or joined layouts file check based on list dimensionality
    dims = get_layout_json_dims(l=layout_jsons)
    blocks = []
    if dims == 1:
        logging.info("You're loading a JSON file representing a single document page.")
        blocks = layout_jsons
        layouts.append(blocks)

    elif dims == 2:
        logging.info("You're loading a JSON file representing a whole document.")
        for layout_json in layout_jsons:
            blocks.extend(layout_json)
    else:
        raise InputJsonStructureError(filename)

    return blocks

def main():
    parser = argparse.ArgumentParser(description="Diard pipeline script")
    parser.add_argument(
        "--json-path",
        help="Path to JSON file you want to load the data from",
        required=True
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s: %(message)s", level=logging.NOTSET
    )
    logging.disable(logging.DEBUG)

    blocks = load_layout_from_json(args.json_path)

    #   further processing
    #   e.g., print out the data of each document object 
    for block in blocks:
        is_table = block['type'] == 'table'
        is_dict = type(block['content']) == dict
        print("\nBlock:")
        for k, v in block.items():
            if is_table and is_dict and k == 'content':
                #   resulting variable is not used in this example
                df = pd.DataFrame.from_dict(v)
                #   do what you want with the tabular data...
            elif k == 'content':
                v = v.strip()

            try:
                print(f"{k}: {v}")
            #   catch key / value pair if section segmentation wasn't used
            except KeyError:
                pass

if __name__ == "__main__":
    main()
