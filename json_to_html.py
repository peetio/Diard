#    Document Image Analysis HTML Result Visualization
import argparse
import os

from modules.utils import filterLayoutJsons
from modules.visualization import getBlocksFromJson, writeToHtml


def main():
    parser = argparse.ArgumentParser(description="JSON to HTML conversion visualization script")
    parser.add_argument(
        "--json_dir",
        help="Path to directory containing JSON files",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        help="Name of the output visualization directory.",
        type=str,
        required=True
    )

    args = parser.parse_args()

    #   Get directory (original PDF document) filenames
    json_dirs = os.listdir(args.json_dir)            
       
    for pdf_filename in json_dirs:
        jsons = os.listdir(args.json_dir + pdf_filename)

        #   Filter out non-json & image data JSON files
        jsons = filterLayoutJsons(jsons)    
        jsons.extend(["joined.json", "sectioned.json"])

        for json_filename in jsons:
            json_path = args.json_dir+pdf_filename+'/'+json_filename
            blocks = getBlocksFromJson(json_path)

            html_dir = args.output_dir + pdf_filename
            writeToHtml(blocks, html_dir, json_filename)

if __name__ == '__main__':
    main()

