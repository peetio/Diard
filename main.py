import argparse
import logging
import os
import warnings

from modules.document import Document
from modules.layoutdetection import LayoutDetection
from modules.tables import TableExtractor


def main():
    parser = argparse.ArgumentParser(description="Diard pipeline script")
    parser.add_argument(
        "--overwrite",
        help="Reprocess and overwrite previously processed documents",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--skip-failures",
        help="Skips files that cannot be processed instead of exiting program",
        action="store_true",
        default=False
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s: %(message)s", level=logging.NOTSET
    )
    logging.disable(logging.DEBUG)
    #   suppressing PyTorch & Detectron warnings
    warnings.filterwarnings(
        "ignore", category=UserWarning
    )  # NOTE: comment out for debugging

    ld_config_path = "./resources/model_configs/cascade/cascade_dit_large.yaml"
    ld_weights_path = "./resources/weights/publaynet_dit-l_cascade.pth"

    ld = LayoutDetection(
        cfg_path=ld_config_path,
        weights_path=ld_weights_path,
        device='cuda', # change to 'cpu' if you don't have CUDA enabled GPU
        batch_size=4,
        workers=1,
        threshold=0.65,
    )

    predictor = ld.get_predictor()
    metadata = ld.get_metadata()
    source_dir = "./resources/pdfs/"
    output_dir = "./output/"

    #   language used in most of your documents (ISO 639-3 format)
    lang = "deu"  

    #   only useful if lang_detect=True (all specified language packs should be installed)
    langs = ["eng", "fra", "deu"]  

    #   process single pdf
    # filenames = ["example.pdf"]

    #   process multiple pdfs
    filenames = os.listdir(source_dir)
    
    #   filter out previously processed documents
    if not args.overwrite:
        processed_files = os.listdir(output_dir)
        filenames = [fn for fn in filenames if '.'.join(fn.split('.')[:-1]) not in processed_files]

    for filename in filenames:
        try:
            doc_path = source_dir + filename

            doc = Document(
                doc_path,
                predictor=predictor,
                metadata=metadata,
                lang=lang,
                lang_detect=True,
                langs=langs
            )

            #   extract & export layouts
            doc.doc_to_images()
            doc.extract_layouts(visualize=True, segment_sections=True)
            doc.order_layouts()
            doc.save_layouts_as_json()
            doc.save_layouts_as_html()

        except Exception as ex:
            if not args.skip_failures:
                logging.warning("Exiting")
                raise ex
            logging.warning(f"Could not process '{filename}'. Exception: {ex}")

if __name__ == "__main__":
    main()
