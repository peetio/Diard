import logging
import os
import warnings

from modules.document import Document
from modules.layoutdetection import LayoutDetection
from modules.tables import TableExtractor


def main():
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
        device='cuda', # change to cpu if you don't have CUDA enabled GPU
        batch_size=1,
        workers=1,
        threshold=0.65,
    )

    predictor = ld.get_predictor()
    metadata = ld.get_metadata()
    source_dir = "./resources/pdfs/"

    #   language used most of your documents (ISO 639-3 format)
    lang = "deu"  

    #   only useful if lang_detect=True (all specified language packs should be installed)
    langs = ["eng", "fra", "deu"]  

    #   process single pdf
    # filenames = ["example.pdf"]

    #   process multiple pdfs
    filenames = os.listdir(source_dir)

    for filename in filenames:
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

if __name__ == "__main__":
    main()
