import logging
import os
import warnings

from modules.document import Document
from modules.layoutdetection import LayoutDetection


def main():
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s: %(message)s", level=logging.NOTSET
    )
    logging.disable(logging.DEBUG)
    #   suppressing PyTorch & Detectron warnings
    warnings.filterwarnings(
        "ignore", category=UserWarning
    )  # NOTE: comment out for debugging

    config_path = "./resources/model_configs/cascade/cascade_dit_large.yaml"
    weights_path = "./resources/weights/publaynet_dit-l_cascade.pth"

    ld = LayoutDetection(
        cfg_path=config_path,
        weights_path=weights_path,
        batch_size=1,
        workers=1,
        threshold=0.75,
    )

    predictor = ld.getPredictor()
    metadata = ld.getMetadata()
    source_dir = "./resources/pdfs/"

    #   process single pdf
    filenames = ["example.pdf"]

    #   process multiple pdfs
    # filenames = os.listdir(source_dir)

    for filename in filenames:
        doc_path = source_dir + filename
        lang = "deu"  #   language used in most documents
        langs = ["eng", "fra", "deu"]  #   only if lang_detect=True

        doc = Document(
            doc_path,
            predictor=predictor,
            metadata=metadata,
            lang=lang,
            lang_detect=True,
            langs=langs,
        )

        #   extract & save layout
        doc.docToImages()
        doc.extractLayouts(visualize=True, segment_sections=True)
        doc.orderLayouts()
        doc.saveLayoutsAsJson()
        doc.saveLayoutsAsHtml()

if __name__ == "__main__":
    main()
