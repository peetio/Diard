import os
import logging
import warnings

from modules.document import Document
from modules.utils import initializeModel
from modules.exceptions import DocumentFileFormatError


def main():
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s: %(message)s", level=logging.NOTSET
    )
    logging.disable(logging.DEBUG)

    #   suppressing PyTorch & Detectron warnings
    warnings.filterwarnings(
        "ignore", category=UserWarning
    )  # NOTE: comment out for debugging

    #   initialize model
    config_path = "./resources/model_configs/cascade/cascade_dit_large.yaml"
    weights_path = "./resources/weights/publaynet_dit-l_cascade.pth"
    predictor, metadata = initializeModel(config_path, weights_path, threshold=0.65)

    docs_dir = "./resources/pdfs/"

    #   process single pdf
    #document_paths = docs_dir + "example.pdf"
    
    #   process multiple pdfs
    filenames = os.listdir(docs_dir)

    for filename in filenames:
        #   create Document instance
        doc_path = docs_dir + filename
        print("document path:", doc_path)
        doc = Document(doc_path, predictor=predictor, metadata=metadata)

        #   extract & save layout
        doc.docToImages()
        #   FIXME: when two figuers are detected right after each other one of them is not saved correctly
        #   TODO: remove overlapping objects of type type! not possible
        doc.extractLayouts(visualize=True, segment_sections=True)
        doc.orderLayouts()
        doc.saveLayoutsAsJson()
        doc.saveLayoutsAsHtml()

if __name__ == "__main__":
    main()
