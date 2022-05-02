import logging
import warnings

from modules.document import Document
from modules.utils import initializeModel


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
    predictor, metadata = initializeModel(config_path, weights_path, threshold=0.7)

    #   create Document instance
    document_path = "./resources/pdfs/example.pdf"
    document = Document(document_path, predictor=predictor, metadata=metadata)

    #   extract & save layout
    document.docToImages()
    document.extractLayouts(visualize=True, segment_sections=True)
    document.orderLayouts()
    document.saveLayoutsAsJson()
    document.saveLayoutsAsHtml()    #   TODO: add table of contents on left side of scree


if __name__ == "__main__":
    main()
