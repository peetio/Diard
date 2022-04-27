import logging
import warnings

from modules.utils2 import initializeModel
from modules.document import Document

def main():
    logging.basicConfig(format='%(asctime)s | %(levelname)s: %(message)s', level=logging.NOTSET)
    logging.disable(logging.DEBUG)

    # Suppressing PyTorch & Detectron warnings 
    warnings.filterwarnings("ignore", category=UserWarning) # NOTE: comment out for debugging

    document_path = './resources/pdfs/example.pdf'
    document = Document(document_path)
    
    document.loadLayoutFromJson('example')
    document.orderLayouts()
    print(document.layouts)



if __name__ == "__main__":
    main()

