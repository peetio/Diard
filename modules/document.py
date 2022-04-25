#   Document class definition
from pdf2image import convert_from_path

from modules.exceptions import DocumentFileFormatError, UnsetAttributeError



class Document():

    def __init__(self, path, predictor=None, metadata=None):
        """Creates instance of Document object
        
        Args:
            path (str): path to document
            predictor (detectron2.engine.defaults.DefaultPredictor, optional): configured default predictor instance
            metadata (detectron2.data.Metadata, optional): dataset metadata
        """

        name = path.split('/')[-1]
        file_format = name.split('.')[-1]
        if file_format not in ['pdf', 'docx']:
            raise DocumentFileFormatError(name)

        self.name = name
        self.file_format = file_format
        self.path = path

        self.metadata = metadata
        self.predictor = predictor



    def convertToImages(self):
        """Converts each page of a document to images"""

        self.images = convert_from_path(self.path) 



    def extractLayout(self, segment_sections=False, visuals_path=None):
        """Extracts layout from document images
        
        Args:
            segment_sections (bool): if True sections are segmented
            visuals_path (str): path to visuals output dir. Defaults to None
        """

        if None in [self.predictor, self.metadata]:
            raise UnsetAttributeError("extractLayout()", ["predictor", "metadata"])
            
        #   TODO: 25 Apr : implement layout extraction and define what the "layout detection object" will look like!

