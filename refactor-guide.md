# Refactoring Guide
This repository was initially a copy of the dia-prototype-v2 repository. Now we are going to refactor it before publicly releasing the code. There are probably a bunch of things that can be improved, and this file is where we will be noting everything down.

## The Pipeline

1. [main.py] -> parses arguments
2. [main.py] -> initialization of variables related to the 
  * initializing the model can be done with a single function call (i.e., initializeModel() and it should return the model variable)
  * do we attach the above to a class, can we?
2. [main.py] -> get all the pdfs in a directory
  * we could make a Document object with a filepath and all the operations that need to be performed in order to extract all the information from it and anything that has to do with visualization and evaluation. In the example we will loop over the files found in a directory, but this doesn't have to be the case for every application. Some people just want to specify the path of a single pdf file and then perform all the necessary operations on it.

  * document = Document(")

  main.py 
  """
    model_config_path = "./resources/model_configs/cascade/cascade_dit_large.yaml"
    weights_path = "./resources/weights/publaynet_dit-l_cascade.pth"
    weights = ["MODEL.WEIGHTS", weights_path]

    predictor, metadata = initializePredictor(model_config_path, weights)  # NOTE: we called weights -> opts and metadata -> md in the main.py

    path_to_document = "./resources/pdfs/example.pdf"
    document = Document(path_to_document) # When creating the instance the filename should be .docx or .pdf, throw exception otherwise, store the filename without extension for further use

    document.convertToImages()  # NOTE: you should also add support for .docx (extension should be deduced from name)
    visuals_path = "./output/filename/visuals/" # NOTE: path to output visuals, if not set the extractLayout method will not visualize the output
    document.extractLayout(predictor, metadata, segment_sections=True, visuals_path=visuals_path)  # visualize=True needs a filepath to store visualizations, should already set extracted data but not order / sort it, that we should leave up to the user to decide!

    # NOTE: the function that sets the data is different for each type of layout object, which layout objects (label_map) are in the document should be the same as that loaded from the configuration file or something (or wherever the label_map is specified)

    document.orderLayout()  # NOTE: some attributes are added here probably that are used in the export of JSONs and other stuff. We should make sure that export is not dependend upon ordered layouts anymore! (if this is the case)

    # NOTE: a functions should also be added to load anything that we export back into an instance of Document class
    
    jsons_path = './output/filename/jsons'
    document.saveAsJson(jsons_path=jsons_path, individual=True, join=True) # NOTE: this should export the layout to JSON files with the options which are all set to True by default. The user can specify the path, but if not specified it will just be outputted to './output/filename/jsons',- where the files can be found will also be logged to the screen using the Python logging lib

    htmls_path= './output/filename/htmls/'
    document.saveAsHtml(htmls_path)

    # NOTE: there should also be a function to join the JSONs, only if the jsons are joined and ordered the sections can be segmented
  """

