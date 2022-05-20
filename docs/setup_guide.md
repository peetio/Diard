# Setup Guide
We recommend you to use a virtual environment for the installation. We used virtualenv with the following commands.

```bash
python -m virtualenv ditenv
source ditenv/bin/activate
```

## STEP 1 - OpenCV ([pip](https://pypi.org/project/opencv-python/))

```bash
pip install opencv-contrib-python
# or
pip install opencv-python
```

## STEP 2 - PyTorch

We used PyTorch v1.10.2 and torchvision v0.11.3 to build the pipeline but PyTorch v1.9.0 and torchvision v0.10.0 were used in the unilm DiT repository by Microsoft. PyTorch versions higher than 1.9.0 should work.

The install command below was used on the authors machine. Depending on your CUDA version, OS, and package manager yours will probably be different. You can get the install command based on your preferences from the [PyTorch get started guide](https://pytorch.org/get-started/locally/)

```bash
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Or if you don't have a CUDA enabled GPU you can install the "CPU version" with PyTorch v1.11.0 and torchvision v0.12.0 which we also tested.

```bash
pip3 install torch==1.11.0+cpu torchvision==0.12.0+cpu torchaudio==0.11.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

## STEP 3 - Detectron2 

The install command below should work to install the Detectron AI toolkit by Facebook. If you experience any issues you can use the official [installation guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html). Note that the library is hard to install on a Windows machine. For that reason we strongly recommend using Linux or MacOS.

```bash
pip install "detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.5#egg=detectron2"
```

## STEP 4 - PyTesseract

A Tesseract OCR Python wrapper is used for text recognition. For more information about the installation you can refer to the [pytesseract](https://github.com/madmaze/pytesseract) GitHub repo.

But first of all you should install the Tesseract engine. For Linux users the command below should suffice, others might want to look at the [official documentation](https://tesseract-ocr.github.io/tessdoc/Installation.html)

```bash
sudo apt install tesseract-ocr-deu
```

The Python wrapper can simply be installed with pip using the following command.

```bash
pip install pytesseract
```

Next, you need to download additional language packs. You can download the language packs from either the [tessdata](https://github.com/tesseract-ocr/tessdata) or [tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast) repository. Keep in mind that you have to make a speed/accuracy compromise when using the fast packs. 

You can either clone the whole repository or download a single pack. During development the (format=language:abbreviation:packname) English='eng'=eng.traineddata, French='fra'=fra.traineddata, and German='deu'=deu.traineddata lanaguage packs were used. 

Put the language packs in a directory called tessdata and set the TESSDATA_PREFIX environment variable like we do below.

```bash
export TESSDATA_PREFIX=/home/user/tessdata
```

## STEP 6 - Layout Parser

For more information about this DIA toolkit you can refer to the [Layout Parser GitHub repository](https://github.com/Layout-Parser/layout-parser). The two install commands below should suffice if you just want to test the pipeline.

```bash
pip install layoutparser
```

## STEP 7 - Python Libraries

Install the required Python libraries from the root of the repository. Don't forget to source into your virtual environment if applicable. You can also manually install them by opening the [requirements.txt](../requirements.txt) file.

```bash
pip install -r requirements.txt
```

## STEP 8 - Pre-trained Models

After your environment is set up, you should download the pre-trained model weights from [this link](https://layoutlm.blob.core.windows.net/dit/dit-fts/publaynet_dit-l_cascade.pth) (1.4GB) and place it in the './resources/weights/' directory.

### STEP 8.1 - (optional) Table Extraction

A different model is used for table structure recognition. If you want table extraction, another pre-trained model's weights have to be downloaded from [here](https://pubtables1m.blob.core.windows.net/model/pubtables1m_detection_detr_r18.pth). Also put this file in the './resources/weights' directory.

Check out the [examples](./examples.md) for a guide on how to enable table extraction in the pipeline.

@Developers -> looking to add table extraction to your own document image analysis pipeline? You can find the original repository [right here](https://github.com/microsoft/table-transformer).
