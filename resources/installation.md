# Installation Guide
We recommend you to use a virtual environment for the installation. We used virtualenv with the following commands.

```bash
python -m virtualenv ditenv
source ditenv/bin/activate
```

## Step 1: OpenCV

```bash
pip install opencv-contrib-python
```

## Step 2: PyTorch

We used PyTorch v1.10.2 and torchvision v0.11.3 to build the pipeline but PyTorch v1.9.0 and torchvision v0.10.0 were used in the unilm DiT repository by Microsoft. PyTorch versions higher than 1.9.0 should work.

The install command below was used on the authors machine. Depending on your CUDA version, OS, and package manager yours will probably be different. You can get the install command based on your preferences from the [PyTorch get started guide](https://pytorch.org/get-started/locally/)

```bash
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Or if you don't have a CUDA enabled GPU you can install the "CPU version" with PyTorch v1.11.0 and torchvision v0.12.0 which we also tested.

```bash
pip3 install torch==1.11.0+cpu torchvision==0.12.0+cpu torchaudio==0.11.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

## Step 3: Detectron2 

The install command below should work to install the Detectron AI toolkit by Facebook. If you experience any issues you can use the official [installation guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html). Note that the library is hard to install on a Windows machine. For that reason we strongly recommend using Linux or MacOS.

```bash
pip install "detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.5#egg=detectron2"
```

## Step 5: Tesseract

The Tesseract engine is used for text recognition. For more information about the installation you can refer to the [official tessdocs](https://tesseract-ocr.github.io/tessdoc/Installation.html).

```bash
sudo apt install tesseract-ocr-deu
```

## Step 6: Layout Parser

For more information about this DIA toolkit you can refer to the [Layout Parser GitHub repository](https://github.com/Layout-Parser/layout-parser). The two install commands below should suffice if you just want to test the pipeline.

```bash
pip install layoutparser # Base library
pip install "layoutparser[ocr]" # OCR toolkit
```



## STEP 7: Python Libraries

Install the required Python libraries from the root of the repository. You can also manually install them by opening the [requirements.txt](../requirements.txt) file.

```bash
pip install -r requirements.txt
```

