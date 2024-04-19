# Faster MaskFormer for Coronary Artery Decease Instance Segmentation
 <img width="2066" alt="Welcome to FigJam" src="https://github.com/VGMitkin/CV703_final/assets/91109627/bd7b2072-77f3-43ac-9cbc-4b9092396fb7">


 ## Abstract
Coronary artery disease (CAD) remains a leading cause of global mortality, necessitating efficient diagnostic techniques. X-ray Coronary Angiography (XCA) is a commonly used imaging modality for CAD identification, but it presents challenges such as non-uniform illumination and low contrast. Medical image segmentation aids in CAD detection, where convolutional neural networks (CNNs) are commonly used. However, CNNs have limitations in modeling global relationships within data. To address this, transformer architectures, originally designed for natural language processing, have been adapted for medical imaging tasks. Inspired by recent advancements like FasterViT and MaskFormer, we introduce Faster MaskFormer, a hybrid model leveraging the strengths of both architectures. Our model exhibits superior performance in semantic and instance segmentation tasks on the ARCADE dataset compared to traditional transformer-based models like MaskFormer-Swin. Furthermore, it demonstrates reduced computational complexity and training time, making it a promising solution for CAD diagnosis.


## Dataset 
The ARCADE challenge dataset, namely "stenosis," which comprises 1500 annotations for stenosis detection and instance segmentation, and "syntax," which comprises another 1500 images for individual vessel classification and segmentation with 25 classes, which will also be converted to more general semantic segmentation task
```
/ARCADE
|
|--- train/
      |--- annotations/
      |------------train.json
      |--- images/
      |------------1.png
      |------------2.png
      ...
      
|--- validation/
|--- test/
```

## Installation

Git clone the repository

```
gh repo clone VGMitkin/Faster_MaskFormer
cd Faster_MaskFormer
```

Create conda enviromnent 

```
conda create -n fastermaskformer python=3.9
conda activate fastermaskformer
pip install -r requirements.txt
```

## To run the training

- Download dataset from URL: (https://zenodo.org/records/10390295)
- Download model weights (https://drive.google.com/file/d/1UUQ0SkOnW5XC9zfg52a0l0tZYsRgEw_x/view?usp=drive_link)
- Extract files
- Config config.yaml file with your parameters. Set root to the extracted files in `DATASET_ROOT` argument
- Start the training by running:
```
python3 main.py 
```
- or run 

```
sh train.sh
```
## To run the testing

To see test results use file `test_arcade_syntax.ipynb`

## Results Example

![fastformer_syntax_overlayed_true](https://github.com/VGMitkin/Faster_MaskFormer/assets/91109627/acbadc6a-a170-42a4-a5ec-4b8ab97020d2)
![fastformer_syntax_overlayed_pred](https://github.com/VGMitkin/Faster_MaskFormer/assets/91109627/a85bfa83-5698-4554-b6fa-7c0f3d628198)

                                      Ground Truth                                                                       Prediction

## Team
- Maxim Popov
- Vladislav Mitkin
- Arsen Abzhanov
