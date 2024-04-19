# Faster MaskFormer for Coronary Artery Decease Instance Segmentation
 <img width="2066" alt="Welcome to FigJam" src="https://github.com/VGMitkin/CV703_final/assets/91109627/bd7b2072-77f3-43ac-9cbc-4b9092396fb7">


 ## Abstract
Coronary artery disease (CAD) remains a leading cause of global mortality, necessitating efficient diagnostic techniques. X-ray Coronary Angiography (XCA) is a commonly used imaging modality for CAD identification, but it presents challenges such as non-uniform illumination and low contrast. Medical image segmentation aids in CAD detection, where convolutional neural networks (CNNs) are commonly used. However, CNNs have limitations in modeling global relationships within data. To address this, transformer architectures, originally designed for natural language processing, have been adapted for medical imaging tasks. We propose a hybrid model inspired by recent advancements that combine CNNs and transformers to perform better sanitation. Our model architecture, FasterUNETR, performs better than traditional methods, particularly on small datasets. Furthermore, incorporating Mask2Former's masked attention method enables instance segmentation, enhancing the model's versatility for various image grouping tasks, including those in medical imaging.


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
conda activate flash
pip install -r requirements.txt
```

## To run the training
- Config config.yaml file with your parameters
- Start the training by running:
```
python3 main.py 
```
- or run 

```
sh train.sh
```
## To run the testing

- Start the testing by running:
```
python3 test.py
```
- or run 

```
sh test.sh
```



## Team
- Maxim Popov
- Vladislav Mitkin
- Arsen Abzhanov
