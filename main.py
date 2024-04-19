import json
import cv2
import yaml
import os
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import albumentations as A
from tqdm.auto import tqdm
from transformers import MaskFormerForInstanceSegmentation
from transformers import MaskFormerImageProcessor
from dataset import ImageSegmentationDataset
from fastervit import FasterViT
from train import train


with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

dataset_root = config['DATASET_ROOT']

id2label = config['ID2LABEL']
label2id = config['LABEL2ID']

img_size = config['IMG_SIZE']
batch_size = config['BATCH_SIZE']
num_queries = config['NUM_QUERIES']
dice_weight = config['DICE_WEIGHT']
encoder_type = config['ENCODER_TYPE']
optimizer_conf = config['OPTIMIZER_CONF']
cons_unfreeze = config['CONS_UNFREEZE']
full_unfreeze = config['FULL_UNFREEZE']
unfreeze_interval = config['UNFREEZE_INTERVAL']
unfreeze_ratio = config['UNFREEZE_RATIO']

num_epochs = config['NUM_EPOCHS']
lr = config['LR']
weight_decay = config['WEIGHT_DECAY']

dim = config['DIM']
in_dim = config['IN_DIM']
depths = config['DEPTHS']
window_size = config['WINDOW_SIZE']
ct_size = config['CT_SIZE']
mlp_ratio = config['MLP_RATIO']
num_heads = config['NUM_HEADS']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def coco_to_polygons(annotations, image_folder):
    name_id = {image["id"]: image["file_name"] for image in annotations["images"]}
    polygons = defaultdict(list)
    for ann in annotations["annotations"]:
        polygons[ann["image_id"]].append(np.array(ann["segmentation"][0], dtype=np.int32).reshape(-1, 2))

    # Create empty numpy arrays for each image
    dataset = [{} for _ in range(len(annotations["images"]))]
    # Draw polygons on the image arrays
    for image_id, polygon_list in polygons.items():
        template = np.zeros((annotations["images"][image_id-1]["height"], annotations["images"][image_id-1]["width"], 3), dtype=np.uint8)
        for j, polygon in enumerate(polygon_list, start=1):
            cv2.fillPoly(template, [polygon], (1, j, 0))
        dataset[image_id-1]["annotation"] = template
        image = Image.open(os.path.join(image_folder, f"{name_id[image_id]}")).convert("RGB")
        dataset[image_id-1]["image"] = image

    for i, image in enumerate(dataset):
        if not image:
            print(f"Image {i+1} is missing")
            print()
            del dataset[i]
   
    return dataset


with open(dataset_root+"/train/annotations/train.json") as f:
    data = json.load(f)
    train = coco_to_polygons(data, dataset_root+"/train/images")

with open(dataset_root+"/val/annotations/val.json") as f:
    data = json.load(f)
    val = coco_to_polygons(data, dataset_root+"/val/images")

with open(dataset_root+"/test/annotations/test.json") as f:
    data = json.load(f)
    test = coco_to_polygons(data, dataset_root+"/test/images")


def collate_fn(batch):
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    pixel_mask = torch.stack([example["pixel_mask"] for example in batch])
    class_labels = [example["class_labels"] for example in batch]
    mask_labels = [example["mask_labels"] for example in batch]

    return {"pixel_values": pixel_values, 
            "pixel_mask": pixel_mask, 
            "class_labels": class_labels, 
            "mask_labels": mask_labels
            }


processor = MaskFormerImageProcessor(reduce_labels=True, 
                                     ignore_index=255, 
                                     do_resize=False, 
                                     do_rescale=False, 
                                     do_normalize=False
                                     )


train_transform = A.Compose([
    A.Resize(width=img_size, height=img_size),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Blur(p=0.5),
    A.HueSaturationValue(p=0.5)
])

val_transform = A.Compose([
    A.Resize(width=img_size, height=img_size),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
])

train_dataset = ImageSegmentationDataset(train, 
                                         processor=processor, 
                                         transform=train_transform
                                         )
val_dataset = ImageSegmentationDataset(val, 
                                       processor=processor, 
                                       transform=train_transform
                                       )

train_dataloader = DataLoader(train_dataset, 
                              batch_size=4, 
                              shuffle=True, 
                              collate_fn=collate_fn
                              )

val_dataloader = DataLoader(val_dataset, 
                            batch_size=4, 
                            shuffle=False, 
                            collate_fn=collate_fn
                            )

model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade",
                                                          id2label = id2label,
                                                          label2id = label2id,
                                                          num_queries = num_queries,
                                                          dice_weight = dice_weight,
                                                          ignore_mismatched_sizes=True
                                                          )

if encoder_type == 'ours':
    encoder = FasterViT(dim=dim,
                        in_dim=in_dim,
                        depths=depths, 
                        window_size=window_size, 
                        ct_size=ct_size, 
                        mlp_ratio=mlp_ratio, 
                        num_heads=num_heads
                        )
    model.model.pixel_level_module.encoder = encoder

PARAM_LIST = [model.model.pixel_level_module.encoder.parameters(),
              model.model.pixel_level_module.decoder.parameters(),
              model.model.transformer_module.parameters(),
              model.class_predictor.parameters(),
              model.mask_embedder.parameters(),
              model.matcher.parameters(),
              model.criterion.parameters()]


if __name__ == '__main__':
    train(model=model, 
          device=device, 
          optimizer_conf=optimizer_conf, 
          encoder_type=encoder_type, 
          num_epochs=num_epochs, 
          lr=lr, 
          weight_decay=weight_decay,
          train_dataloader=train_dataloader,
          val_dataloader=val_dataloader,
          dice_weight=dice_weight,
          num_queries=num_queries, 
          encoder=encoder,
          unfreeze_layers=PARAM_LIST,
          unfreeze_ratio=unfreeze_ratio,
          cons_unfreeze=cons_unfreeze,
          full_unfreeze=full_unfreeze,
          unfreeze_interval=unfreeze_interval
          )
