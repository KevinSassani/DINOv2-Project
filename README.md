# DINOv2-Project

This project implements DINOv2 ViT-S/14 model for segmentation and classification tasks, based on the findings by Meta AI research team in 2024 [8]. 

#### Segmentation
The pretrained ViT-S/14 model was fine-tuned and evaluated on the ADE20K dataset for segmentation and the Food-101 dataset for classification. For segmentation, we tested both frozen file: "frozen_DINOV2_segmentation_ADE20k.ipynb" and fully fine-tuned backbones file: DINOV2_segmentation_ADE20k.ipynb.

#### Classification

#### Sample efficiency
The file "Food101_sample_efficiency.ipynb" filters out smaller datasets containing either one or 10 samples of each class, appends a classification head to the backbone and then trains using either full fine-tuning or frozen backbone as well as either one or ten samples per class
