# Comparative Analysis of Deep Learning Models for Waste Classification

## Project Overview
This repository contains the results and core methodology of a deep learning project focused on automated waste classification for autonomous recycling systems. We evaluated `ResNet50`, `MobileNetV2`, `InceptionV3`, and `VGG16` using transfer learning on a dataset of 15,150 images across 12 categories. 
## Dataset
The models were trained and evaluated using the [Garbage Classification Dataset](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) from Kaggle, which contains 15,150 images across 12 distinct classes (paper, cardboard, plastic, metal, etc.).
The goal was to identify the most efficient model for resource-constrained Edge AI environments (e.g., smart waste bins).

## Key Performance Indicators
Based on our experimental training, `MobileNetV2` emerged as the optimal architecture due to its high accuracy and low computational footprint.

| Model Architecture | Test Accuracy | Note |
| :----------------- | :----------- | :--- |
| **MobileNetV2**    | **91.44%**   | **Recommended for Edge AI (Best Accuracy / Low Memory Cost)** |
| InceptionV3        | 88.75%       | Balanced performance |
| VGG16               | 81.67%       | Moderate accuracy, higher memory footprint |
| ResNet50           | 46.42%       | Suboptimal convergence within frozen training limits |

## Full Report & Team Contributors
The original, comprehensive academic report (in Turkish) including detailed confusion matrices, literature review, and full team contributor credits is included in this repository as a PDF file.

## Repository Contents
*   `Atık Sınıflandırmada Derin Öğrenme Performans Analizi.pdf`: The original comprehensive project report.
*   `train_mobilenetv2.py`: The core Python script demonstrating the transfer learning setup, frozen layers, and training loop for the winning MobileNetV2 architecture.
