# Medical-Image-Analysis-with-CNN
Built a custom Convolutional Neural Network (CNN) in Python using TensorFlow/Keras to classify chest X-ray images as NORMAL or PNEUMONIA. Trained and evaluated the model on Kaggle’s chest X-ray dataset, achieving ~91% accuracy. Implemented data augmentation and visualized results. Deployed predictions on new images with interpretation.

**Chest X-ray Binary Classification: NORMAL vs PNEUMONIA**

## Project Overview

This project is an end-to-end implementation of a Convolutional Neural Network (CNN) for binary classification of chest X-ray images. The model predicts whether an image shows signs of **PNEUMONIA** or is **NORMAL**. It is built using TensorFlow/Keras and trained on a publicly available medical image dataset from Kaggle.

The project includes:
- Data preprocessing and augmentation
- Custom CNN model architecture
- Model training, evaluation, and visualization
- Prediction on new X-ray images
- Clean, reproducible code (Kaggle-ready & GitHub-hosted)

---

## Dataset

**Source**: [Chest X-Ray Images (Pneumonia) | Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

- Classes: `NORMAL`, `PNEUMONIA`
- Total images: ~5,800
- Format: `.jpeg` chest X-ray scans
- Pre-split into `train`, `val`, and `test` directories

---

## Model Architecture

A custom CNN built from scratch:

- `Conv2D` layers with ReLU activations
- `MaxPooling2D` for downsampling
- `Dropout` layer to prevent overfitting
- `Dense` layers for classification
- `Sigmoid` output for binary classification

- python
Input → Conv2D → MaxPool → Conv2D → MaxPool → Conv2D → MaxPool → Flatten → Dense → Output


