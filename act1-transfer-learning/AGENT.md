# Transfer Learning Activity — Agent Instructions (Python Notebook)

## Objective

Implement an image classification pipeline using transfer learning with ResNet50 in a Python notebook environment (Google Colab or Jupyter Notebook).

The notebook must:
- Load a custom image dataset
- Preprocess images correctly for ResNet50
- Train a classification head on top of a frozen pretrained model
- Fine-tune part of the pretrained network
- Visualize and analyze training performance

---

# Technical Stack

Use:
- Python
- TensorFlow / Keras
- Matplotlib
- NumPy

Primary model:
- ResNet50 pretrained on ImageNet

---

# Dataset Structure

Assume the dataset is organized as:

dataset/
    train/
        class_a/
        class_b/
        ...
    
    validation/
        class_a/
        class_b/
        ...

Optional:
dataset/
    test/
        class_a/
        class_b/

Folder names represent class labels.

---

# Notebook Structure

The notebook should contain the following sections in order.

---

# Section 1 — Imports

Import all required libraries.

Required modules:
- tensorflow
- keras
- numpy
- matplotlib.pyplot
- pathlib or os

Required Keras utilities:
- ResNet50
- preprocess_input
- GlobalAveragePooling2D
- Dense
- Dropout
- Model
- ImageDataGenerator OR image_dataset_from_directory

---

# Section 2 — Dataset Configuration

Define:
- dataset paths
- image size
- batch size
- epochs

Use:
- image size = (224, 224)

Reason:
ResNet50 requires 224×224 RGB images.

---

# Section 3 — Dataset Loading

Load datasets using either:
- ImageDataGenerator
OR
- image_dataset_from_directory

Requirements:
- automatic label generation from folder names
- RGB images only
- resize to 224×224
- batching enabled
- shuffling enabled for training

Training dataset:
- apply preprocess_input

Validation dataset:
- apply preprocess_input

If using ImageDataGenerator:
- use preprocessing_function=preprocess_input

---

# Section 4 — Verify Dataset

Display:
- class names
- number of classes
- sample batch shape

Optionally:
- visualize several images

Confirm:
- shape should be:
(batch_size, 224, 224, 3)

---

# Section 5 — Load Pretrained ResNet50

Load ResNet50 with:
- weights='imagenet'
- include_top=False

Freeze the pretrained layers:

base_model.trainable = False

Explanation:
The pretrained convolutional layers act as a feature extractor.

---

# Section 6 — Build Custom Classification Head

Construct a new model on top of ResNet50.

Recommended architecture:
- ResNet50 base
- GlobalAveragePooling2D
- Dense layer with ReLU
- Dropout
- Final Dense output layer with softmax

Output layer:
- units = number of classes

Use softmax for multiclass classification.

---

# Section 7 — Compile Model

Compile using:
- optimizer='adam'
- loss='categorical_crossentropy'
OR
- sparse_categorical_crossentropy

Metrics:
- accuracy

---

# Section 8 — Initial Training (Frozen Base)

Train only the classification head.

IMPORTANT:
Store training output in:

history = model.fit(...)

This variable is required later for plotting.

Suggested:
- 5–10 epochs initially

Monitor:
- training accuracy
- validation accuracy
- training loss
- validation loss

---

# Section 9 — Plot Training History

Generate plots for:
- accuracy
- loss

Plot:
- training vs validation curves

Required plots:
1. Accuracy vs Epoch
2. Loss vs Epoch

Interpretation goals:
- detect convergence
- detect overfitting
- compare training vs validation performance

---

# Section 10 — Fine-Tuning

Enable partial retraining of ResNet50.

Procedure:
1. Unfreeze selected deeper layers
2. Keep earlier layers frozen
3. Recompile with lower learning rate

Recommended:
- unfreeze last 20–50 layers

Use:
- Adam optimizer with small learning rate
  Example:
  1e-5

Reason:
Fine-tuning should make small adjustments to pretrained weights.

---

# Section 11 — Fine-Tuning Training

Train again after unfreezing.

Store result separately:

fine_tune_history = model.fit(...)

Use fewer epochs than initial training.

Compare:
- before fine-tuning
- after fine-tuning

---

# Section 12 — Preprocessing Experiment

Run an additional experiment WITHOUT preprocess_input.

Purpose:
Demonstrate the importance of ResNet50 preprocessing.

Procedure:
- remove preprocess_input
- retrain briefly
- compare metrics

Expected outcome:
- unstable learning
- poor convergence
- lower accuracy

---

# Section 13 — Result Analysis

Analyze:
- effect of preprocessing
- effect of transfer learning
- effect of fine-tuning
- signs of overfitting

Overfitting indicators:
- training accuracy rises sharply
- validation accuracy stagnates or decreases
- validation loss increases

Healthy generalization:
- training and validation curves improve together

---

# Implementation Guidelines

## Use GPU Runtime

If using Google Colab:
- enable GPU runtime

Recommended:
Runtime → Change Runtime Type → GPU

---

## Random Seeds

Set random seeds for reproducibility.

Suggested:
- TensorFlow seed
- NumPy seed

---

## Training Recommendations

If dataset is small:
- use augmentation

Possible augmentations:
- rotation
- zoom
- horizontal flip
- width/height shift

Do NOT apply augmentation to validation data.

---

## Performance Considerations

If memory issues occur:
- reduce batch size

Suggested:
- batch size = 16 or 32

---

# Expected Outputs

The notebook should produce:
- model summary
- training logs
- accuracy plots
- loss plots
- evaluation metrics
- observations from preprocessing and fine-tuning

---

# Recommended Workflow Summary

1. Load dataset
2. Preprocess images
3. Load pretrained ResNet50
4. Freeze base model
5. Train classification head
6. Plot results
7. Fine-tune deeper layers
8. Retrain
9. Compare performance
10. Analyze overfitting and preprocessing effects