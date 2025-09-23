# My report for how AlexNet works goes here:

# Use lot of pictures and links if needed.
# Explain in a way that assumes that the reader doesn't know about the subject.

# What is alexnet? Explain from scratch to new data students. Full details. Don't assume that they know anything about the subject. Explain in full detail. 


# Lab Report: Understanding and Applying AlexNet in Paleontology

## Abstract
This lab report introduces **AlexNet**, a foundational convolutional neural network (CNN) architecture that revolutionized computer vision in 2012. The report is written for students with no prior experience in machine learning or neural networks. It explains the network step by step, introduces its key components, and demonstrates how its design enables computers to automatically recognize images. To contextualize the theory, the report applies AlexNet concepts to paleontology, specifically in the study and classification of dinosaurs and fossils.

---

## Introduction
In the early 2010s, researchers faced challenges in teaching computers to recognize images. Before deep learning, most systems relied on handcrafted features, where human experts defined what patterns to look for in images. This approach limited accuracy and adaptability.

The breakthrough came in 2012 with **AlexNet**, designed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton. By winning the ImageNet Large-Scale Visual Recognition Challenge (ILSVRC) by a significant margin, AlexNet demonstrated that **deep convolutional neural networks**, when combined with large datasets and graphics processing units (GPUs), could outperform traditional methods. Since then, AlexNet has been considered a starting point for modern computer vision.

For paleontology, AlexNet provides a framework for automatically classifying images of fossils, bones, or dinosaur remains, offering new tools for research, cataloging, and fieldwork.

---

## Background

### What is an Image?
A digital image is a grid of numbers called **pixels**. For a color image, each pixel has three values: red, green, and blue (RGB). A computer "sees" only numbers, not objects.

### What is a Neural Network?
A neural network is a collection of **neurons** (simple mathematical functions). Each neuron receives numbers, performs a weighted sum, and passes the result through an activation function. When many neurons are arranged in **layers**, the network can represent complex relationships.

### Why Convolutions?
In images, patterns such as edges and textures are **local**. Convolutional layers use small filters that slide across the image to detect these patterns. This allows the network to recognize shapes regardless of their position.

---

## AlexNet Architecture

### Overview
AlexNet processes an input image through several stages:

1. **Input**: A color image (usually resized to 224×224 pixels).
2. **Convolutional Layers**: Detect edges, textures, and shapes.
3. **Pooling Layers**: Reduce image size, keeping important features.
4. **Fully Connected Layers**: Combine features to make decisions.
5. **Output**: Probabilities for each possible class.

### Key Features of AlexNet
- **ReLU activation**: A nonlinearity defined as `max(0, x)`. It allows faster training compared to older functions.
- **Dropout**: Randomly disables neurons during training to reduce overfitting.
- **Data augmentation**: Artificially increases dataset size by flipping, cropping, and altering colors of images.
- **GPU training**: Accelerated training, making it practical for large datasets.
- **Parameter size**: Approximately 60 million trainable parameters.

### Layer-by-Layer Structure
1. **Conv1**: 96 filters, 11×11 size, stride 4 → detects broad edges and colors.
2. **Conv2**: 256 filters, 5×5 size → detects textures and curved shapes.
3. **Conv3**: 384 filters, 3×3 size → detects more complex patterns.
4. **Conv4**: 384 filters, 3×3 size → refines higher-level features.
5. **Conv5**: 256 filters, 3×3 size → detects object parts.
6. **FC6**: Fully connected with 4096 neurons.
7. **FC7**: Fully connected with 4096 neurons.
8. **FC8**: Final fully connected layer with outputs equal to the number of classes.

---

## Training Procedure
1. **Data preparation**: Images are labeled by class (e.g., femur, rib, vertebra).
2. **Forward pass**: Input → convolutional layers → fully connected layers → output scores.
3. **Loss computation**: Compare predictions to true labels using cross-entropy loss.
4. **Backpropagation**: Compute gradients of the loss with respect to network weights.
5. **Weight updates**: Adjust weights using stochastic gradient descent (SGD).
6. **Repeat**: Iterate over the dataset for many epochs until accuracy improves.

---

## Applications in Paleontology

### Example 1: Classifying Dinosaur Bones
AlexNet can be trained on labeled images of dinosaur bones. For instance:
- Input: photo of a bone.
- Output: prediction (e.g., "Tyrannosaurus femur").
The model learns which shapes correspond to different bones without manual rules.

### Example 2: Detecting Fossils in Field Photos
Field images often contain rocks and soil. AlexNet-based models can highlight regions most likely to contain fossils, guiding paleontologists during excavations.

### Example 3: Museum Cataloging
Thousands of fossil photographs in museum archives can be automatically sorted into categories. AlexNet reduces manual labor and speeds up classification.

### Example 4: CT Scan Analysis
AlexNet extensions can analyze internal bone structures from CT or micro-CT scans, distinguishing between fossilized bone tissue and surrounding rock.

---

## Practical Implementation
A simplified PyTorch implementation of AlexNet for fossil classification:

```python
import torch
import torch.nn as nn
from torchvision import models

# Load pretrained AlexNet
model = models.alexnet(pretrained=True)

# Replace final classification layer with fossil categories
num_classes = 5  # e.g., femur, rib, vertebra, skull, other
model.classifier[6] = nn.Linear(4096, num_classes)

# Example loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
This example demonstrates how AlexNet can be adapted to paleontology datasets through transfer learning.

Discussion
AlexNet was a milestone in deep learning and remains a valuable teaching tool. While newer models (e.g., ResNet, EfficientNet) achieve better results, AlexNet provides a straightforward entry point to understanding convolutional networks.

In paleontology, AlexNet can reduce manual classification tasks, speed up fossil identification, and expand research capacity. However, challenges remain, including limited labeled datasets, differences between controlled museum images and variable field photos, and the need for interpretability to ensure scientific reliability.

Conclusion
AlexNet marked the beginning of deep learning’s dominance in computer vision. By using convolutional layers, ReLU activations, dropout, and GPU acceleration, it achieved unprecedented accuracy on image classification tasks. In paleontology, AlexNet provides tools for fossil classification, detection, and analysis, bridging modern artificial intelligence with the study of ancient life.

References
Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. Advances in Neural Information Processing Systems, 25, 1097–1105.

Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.

Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective. MIT Press.
