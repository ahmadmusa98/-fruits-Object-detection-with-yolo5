# YOLOv5 Custom Object Detection - Fruit Detection(APPLE detection )

## Overview

This project demonstrates the training of a **YOLOv5** model for detecting **fruits** in images. YOLOv5, developed by Ultralytics, is one of the most popular deep learning models for real-time object detection. It can detect objects with high accuracy and speed in images and videos.

In this project, we focus on training a **custom YOLOv5 model** using a dataset of various fruits. The dataset is sourced from **Roboflow**, a platform that provides easy access to datasets and facilitates the training process. The goal is to configure and train a model that can identify fruits such as apples, bananas, oranges, and more.

### Key Components:
- **YOLOv5 Architecture**: The model configuration, including the depth and width of the network, the number of classes, and anchor box sizes.
- **Training**: The process of training the YOLOv5 model on a custom fruit dataset.
- **TensorBoard**: Used for visualizing the training process, including metrics like loss, accuracy, precision, and recall.

## Deep Learning-Based Model Description

### 1. **YOLOv5 Overview**
YOLOv5 is an implementation of the "You Only Look Once" (YOLO) algorithm, which is designed for real-time object detection. It predicts the class and bounding box coordinates for each object in an image. The model consists of:
- **Backbone**: A feature extractor (like convolutional layers).
- **Head**: The detection mechanism that makes final predictions about the objects in the image.

The model is highly efficient and optimized for performance, providing a balance between speed and accuracy.

### 2. **Dataset Overview**
The dataset used in this project contains labeled images of fruits. The dataset is split into two main parts:
- **Training Set**: Used to train the model, which includes annotated images of fruits with labels.
- **Validation Set**: Used to evaluate the model's performance during training.

The dataset was sourced from **Roboflow**, which allows users to easily upload and annotate images, and download the dataset in formats compatible with YOLOv5.

### 3. **Training Process**
The YOLOv5 model was trained using the following hyperparameters:

- **Image Size (`--img`)**: 
   - The model uses an input image size of `416x416` pixels. This size is a trade-off between computational efficiency and the ability to detect small objects.
   
- **Batch Size (`--batch`)**:
   - The batch size of `16` was used. This refers to the number of images processed per training step. A larger batch size can speed up training but may require more GPU memory.

- **Number of Epochs (`--epochs`)**:
   - The model was trained for `50` epochs. An epoch refers to one complete pass through the entire dataset during training. The number of epochs determines how many times the model will see the entire dataset.

- **Learning Rate**:
   - A default learning rate was used, which adjusts how much the model's weights are updated after each training step. The learning rate can affect how quickly the model converges and its final performance.

- **Model Configuration (`--cfg`)**:
   - A custom YAML configuration file was created for the YOLOv5 model. This file defines the structure of the model, such as the number of classes (for fruit detection) and the architecture of the network (backbone and head).

   - **Number of Classes (`nc`)**: The number of fruit classes (e.g., 3 classes for apple, banana, and orange).
   - **Depth Multiple (`depth_multiple`)**: A parameter that controls how deep the model is (i.e., how many layers are in the network).
   - **Width Multiple (`width_multiple`)**: A parameter that adjusts the number of filters or channels in each layer, affecting the complexity of the model.
   
   - **Anchors**: These are predefined bounding box shapes that help the model learn how to detect objects at various scales. The anchor sizes are manually set to detect small, medium, and large objects:
     - Small (P3/8)
     - Medium (P4/16)
     - Large (P5/32)

### 4. **Model Architecture**
YOLOv5's architecture consists of two main parts:
1. **Backbone**:
   - The backbone extracts important features from the input image through a series of convolutional layers, followed by more complex layers like **BottleneckCSP** and **SPP** (Spatial Pyramid Pooling).
   - The backbone enables the model to learn hierarchical features, allowing it to detect objects at multiple scales.

2. **Head**:
   - The head combines the features extracted by the backbone and makes the final predictions. The head consists of several layers that concatenate features from different levels of the backbone, followed by convolution layers and upsampling operations.
   - The final layer outputs predictions for the number of classes (`nc`), bounding box coordinates, and object confidence scores.

### 5. **Loss Function**
YOLOv5 uses a combination of three loss components during training:
- **Objectness Loss**: Measures how confident the model is that an object exists in a given region.
- **Classification Loss**: Measures the accuracy of class predictions (i.e., what fruit the model thinks is in the image).
- **Bounding Box Loss**: Measures how accurate the predicted bounding box is compared to the ground truth.

The total loss is the sum of these components, and the model is trained to minimize this loss.

---

## Parameters Used in the Training

- **Number of Classes (`nc`)**: 
   - This value is derived from the number of different fruit categories in the dataset. For example, if the dataset contains apples, bananas, and oranges, `nc = 3`.

- **Depth Multiple (`depth_multiple`)**: 
   - A value of `0.33` is used. This parameter controls the depth (number of layers) of the network. Smaller values reduce the depth and make the model more efficient, while larger values increase the depth for more complex models.

- **Width Multiple (`width_multiple`)**: 
   - A value of `0.50` is used. This parameter controls the number of channels in each layer of the network. Smaller values reduce the number of channels, making the model more lightweight, while larger values increase the number of channels for higher capacity.

- **Anchor Boxes**: 
   - The model uses anchor boxes to help predict bounding boxes for objects. The anchor sizes for this model are:
     - **Small**: [10, 13, 16, 30, 33, 23]
     - **Medium**: [30, 61, 62, 45, 59, 119]
     - **Large**: [116, 90, 156, 198, 373, 326]

- **Training Settings**: 
   - **Batch Size**: `16` images per batch.
   - **Epochs**: `50` epochs for model training.
   - **Image Size**: `416x416` pixels for input images.
   - **Learning Rate**: Default settings used.

- **Optimizer**:
   - The model uses the **Adam** optimizer with default parameters. This optimizer adapts the learning rate during training to improve convergence.

---

## Conclusion

The **YOLOv5 Custom Object Detection** project successfully trains a model for fruit detection, achieving high performance with a custom configuration that adapts to specific needs. The use of Roboflow's dataset and the flexible architecture of YOLOv5 allows for easy customization to detect other objects as well.

The model can be fine-tuned further, depending on the application. You can adjust the number of epochs, batch size, image size, and network depth to optimize the model for your specific dataset and hardware.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more information.
