# YOLOv5 Custom Object Detection - Fruit Detection

## Overview

This project demonstrates the training of a custom **YOLOv5** model for **object detection**, specifically focused on detecting various **fruits** in images. YOLO (You Only Look Once) is a state-of-the-art real-time object detection algorithm that efficiently detects objects in images and videos. In this project, we leverage YOLOv5 to detect different types of fruits, using a dataset obtained from **Roboflow**. The model is trained from scratch using a custom configuration.

## Requirements

Before you start, make sure you have the following installed:

### Prerequisites
- Python 3.8+
- PyTorch 1.7+ (with CUDA support for GPU acceleration)
- Git
- TensorBoard (for monitoring training)

### Installing Dependencies

Clone the YOLOv5 repository and install the required dependencies:

```bash
# Clone the YOLOv5 repository
git clone https://github.com/ultralytics/yolov5  # Clone the repo
cd yolov5  # Navigate into the YOLOv5 directory

# Reset to a specific commit for consistency
git reset --hard 886f1c03d839575afecb059accf74296fad395b6  

# Install dependencies (ignore any errors)
pip install -qr requirements.txt  
pip install -q roboflow  # Install Roboflow library for dataset management
Step 1: Set Up Roboflow Dataset
Create a Roboflow account and navigate to your project (e.g., Fruit Detection).

Obtain your API Key from Roboflow.

Download the dataset and configure it for YOLOv5:

python
Copy
Edit
from roboflow import Roboflow

# Initialize Roboflow and load the dataset using your API key
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace().project("fruit-detection-iqgy7")  # Your project ID
dataset = project.version(1).download("yolov5")  # Download dataset formatted for YOLOv5
This will download the dataset and save it in the dataset.location folder.

Step 2: Check YAML Configuration
To check the dataset’s configuration, view the data.yaml file, which contains important information like the number of classes and paths to training/validation images.

bash
Copy
Edit
cat {dataset.location}/data.yaml  # Display the YAML configuration for the dataset
The data.yaml file contains:

nc: The number of classes in the dataset (e.g., the number of different fruits you want to detect).

train: Path to the training images.

val: Path to the validation images.

names: List of class names (e.g., 'apple', 'banana', etc.).

Step 3: Customize the YOLOv5 Model
Now we will modify the YOLOv5 model to fit our custom dataset. This involves adjusting the configuration of the model based on the number of classes in your dataset and setting up custom anchors.

First, load the number of classes from the YAML configuration:

python
Copy
Edit
import yaml

with open(dataset.location + "/data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])  # Get the number of classes from the YAML file
Model Configuration (custom_yolov5s.yaml)
Now, let’s define the custom YOLOv5 model by configuring the following hyperparameters:

Number of Classes (nc): Based on the number of fruit classes.

Depth and Width Multiples: Control the complexity of the model (how deep and wide the network is).

Anchors: Predefined bounding box shapes that help the model learn to detect objects.

Here is a sample model configuration:

yaml
Copy
Edit
# Model parameters
nc: {num_classes}  # Number of classes (dynamically replaced with the number of classes in the dataset)
depth_multiple: 0.33  # Depth multiplier (controls the number of layers in the model)
width_multiple: 0.50  # Width multiplier (controls the number of channels in each layer)

# Anchors
anchors:
  - [10,13, 16,30, 33,23]  # P3/8 (small objects)
  - [30,61, 62,45, 59,119]  # P4/16 (medium objects)
  - [116,90, 156,198, 373,326]  # P5/32 (large objects)

# Backbone: Extracts features from the image
backbone:
  [[-1, 1, Focus, [64, 3]],  # Focus Layer (64 channels, kernel size of 3)
   [-1, 1, Conv, [128, 3, 2]],  # Conv Layer (128 channels, kernel size 3, stride 2)
   [-1, 3, BottleneckCSP, [128]],  # BottleneckCSP (128 channels)
   [-1, 1, Conv, [256, 3, 2]],  # Conv Layer (256 channels, kernel size 3, stride 2)
   [-1, 9, BottleneckCSP, [256]],  # BottleneckCSP (256 channels)
   [-1, 1, Conv, [512, 3, 2]],  # Conv Layer (512 channels, kernel size 3, stride 2)
   [-1, 9, BottleneckCSP, [512]],  # BottleneckCSP (512 channels)
   [-1, 1, Conv, [1024, 3, 2]],  # Conv Layer (1024 channels, kernel size 3, stride 2)
   [-1, 1, SPP, [1024, [5, 9, 13]]],  # Spatial Pyramid Pooling (SPP) Layer
   [-1, 3, BottleneckCSP, [1024, False]],  # BottleneckCSP (1024 channels, without residual connections)
  ]

# Head: Detection head of the model
head:
  [[-1, 1, Conv, [512, 1, 1]],  # Conv Layer (512 channels, kernel size 1x1)
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],  # Upsample the feature map
   [[-1, 6], 1, Concat, [1]],  # Concatenate feature map from backbone
   [-1, 3, BottleneckCSP, [512, False]],  # BottleneckCSP (512 channels)
   
   [-1, 1, Conv, [256, 1, 1]],  # Conv Layer (256 channels, kernel size 1x1)
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],  # Upsample the feature map
   [[-1, 4], 1, Concat, [1]],  # Concatenate feature map from backbone
   [-1, 3, BottleneckCSP, [256, False]],  # BottleneckCSP (256 channels)
   
   [-1, 1, Conv, [256, 3, 2]],  # Conv Layer (256 channels, kernel size 3, stride 2)
   [[-1, 14], 1, Concat, [1]],  # Concatenate feature map from head
   [-1, 3, BottleneckCSP, [512, False]],  # BottleneckCSP (512 channels)
   
   [-1, 1, Conv, [512, 3, 2]],  # Conv Layer (512 channels, kernel size 3, stride 2)
   [[-1, 10], 1, Concat, [1]],  # Concatenate feature map from head
   [-1, 3, BottleneckCSP, [1024, False]],  # BottleneckCSP (1024 channels)
   
   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect Layer (final detection output for classes)
  ]
Focus Layer: A specialized layer that extracts spatially focused features from the input image.

Conv Layers: Convolution layers that reduce image resolution while extracting features.

BottleneckCSP: A residual block designed to improve training efficiency by maintaining gradient flow through deeper layers.

SPP: Spatial Pyramid Pooling increases the receptive field and allows the network to detect objects at multiple scales.

Step 4: Training the YOLOv5 Model
Once the dataset and model configuration are set, you can start training the YOLOv5 model:

bash
Copy
Edit
cd /content/yolov5/
python train.py --img 416 --batch 16 --epochs 50 --data {dataset.location}/data.yaml --cfg ./models/custom_yolov5s.yaml --weights '' --name yolov5s_results --cache
Parameters:
--img 416: Specifies the input image size (416x416 pixels).

--batch 16: Batch size (number of images processed per step).

--epochs 50: Number of training epochs (iterations through the entire dataset).

--data {dataset.location}/data.yaml: Path to the dataset configuration file.

--cfg ./models/custom_yolov5s.yaml: Path to the custom model configuration file.

--weights '': Start training from scratch (without pre-trained weights).

--name yolov5s_results: Directory name to save the results.

--cache: Cache images to speed up training.

Step 5: Monitor Training with TensorBoard
You can monitor the training progress using TensorBoard, which will allow you to visualize various metrics like loss, precision, recall, and more:

bash
Copy
Edit
# Start TensorBoard
%load_ext tensorboard
%tensorboard --logdir runs
This will open a TensorBoard interface to track training performance and optimize model parameters.

Conclusion
This repository provides a detailed, step-by-step guide for training a custom YOLOv5 model for fruit detection. The flexibility of YOLOv5 allows you to fine-tune the model with custom datasets and configurations to suit your specific needs.

By adjusting parameters like depth_multiple, width_multiple, and nc, you can customize the model's performance for a wide range of applications.

License
This project is licensed under the MIT License - see the LICENSE file for details.

markdown
Copy
Edit

---

### Instructions to Add to GitHub:
1. **Copy the entire content** above.
2. **Go to your repository** on GitHub.
3. **Create or Edit the `README.md`** file.
4. **Paste the content** into the editor.
5. **Commit the changes** to save the updated README.

Once done, your **README.md** will be fully populated and displayed on your GitHub repository.
