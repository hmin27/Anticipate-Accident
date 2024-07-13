# Anticipating Car Accidents in Dashcam Videos
This project is a re-implementation & improvement of the model described in the paper Anticipating Accidents in Dashcam Videos. The goal is to anticipate car accidents in dashcam videos using a Recurrent Neural Network (RNN) with a dynamic soft attention mechanism, which focuses on objects within the video frames.

# Overview
Objective: To predict car accidents in dashcam videos by focusing on critical objects using an RNN with a soft attention mechanism.
Current Status: The model has been re-implemented with an RNN (LSTM) and dynamic soft attention. For different base model for object detection, YOLO have been experimented with.
Future Work: Plan to enhance the architecture using transformer-based models or by incorporating segmentation and point tracking in 3D space.

# Features
RNN (LSTM) with Soft Attention: Utilizes Long Short-Term Memory (LSTM) networks combined with a dynamic soft attention mechanism to process sequential video frames.
Object Detection: Integrates state-of-the-art object detection models (e.g., YOLO) to identify and focus on relevant objects within the video frames.
Anticipation of Accidents: Predicts the likelihood of an accident occurring in the near future by analyzing the sequence of frames and the detected objects.

# Requirements
python 3.8
pytorch 2.3.1
Opencv 2.4.9
Matplotlib
Numpy

# Dataset
For the datset, [Dashcam videos](https://aliensunmin.github.io/project/dashcam/) are used. The dataset consists of 620 videos captured in six major cities in Taiwan. Our diverse accidents include: 42.6% motorbike hits car, 19.7% car hits car, 15.6% motorbike hits motorbike, and 20% other type.
<p align="center">
  <img src="https://github.com/user-attachments/assets/8a2c7d85-c2cf-49ff-8c23-40ec8b3e056e">
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/337020a7-004b-4ccd-8525-96a3de07e7e0">
</p>

# How to run
```
conda create --name dashcam python=3.8
conda activate dashcam
conda install pytorch torchvision torchaudio -c pytorch -c nvidia
git clone https://github.com/hmin27/Anticipate-Accident.git
```

For training
```
python Run.py
```

For Visualization
```
python Visualization.py
```

</br>
</br>
This project was conducted in the first semester of 2024 with 2 team members at Korea University.

