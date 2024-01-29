<h1 align="center">
Using YOLO for real-time object recognition through a camera. 
</h1>

<div align="center">
<h3>
<a href="https://www.linkedin.com/in/beatriz-emiliano/">Beatriz Emiliano Maciel de Sousa</a>
</h3>
</div>

# Guide
- [Description](#description)
- [Features](#features)
- [Technologies used](#technologies-used)
- [Installations](#installations)
- [Application](#application)

# Description
This repository provides an implementation in CoppeliaSim for the calculation of 3D to 2D projection using the focal length method. The goal is to transform three-dimensional points into two-dimensional pixel coordinates, simulating the projection process in a virtual camera. The method takes into account the camera's focal length to compute pixel coordinates corresponding to 3D points, crucial in various applications such as computer vision and virtual environment simulations.
<div align="center">

![overview](img/img1.png)

</div>

# Features
- YOLOv3: The code employs a specific version of the YOLO architecture, providing accurate and efficient object detection.
- Camera Integration: The application captures and processes real-time images directly from a camera, enabling continuous object recognition.
- Pre-trained AI: The YOLO model is pre-trained on a comprehensive dataset, resulting in robust recognition of various objects.
- CSV Output: All recognized objects in the scene are saved to a .csv file, providing a convenient way to analyze and track detected objects.

# Technologies used:
- Yolov3
- Python3
- Numpy
- OpenCV
- CSV

# Installations:

-  Numpy && OpenCV
```
$ pip install opencv-python
$ pip install numpy
```

# Application

- Clone the repository:
```
$ git clone 
```
- Run the script: 
```
$ python3 obj_recognition_real_time.py
```

<div align="center">

![overview](img/im2.png)

</div>
