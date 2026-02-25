# Update
## February 25, 2026
- Move hard-coded parameters into config.yaml and load them via config.py.
- Encapsulate repetitive logic (model loading, rendering, etc.) into utils.py.
- No more coffee but milk.

## February 23, 2026
- Remove unnecessary code.
- All enabled functions cam be displayed on one screen.
- 为什么我改模块里的代码，main也跟着一块崩啊？

## February 21, 2026
- We have implemented support for local models.
- Optimized file logic.
- Mac is no longer supported!

# Visual Detection Python Project
A real-time visual detection project implemented with Python + OpenCV + MediaPipe that captures and marks detection targets in real-time via camera.
1. Face Detection
2. Hand Gesture Detection
3. Human Stickman Detection (new!)

## Environment Requirements
Python 3.12 recommanded but 3.11 at least.
Computer with camera.

You need to install dependency libraries manually, or try using pipenv to call flie in ./env/Pipfile:
    - opencv-python (core of image processing)
    - mediapipe (core of hand gesture recognition)
    - pyyaml (for config file parsing)

There is the list of packages for this project.
| Packages | Version | Python Version Requested |
| --- | --- | --- |
| absl-py | 2.4.0 | >= 3.10 |
| cffi | 2.0.0 | >= 3.9 |
| contourpy | 1.3.3 | >= 3.11 |
| cycler | 0.12.1 | >= 3.8 |
| flatbuffers | 25.12.19 | --  |
| fonttools | 4.61.1 | >= 3.10 |
| kiwisolver | 1.4.9 | >= 3.10 |
| matplotlib | 3.10.8 | >= 3.10 |
| mediapipe | 0.10.32 | --  |
| numpy | 2.4.2 | >= 3.11 |
| opencv-contrib-python | 4.13.0.92 | >= 3.6 |
| opencv-python | 4.13.0.92 | >= 3.6 |
| packaging | 26.0 | >= 3.8 |
| pillow | 12.1.1 | >= 3.10 |
| pycparser | 3.0 | >= 3.10 |
| pyparsing | 3.3.2 | >= 3.9 |
| python-dateutil | 2.9.0.post0 | >= 2.7 nor 3.0/3.1/3.2 |
| pyyaml | 6.0.3 | >= 3.8 |
| six | 1.17.0 | >= 2.7 nor 3.0/3.1/3.2 |
| sounddevice | 0.5.5 | >= 3.7 |

# Installation Steps
1. Clone / Download this project to your local machine
```
git clone https://github.com/MakoMori2023/Visual-Detection-Python
cd Visual-Detection-Python
```

2. Install the required dependency libraries
```
pip install opencv-python mediapipe pyyaml
```

3. Create a Model directory in the project root directory (**if it does not exist**)

    Place the following model files into the Model directory:
    - blaze_face_short_range.tflite (for face detection)
    - hand_landmarker.task (for hand gesture detection)
    - pose_landmarker_full.task (for human stickman detection)

5. Ensure the model file names and paths are consistent with the model_path configuration in config.yaml

6. Start with type `python ./main.py` in the project directory. 


Visual Detection Python Project. Copyright (C) Akira Amatsume

This program comes with ABSOLUTELY NO WARRANTY; for details see LICENSE. This is free software, and you are welcome to redistribute it under certain conditions.



