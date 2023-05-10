# CFI-Project-Updates
This repository contains codes developed by me and my daily task updates for the CFI project

## Monday, May 8th

This was my first meeting of the CFI project with Shihao, Vijayi. The project involves data collection & calibration to produce 3D MoCap in the wild. We discussed the project objectives, environmental setup & future prospects. The project involves data capture in the wild, data processing and multimodality. There will be two test runs- on May 30th, June 11th and a final run probably in August.

Data Collection Stage:
There will be 6 stationery Azure Kinect cameras, two non-stationary GoPros and eye trackers for recording the subjects. The cameras capture frames and write them to disk simaltaneously

Data Processing:
This is the most important step as the data frames need to be time-synchronized.

Multimodality:
Apart from the visual features, we can also incorporate Acoustic signals obtained from processing the speech captured by the Azure Kinect cameras and signals obtained from the eye-tracking glasses. There is also a possibility for capturing frames with drones, however calibrating this seems challenging

Assigned Tasks

Read up Camera Calibration from “Multiple View Geometry in Computer Vision, Second Edition”

The current pipeline captures only image frames periodically and performs the camera calibration. Develop a C++ based code for capturing speech inputs with our Azure Kinect camera

My Ideas

Possibility for Improvement in camera calibration? Can we achieve a better FPS? The current pipeline performs frame capture and writing to disk simaltaneously at 10 FPS. We can save the overhead of writing to disk simultaneously by writing to a small buffer memory first and then periodically writing to the disk.

Audio features can be extracted using tools such as openSMILE, which is an open-source toolkit for extracting features from speech signals. We can extract features such as pitch, loudness, spectral features, and prosody features, which capture aspects such as the speaker’s intonation and rhythm. Further, the librosa package in Python can be used to extract audio features like Mel-frequency cepstral coefficients (MFCC) which are commonly used in speech recognition tasks

To capture the facial features, we can use the OpenFace toolkit which provides a deep neural network-based facial recognition system that extracts facial landmarks and expressions.

