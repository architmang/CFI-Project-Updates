# CFI-Project-Updates
Hi! This is Archit Mangrulkar. This repository contains codes developed by me and my daily task updates for the CFI project

## Thursday, May 18th 

1. We captured new data today however we still face the same problem of chessboard identification ith infrared images as well. I & Kinjawl tried several approaches to solve this problem such as increasing contrast, scaling, histogram equalization, gaussian smoothing, dilation, and Canny edge detection however none of them are working. 

<p align="center">
    <img width="800" height="400" src="images/low_thresh.png" alt="Load Image">
</p>

It would be better to record the data in a closed room as the infrared images are picking up heat noises which is clearly visible in the below image.

<p align="center">
    <img width="800" height="400" src="images/best_image_heat_noise.png" alt="Load Image">
</p>

2. After Shihao went, we even tried to record some more infrared data with the subject (yes, its me :)) being closer to the camera however all the preprocessing steps fail to detect chessboards. 

<p align="center">
    <img width="800" height="400" src="images/new_data.jpg" alt="Load Image">
</p>

After applying some image transforms, you can see that some squares are visible however not all.

<p align="center">
    <img width="800" height="400" src="images/some_squares_visible.png" alt="Load Image">
</p>

### Future Tasks

1. Transform the three 3D poses to the same coordinates using extrinsic calibration

2. Perform person tracking over time in this unified 3D view by verifying the extent of overlap between bounding boxes after tranformation of these 3D views

3. Once you are done with these, try markerless motion capture using [EasyMoCap](https://github.com/zju3dv/EasyMocap)

## Wednesday, May 17th 

1. I worked on the code for finding the intrinsic matrices for all kinects using the demo data images. The code for rgb camera matrix calibration is working correctly however OpenCV is not able to detect check board corners in the depth images. I requested Shihao to come to the lab tomorrow so we could collect infrared image data that will help us perform intrinsic camera matrix calibration for the depth camera as well. Sample results for intrinsic matrix for rgb cammera for one of the kinects:

            azure_kinect1_2
            
            ret: 1.1692189510474589
            Camera matrix:
             [[919.35146135   0.         987.83993318]
             [  0.         913.10601197 567.97330538]
             [  0.           0.           1.        ]]
            Distortion coefficients:
             [[ 0.05701473  0.04425453  0.00379301  0.01158766 -0.14838415]]

2.  I modified the python otebook provided by you into two seperate modules for depth 2 color registration with and without distortion. Here are the results:

<p align="center">
    <img width="800" height="400" src="images/depth2color_distorted.png" alt="Load Image">
</p>

<p align="center">
    <img width="800" height="400" src="images/depth2color_undistorted.png" alt="Load Image">
</p>

### Future Tasks

1. Record new data with infrared images as well.

2. Transform the three 3D poses to the same coordinates using extrinsic calibration

3. Perform person tracking over time in this unified 3D view by verifying the extent of overlap between bounding boxes after tranformation of these 3D views

## Tuesday, May 16th 

1. After facing a lot of issues with markerless human pose estimation using [HMR](https://github.com/akanazawa/hmr) and [SPIN](https://github.com/nkolot/SPIN), Shihao suggested to leave it for now. However, I came across a new library [EasyMocap](https://github.com/zju3dv/EasyMocap). I am willing to give it a last try.

2. Kinjawl tried performing depth-to-color registration using the camera matrices provided by Shihao but the results were not good as we realised that intrinsic camera matrices depedon the camera and the camera mode which was used to record our data was in 1536P while the one used by Shihao was of 1080P.

<p align="center">
    <img width="800" height="400" src="images/depth2color_newdata_old_calibration.png" alt="Load Image">
</p>

3. Currently we are working on finding the intrinsic matrices using the demo data images.

### Future Tasks

1. Transform the three 3D poses to the same coordinates using extrinsic calibration

2. Perform person tracking over time in this unified 3D view by verifying the extent of overlap between bounding boxes after tranformation of these 3D views

3. Use distortion parameters to get error free results

## Friday, May 12th

1. I removed some of the blurred images in our dataset. However the YOLLOv8 models run smoothly on our dataset, therefore removal of all blurred images was not needed. 

2. I worked on the pose estimation module. I used MediaPipe for this purpose however it does perform multi-object pose estimation rather it focuses on one object. Below is one of the results. I'll try to use other MediaPipe models in the meantime.

<p align="center">
    <img width="800" height="400" src="images/MediaPipe.jpg" alt="Load Image">
</p>

3. Kinjawl worked on the human detection model using two YOLOv8 models- the largest one with 700ms CPU latency and the smallest and least accurate model with 65.4 ms CPU latency

<p align="center">
    <img width="800" height="400" src="images/yolov8.jpg" alt="Load Image">
</p>

4. We had a online meeting with Shihao where he explained us the intrinsic & extrinsic camera calibration codes.  
### Assigned Tasks

1. Use OpenPose to achieve better pose estomation results

2. Perform 3D pose reconstruction from these 2D poses using depth2color calibration codes

3. Transform the three 3D poses to the same coordinates using extrinsic calibration

4. Perform person tracking over time in this unified 3D view by verifying the extent of overlap between bounding boxes after tranformation of these 3D views

5. Use distortion parameters to get error free results  

## Thursday, May 11th

1. Shihao came over and explained the python client and the azure kinect c++ server codes. 

2. With Shihao's help we established a three camera setup and captured a small dataset of humans in the wild. 

<p align="center">
    <img width="800" height="400" src="images/captured_data.jpg" alt="Load Image">
</p>


### Assigned Tasks

1. Remove the blurred images in our dataset

2. Integrate the speech signal capture pipeline with our image capture pipeline by following the [Azure Speech SDK tutorial](https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/speech-sdk) 

3. Perform object detection using [OpenCV](https://opencv.org/) and pose estimation using [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), [MMpose](https://github.com/open-mmlab/mmpose)

## Wednesday, May 10th

1. Going through the [Azure Speech SDK tutorial](https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/speech-sdk). The tutorial is really helpful. The speech signal capture pipeline would be integrated with our main pipeline in a few days.

<p align="center">
    <img width="800" height="400" src="images/azure.jpg" alt="Load Image">
</p>

## Tuesday, May 9th

1. I have completed the reading up on the portion of Camera Calibration from [Multiple View Geometry in Computer Vision, Second Edition](http://www.r-5.org/files/books/computers/algo-list/image-processing/vision/Richard_Hartley_Andrew_Zisserman-Multiple_View_Geometry_in_Computer_Vision-EN.pdf)

2. Currently going through the [camera callibration codes](https://drive.google.com/drive/folders/1jtcK4WQyD9mzVvs61w-aSvgjQe_X-reO?usp=share_link) shared by Shihao. The next goal is to integrate the speech signal capture with our current work.

3. I helped Kinjawl with the project onboarding, explained him the data collection pipeline and shared some of the resources with him

## Monday, May 8th

This was my first meeting of the CFI project with Shihao, Vijay. The project involves data collection & calibration to produce 3D MoCap in the wild. We discussed the project objectives, environmental setup & future prospects. The project involves data capture in the wild, data processing and multimodality. There will be two test runs- on May 30th, June 11th and a final run probably in August.

### Data Collection Stage:
There will be 6 stationery Azure Kinect cameras, two non-stationary GoPros and eye trackers for recording the subjects. The cameras capture frames and write them to disk simaltaneously

<p align="center">
    <img width="800" height="400" src="images/setup.jpg" alt="Load Image">
</p>

### Data Processing:
This is the most important step as the data frames need to be time-synchronized.

### Multimodality:
Apart from the visual features, we can also incorporate Acoustic signals obtained from processing the speech captured by the Azure Kinect cameras and signals obtained from the eye-tracking glasses. There is also a possibility for capturing frames with drones, however calibrating this seems challenging

### Assigned Tasks

1. Read up Camera Calibration from [Multiple View Geometry in Computer Vision, Second Edition](http://www.r-5.org/files/books/computers/algo-list/image-processing/vision/Richard_Hartley_Andrew_Zisserman-Multiple_View_Geometry_in_Computer_Vision-EN.pdf)

2. The current pipeline captures only image frames periodically and performs the camera calibration. Develop a C++ based code for capturing speech inputs with our [Azure Kinect camera](https://azure.microsoft.com/en-us/products/kinect-dk#layout-container-uid9b7e)

### My Ideas

1. Possibility for Improvement in camera calibration? Can we achieve a better FPS? The current pipeline performs frame capture and writing to disk simaltaneously at 10 FPS. We can save the overhead of writing to disk simultaneously by writing to a small buffer memory first and then periodically writing to the disk.

2. Audio features can be extracted using tools such as [openSMILE](https://audeering.github.io/opensmile-python/), which is an open-source toolkit for extracting features from speech signals. We can extract features such as pitch, loudness, spectral features, and prosody features, which capture aspects such as the speaker’s intonation and rhythm. Further, the [librosa package](https://librosa.org/doc/latest/index.html) in Python can be used to extract audio features like Mel-frequency cepstral coefficients (MFCC) which are commonly used in speech recognition tasks

3. To capture the facial features, we can use the [OpenFace toolkit](https://cmusatyalab.github.io/openface/) which provides a deep neural network-based facial recognition system that extracts facial landmarks and expressions.

