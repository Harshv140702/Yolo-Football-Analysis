# Yolo-Football-Analysis

## Overview
This project leverages the YOLOv11 Large model, fine-tuned with a dataset from Roboflow (https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1), to detect football players in a match. Once detected, players are clustered based on their jersey colors using KMeans clustering. Additionally, camera movement is estimated using OpenCV's perspective transformer, which is further utilized along with real-world football field dimensions to estimate player speed.

## Input and Output Videos
**Input Video:** https://github.com/Harshv140702/Yolo-Football-Analysis/blob/main/input_videos/08fd33_4.mp4

**Output Video:** https://github.com/Harshv140702/Yolo-Football-Analysis/blob/main/output/output_video.avi


## Running the Project
To test and run the project, execute the following command:
```bash
python main.py
```
Ensure that all following required dependencies are installed:
- Python 3.x
- ultralytics
- supervision
- OpenCV
- NumPy
- Matplotlib
- Pandas
