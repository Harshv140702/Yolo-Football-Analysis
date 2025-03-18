# Yolo-Football-Analysis

## Overview
This project leverages the YOLOv11 Large model, fine-tuned with a dataset from Roboflow (https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1), to detect football players in a match. Once detected, players are clustered based on their jersey colors using KMeans clustering. Additionally, camera movement is estimated using OpenCV's perspective transformer, which is further utilized along with real-world football field dimensions to estimate player speed.

## Input and Output Videos
**Input Video:** https://github.com/Harshv140702/Yolo-Football-Analysis/blob/main/input_videos/08fd33_4.mp4

**Output Video:** https://github.com/Harshv140702/Yolo-Football-Analysis/blob/main/output/output_video.avi

## Observed Limitations
One of the key challenges faced during implementation was **player misclassification** by KMeans. Specifically, **Player 19** often gets misclassified. Another major limitation is the **restriction of speed calculation to the middle of the field**, which prevents obtaining a comprehensive speed analysis across the entire playing area. Calculations such as perspective transformation have been hardcoded to this video, using empirical data for dimensions of football fields.

## Suggested Improvements
To enhance the accuracy and robustness of the project, several improvements can be considered. Kmeans can be used with more random initializations in order to enhance accuracy. A more advanced clustering technique, such as **DBSCAN or a supervised classification model**, could improve player differentiation. Moreover, speed estimation could be refined by dynamically adapting the **field transformation** across different sections of the pitch. Finally, automating parameter adjustments would improve generalizability, making the solution applicable to multiple video inputs.

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
