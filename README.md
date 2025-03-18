# Yolo-Football-Analysis

## Overview
This project leverages the **YOLOv11 Large** model, fine-tuned with a dataset from **Roboflow**, to detect football players in a match. Once detected, players are clustered based on their **jersey colors** using **KMeans clustering**. Additionally, **camera movement** is estimated using **OpenCV's perspective transformer**, which is further utilized along with real-world football field dimensions to estimate player speed.

## Input and Output Videos
To demonstrate the effectiveness of the approach, an **input video** containing a football match is processed, and the results are saved in an **output video**.

**Input Video:** https://github.com/Harshv140702/Yolo-Football-Analysis/blob/main/input_videos/08fd33_4.mp4

**Output Video:** https://github.com/Harshv140702/Yolo-Football-Analysis/blob/main/output/output_video.avi

## Methodology
The detection phase employs the **YOLOv11 Large** model, which was fine-tuned using a dataset from **Roboflow** to improve accuracy in identifying players on the field. After detection, **KMeans clustering** is applied to differentiate players based on their jersey colors, allowing for team classification.

To track camera motion, **OpenCV’s Perspective Transformer** is utilized. This transformation helps in compensating for changes in camera angles and movements. Furthermore, by integrating real-world dimensions of a football field, the perspective transform is employed to estimate the speed of players. However, speed calculations are currently restricted to the **middle of the field**, leading to certain limitations.

## Observed Limitations
One of the key challenges faced during implementation was **player misclassification** by KMeans. Specifically, **Player 19** often gets misclassified due to the similarity in jersey colors with other players. Another major limitation is the **restriction of speed calculation to the middle of the field**, which prevents obtaining a comprehensive speed analysis across the entire playing area. Additionally, several calculations are **hardcoded** for this specific video, reducing the flexibility and adaptability of the approach.

## Suggested Improvements
To enhance the accuracy and robustness of the project, several improvements can be considered. A more advanced clustering technique, such as **DBSCAN or a supervised classification model**, could improve player differentiation. Moreover, speed estimation could be refined by dynamically adapting the **field transformation** across different sections of the pitch. Finally, automating parameter adjustments would improve generalizability, making the solution applicable to multiple video inputs.

## Running the Project
To test and run the project, execute the following command:
```bash
python main.py
```
Ensure that all required dependencies, including OpenCV, YOLOv11, and Scikit-learn, are properly installed.
