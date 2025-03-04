import cv2
import numpy as np
import pickle
import os

class CameraMovement():
    def __init__(self, frame):

        self.min_movement = 5

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        #We only wanna extract features from top and bottom corners
        mask_features[:, 0:20] = 1
        mask_features[:,900:1050] = 1

        self.params = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance = 3,
            blockSize = 7, #search size for the features
            mask = mask_features
        )

        self.params2 = dict(
            winSize = (15,15),
            maxLevel = 2, #Downscaling level
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            #Either we loop till stopping criteria or after 10 loops our quality score isn't above 0.03
        )

    def adjust_positions(self,tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adj = (position[0]-camera_movement[0],position[1]-camera_movement[1])
                    tracks[object][frame_num][track_id]['adjusted_position'] = position_adj    

    def get_camera_movement(self, frames, already_available=False, movement_path=None):

        if already_available and movement_path is not None and os.path.exists(movement_path):
            with open(movement_path, 'rb') as f:
                return pickle.load(f)

        camera_movement = [[0,0]]*len(frames) #initialization of [x_movement, y_movement] list for all frames
        
        prev_grayscale = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY) #Grayscale version of last frame
        prev_features = cv2.goodFeaturesToTrack(prev_grayscale, **self.params)

        for frame in range(1, len(frames)):
            frame_grayscale = cv2.cvtColor(frames[frame], cv2.COLOR_BGR2GRAY)
            curr_features, _, _ = cv2.calcOpticalFlowPyrLK(prev_grayscale, frame_grayscale, prev_features, None, **self.params2)

            max_distance = 0
            camera_mov_x, camera_mov_y = 0, 0

            for i, (curr, prev) in enumerate(zip(curr_features, prev_features)):
                curr_features_point = curr.ravel()
                prev_features_point = prev.ravel()       

                dist = ((curr_features_point[0]-prev_features_point[0])**2 + (curr_features_point[1]-prev_features_point[1])**2)**0.5
                if dist > max_distance:
                    max_distance = dist
                    camera_mov_x = curr_features_point[0] - prev_features_point[0]
                    camera_mov_y = curr_features_point[1] - prev_features_point[1]

            if max_distance > self.min_movement:
                camera_movement[frame] = [camera_mov_x, camera_mov_y]
                prev_features = cv2.goodFeaturesToTrack(frame_grayscale, **self.params)

            prev_grayscale = frame_grayscale.copy()
        
        if movement_path is not None:
            with open(movement_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement

        
    def draw_camera_movement(self,frames, camera_movement):
        output_frames=[]

        for frame_num, frame in enumerate(frames):
            frame= frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay,(0,0),(500,100),(255,255,255),-1)
            alpha =0.6
            cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

            x_movement, y_movement = camera_movement[frame_num]
            frame = cv2.putText(frame, "Estimated Camera movements",(10,25), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
            frame = cv2.putText(frame,f"Camera Movement X: {x_movement:.2f}",(10,55), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
            frame = cv2.putText(frame,f"Camera Movement Y: {y_movement:.2f}",(10,85), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)

            output_frames.append(frame) 

        return output_frames
       

