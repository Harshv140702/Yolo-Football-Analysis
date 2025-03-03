import numpy as np 
import cv2

class PerspectiveTransformer():
    def __init__(self):
        court_width = 68
        court_length = 23.32 #Calculated for 4 even rectangles in the middle 

        self.distorted_vertices = np.array([[110, 1035], [265, 275], [910, 260], [1640, 915]]).astype(np.float32) #Estimated pixel values for the middle of the court
        
        self.real_vertices = np.array([[0,court_width],[0, 0],[court_length, 0],[court_length, court_width]]).astype(np.float32)

        self.persepctive_trasnformer = cv2.getPerspectiveTransform(self.distorted_vertices, self.real_vertices)

    def transform_point(self,point):
        p = (int(point[0]),int(point[1]))
        is_inside = cv2.pointPolygonTest(self.distorted_vertices,p,False) >= 0  #Check if point is in the middle
        if not is_inside:
            return None

        reshaped_point = point.reshape(-1,1,2).astype(np.float32)
        transform_point = cv2.perspectiveTransform(reshaped_point,self.persepctive_trasnformer)
        return transform_point.reshape(-1,2)

    def transform_positions(self,tracks):
        for obj, obj_tracks in tracks.items():
            for frame, track in enumerate(obj_tracks):
                for track_id, track_info in track.items():
                    position = track_info['adjusted_position']
                    position = np.array(position)
                    new_pos = self.transform_point(position)
                    if new_pos is not None:
                        new_pos = new_pos.squeeze().tolist()
                    tracks[obj][frame][track_id]['transformed_position'] = new_pos