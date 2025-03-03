from ultralytics import YOLO
import supervision as sv
import os
import pickle
import cv2
import numpy as np
import pandas as pd

class Tracker:
    def __init__(self, model):
        self.model = YOLO(model)
        self.tracker = sv.ByteTrack()

    def add_positions(self, tracks):
        for obj, obj_tracks in tracks.items():
            for frame, track in enumerate(obj_tracks):
                for track_id, track in track.items():
                    bbox = track['bbox']
                    if obj == 'ball':
                        pos = int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2) #center position
                    else:
                        pos = int((bbox[0]+bbox[2])/2), int(bbox[3]) #foot position
                    tracks[obj][frame][track_id]["position"] = pos



    def interpolate_ball(self, ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions] #Incase there's no bbox, it will be [] and pandas will interpolate it
        df_ball = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])
        df_ball = df_ball.interpolate().bfill()
        ball_positions = [{1:{"bbox":x}} for x in df_ball.to_numpy().tolist()] #1 is the tracking id and it is a list of dicts
        return ball_positions

    def detect_frames(self, frames):
        batch_size = 28
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_detections = self.model.predict(frames[i: i+batch_size], conf = 0.1) #can increase confidence in order to remove forced detections
            detections += batch_detections
        return detections

    def get_object_tracking(self, frames, tracks_available=False, track_path = None):
        
        if tracks_available and track_path is not None and os.path.exists(track_path):
            with open(track_path, 'rb') as f:
                obj_tracks = pickle.load(f)
                return obj_tracks

        detections = self.detect_frames(frames)

        obj_tracks ={
                "players":[],
                "referees":[],
                "ball":[]
            }
     
        for frame , detection in enumerate(detections):
            cls = detection.names
            cls_dict = {v:k for k,v in cls.items()} #Saving class info as {referee: 3} type pairs
            sv_detections = sv.Detections.from_ultralytics(detection) #For each frame, coordinates of pixels of bounding boxes, along with confidence, and, class_id
            #The model has issues distinguishing goalkeepers, keeps assigning them player at times. So we're gonna make processing easier
            for object_ind, class_id in enumerate(sv_detections.class_id):
                if cls[class_id] == "goalkeeper":
                    sv_detections.class_id[object_ind] = cls_dict["player"] 

            #Tracking
            tracks_detection = self.tracker.update_with_detections(sv_detections) #adds a tracking id(tracker_id) to the supervision detections

            obj_tracks["players"].append({})
            obj_tracks["referees"].append({})
            obj_tracks["ball"].append({})

            for frame_tracked in tracks_detection:
                boundary = frame_tracked[0].tolist()
                classid = frame_tracked[3] #1 contains mask, 2 contains confidence
                track_id = frame_tracked[4]

                if classid == cls_dict["player"]:  obj_tracks["players"][frame][track_id] = {"bbox":boundary} #A dict of {tracking_id: bbox} for each frame
                if classid == cls_dict["referee"]:  obj_tracks["referees"][frame][track_id] = {"bbox":boundary}

            #Handling ball separately because tracking id is not required (only one ball)
            for frame_tracked in sv_detections:
                boundary = frame_tracked[0].tolist()
                classid = frame_tracked[3] 

                if classid == cls_dict["ball"]:  obj_tracks["ball"][frame][1] = {"bbox":boundary}

        if track_path is not None:
            with open(track_path, 'wb') as f:
                pickle.dump(obj_tracks,f)

        return obj_tracks

    def draw_boundary(self, frame, bbox, color, track_id = None):
        y2 = int(bbox[3]) #We want it to be at the bottom
        x_centre = int((bbox[0]+bbox[2])/2)
        # y_centre = int((bbox[1]+bbox[3])/2)
        box_width = int(bbox[2] - bbox[0])

        cv2.ellipse(frame, center=(x_centre, y2), axes= (int(box_width*0.5), int(box_width*0.25)), angle = 0.0, startAngle=-40, endAngle=240, color= color, thickness=2, lineType=cv2.LINE_4)
        
        rect_w = 40
        rect_h = 20
        x1_rect = x_centre - rect_w//2
        x2_rect = x_centre + rect_w//2
        y1_rect = (y2 - rect_h//2) +15
        y2_rect = (y2 + rect_h//2) +15

        if track_id is not None:
            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED )
            x1_text = x1_rect + 12
            if track_id > 99: x1_text -= 10
            cv2.putText(frame, f"{track_id}", (int(x1_text), int(y1_rect+15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        return frame

    def draw_indicator(self, frame, bbox, color):
        y = int(bbox[1])
        x = int((bbox[0]+bbox[2])/2)
        edges = np.array([
           [x,y],
           [x-10,y-20],
           [x+10, y-20]
        ])
        cv2.drawContours(frame, [edges],0,color,cv2.FILLED)
        cv2.drawContours(frame, [edges],0,(0,0,0),2)
        return frame

    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # Draw a semi-transparent rectaggle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame
    
    def draw_annotations(self, frames, obj_tracks, team_ball_control):
        output_frames = []
        for frame_id, frame in enumerate(frames):
            frame = frame.copy()

            players = obj_tracks["players"][frame_id]
            balls = obj_tracks["ball"][frame_id]
            referees = obj_tracks["referees"][frame_id]

            for track_id, player in players.items():
                color = player.get("team_color", (0,0,255))
                frame = self.draw_boundary(frame, player["bbox"], color, track_id)
                if player.get("has_ball", False):
                    frame = self.draw_indicator(frame,player['bbox'], (0,0,255))
            
            for _, referee in referees.items():
                frame = self.draw_boundary(frame, referee["bbox"], (0,255,255)) #No point tracking individual referees
            
            for _, ball in balls.items():
                frame = self.draw_indicator(frame, ball["bbox"], (0,255,0))

            frame = self.draw_team_ball_control(frame, frame_id, team_ball_control)    

            output_frames.append(frame)
        
        return output_frames









