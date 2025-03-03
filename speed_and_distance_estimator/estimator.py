import cv2
import sys 

class SpeedAndDistance():
    def __init__(self):
        self.frame_window=5 #Calculates the speed per every 5 frames
        self.frame_rate=24
    
    def measure_speed_and_distance(self,tracks):
        total_distance= {}

        for obj, obj_tracks in tracks.items():
            if obj == "ball" or obj == "referees": continue 
            total_frames = len(obj_tracks)
            for frame in range(0,total_frames, self.frame_window):
                final_frame = min(frame+self.frame_window,total_frames-1 ) #Capture the last frame

                for track_id,_ in obj_tracks[frame].items():
                    if track_id not in obj_tracks[final_frame]: continue #If the track id is in start frame but not in final, we continue

                    start_pos = obj_tracks[frame][track_id]['transformed_position']
                    end_pos = obj_tracks[final_frame][track_id]['transformed_position']

                    if start_pos is None or end_pos is None: continue #if the user is outside the middle region we continue
                    
                    distance_covered = ((start_pos[0]-end_pos[0])**2 + (start_pos[1]-end_pos[1])**2)**0.5
                    time_elapsed = (final_frame-frame)/self.frame_rate
                    speed = distance_covered/time_elapsed
                    speed = speed*3.6 #Transforming into km/h

                    if obj not in total_distance: total_distance[obj]= {}
                    
                    if track_id not in total_distance[obj]:
                        total_distance[obj][track_id] = 0
                    
                    total_distance[obj][track_id] += distance_covered

                    for frame_batch in range(frame, final_frame):
                        if track_id not in tracks[obj][frame_batch]: continue
                        tracks[obj][frame_batch][track_id]['speed'] = speed
                        tracks[obj][frame_batch][track_id]['distance'] = total_distance[obj][track_id]
    
    def draw_speed_and_distance(self,frames,tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for obj, obj_tracks in tracks.items():
                if obj == "ball" or obj == "referees":
                    continue 
                for _, track in obj_tracks[frame_num].items():
                   if "speed" in track:
                       speed = track.get('speed',None)
                       distance = track.get('distance',None)
                       if speed is None or distance is None:
                           continue
                       
                       bbox = track['bbox']
                       position = [int((bbox[0]+bbox[2])/2), int(bbox[3])]
                       position[1]+=40

                       position = tuple(map(int,position))
                       cv2.putText(frame, f"{speed:.2f} km/h",position,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
                       cv2.putText(frame, f"{distance:.2f} m",(position[0],position[1]+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
            output_frames.append(frame)
        
        return output_frames