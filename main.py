from utils import load_frames, save_video
from tracking import Tracker, TeamAssignment, PlayerBallAssigner
from detect_camera_movement import CameraMovement
from perspective_transformer import PerspectiveTransformer
from speed_and_distance_estimator import SpeedAndDistance
import cv2
import numpy as np

def main():
    #Read frames from video
    frames = load_frames('input_videos/08fd33_4.mp4')
    
    #Initialize tracking
    tracker = Tracker('models/best.pt') 

    tracks = tracker.get_object_tracking(frames, tracks_available=True, track_path= 'prerun/track.pkl')

    #Add object positions
    tracker.add_positions(tracks)
    
    #Estimating camera movement per frame
    camera_movement = CameraMovement(frames[0])
    camera_movements = camera_movement.get_camera_movement(frames, already_available=True, movement_path='prerun/camera_movement.pkl')
    
    #adjusting positions according to camera movement
    camera_movement.adjust_positions(tracks, camera_movements)

    #Adding transformed positions for player speed tracking
    perspective_transformer = PerspectiveTransformer()
    perspective_transformer.transform_positions(tracks)

    tracks["ball"] = tracker.interpolate_ball(tracks["ball"]) #Interpolation for ball for better tracking

    #Estimate speed and distance of players in middle region (most of the frame)
    estimator = SpeedAndDistance()
    estimator.measure_speed_and_distance(tracks)

    #Player team assignment
    team_assigner = TeamAssignment()
    team_assigner.assign_team(frames[0], tracks['players'][0])

    for frame, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(frames[frame], track['bbox'], player_id)
            tracks['players'][frame][player_id]['team'] = team
            tracks['players'][frame][player_id]['team_color'] = team_assigner.team_colors[team]
    
    #Assigning who has the ball per frame and calculating team ball control accordingly
    ball_assigner = PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = ball_assigner.assign_ball(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)

    #Adding annotations and indicators to the video
    output_frames = tracker.draw_annotations(frames, tracks, team_ball_control)

    #Adding camera movement display
    output_frames = camera_movement.draw_camera_movement(output_frames, camera_movements)

    #Adding speed and distance display
    estimator.draw_speed_and_distance(output_frames,tracks)
    
    #saving final output
    save_video(output_frames, 'output/output_video.avi')

if __name__ == '__main__':
    main()