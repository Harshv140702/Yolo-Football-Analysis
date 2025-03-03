class PlayerBallAssigner():
    def __init__(self):
        self.max_distance = 70
    
    def assign_ball(self,players,bbox):
        ball_x = int((bbox[0]+bbox[2])/2)
        ball_y = int((bbox[1]+bbox[3])/2)

        miniumum_distance = 99999
        assigned_player=-1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = ((player_bbox[0]-ball_x)**2+(player_bbox[-1]-ball_y)**2)**0.5
            distance_right = ((player_bbox[2]-ball_x)**2+(player_bbox[-1]-ball_y)**2)**0.5
            distance = min(distance_left,distance_right)

            if distance < self.max_distance:
                if distance < miniumum_distance:
                    miniumum_distance = distance
                    assigned_player = player_id

        return assigned_player