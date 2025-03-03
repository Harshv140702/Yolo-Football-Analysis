from sklearn.cluster import KMeans

class TeamAssignment:
    def __init__(self):
        self.team_colors = {}
        self.player_team = {}
    
    #Replicating the code written in player_color_assignment notebook
    def get_team(self, frame, bbox):
        img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half_img = img[0: int(img.shape[0]/2), :]
        img_2d = top_half_img.reshape(-1,3)
        kmeans = KMeans(n_clusters=2, init="k-means++",random_state=0).fit(img_2d)
        labels = kmeans.labels_
        img_clustered = labels.reshape(top_half_img.shape[0], top_half_img.shape[1]) #reshaping it into original image
        corners = [img_clustered[0,0],img_clustered[0,-1],img_clustered[-1,0],img_clustered[-1,-1]]
        bg = max(set(corners), key= corners.count) #The background cluster should have the most appearances
        players = 1 - bg
        team = kmeans.cluster_centers_[players]
        return team

    def assign_team(self, frame, players):
        teams = []
        for _, player in players.items():
            bbox = player["bbox"]
            team = self.get_team(frame, bbox)
            teams.append(team)

        kmeans =KMeans(n_clusters=2, init="k-means++",random_state=0).fit(teams)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, bbox, player_id):
        if player_id in self.player_team:
            return self.player_team[player_id]
        
        player_team = self.get_team(frame, bbox)
        team = self.kmeans.predict(player_team.reshape(1,-1))[0]
        team += 1 # we want team 1 and 2 not team 0 and 1
        self.player_team[player_id] = team
        return team
