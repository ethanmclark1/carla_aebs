import torch
import torch.nn as nn
from torch.optim import Adam
from architecture.lane_follower import LaneFollower

class EmergencyBraking(nn.Module):
    def __init__(self, state_size=3, n_actions=4, alpha=5e-4):
        super(EmergencyBraking, self).__init__()
        self.lane_follow = LaneFollower()
        self.lane_follow.load_state_dict(torch.load('./trained_model/lane_follower.pth'))
        self.lane_follow.eval()
        
        self.fc = nn.Sequential(
            nn.Linear(state_size, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.value = nn.Sequential(nn.Linear(16, 1))
        self.advantage = nn.Sequential(nn.Linear(16, n_actions))

        self.loss = nn.MSELoss()
        self.optim = Adam([*self.fc.parameters(), *self.value.parameters(), *self.advantage.parameters()], lr=alpha)

    def forward(self, state):
        rgb_image = state[0]; distance = state[2]; kmph = state[3]
        rgb_flattened = rgb_image.flatten()
        lane_follow_input = torch.zeros([1,1,200,75])
        for i in range(200):
            for j in range(75):
                lane_follow_input[0][0][i][j] = 1 if rgb_flattened[i + j * 200] > 0 else 0

        with torch.no_grad():
            steering_angle = self.lane_follow(lane_follow_input).to('cpu').squeeze().item()
        
        fc_input = torch.FloatTensor([steering_angle, distance, kmph])
        fc_output = self.fc(fc_input)

        value = self.value(fc_output)
        advantage = self.advantage(fc_output)
        q_value = value + (advantage - advantage.mean())
        return q_value