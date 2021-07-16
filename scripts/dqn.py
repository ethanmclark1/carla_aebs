import time
import pdb
import torch
import random
import itertools
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from collections import deque
from lane_follower.model import LaneFollower

ALPHA = 5e-3
DECAY_FACTOR = 0.90
EPSILON = .95
GAMMA = .99
MIN_EPSILON = 0.01
REPLAY_SIZE = 10_000
REPLAY_BATCH = 32
REPLACE_COUNT = 100
TRAUMA_SIZE = 1_000
TRAUMA_BATCH = 10

class DQN(nn.Module):
    def __init__(self, device, state_size=3, n_actions=4):
        super(DQN, self).__init__()
        self.lane_follow = LaneFollower()
        self.lane_follow.load_state_dict(torch.load('./lane_follower/learned_model.pth'))
        self.lane_follow.eval()
        
        self.aebs = nn.Sequential(
            nn.Linear(state_size, 12),
            nn.ReLU(),
            nn.Linear(12, 24),
            nn.ReLU(),
            nn.Linear(24, n_actions)
        )

        self.device = device
        self.loss = nn.MSELoss()
        self.optim = Adam(self.aebs.parameters(), lr=ALPHA)

    def forward(self, rgb_image, distance, speed):
        rgb_flattened = rgb_image.flatten()
        lane_follow_input = torch.zeros([1,1,200,75], device=self.device)
        for i in range(200):
            for j in range(75):
                lane_follow_input[0][0][i][j] = 1 if rgb_flattened[i + j * 200] > 0 else 0

        with torch.no_grad():
            steering_angle = self.lane_follow(lane_follow_input).to('cpu').squeeze().item()

        aebs_input = torch.FloatTensor([steering_angle, distance, speed]).to(self.device)
        brake_force = self.aebs(aebs_input)
        return brake_force

class Agent():
    def __init__(self, device, state_size=3, n_actions=4):
        self.epsilon = EPSILON
        self.n_actions = n_actions
        self.state_size = state_size
        self.network = DQN(device, state_size, n_actions)
        self.network.to(torch.device(device))
        self.target = DQN(device, state_size, n_actions)
        self.target.to(torch.device(device))
        self.replay_mem = deque(maxlen=REPLAY_SIZE)
        self.trauma_mem = deque(maxlen=TRAUMA_SIZE)
    
    def update_target(self, episode):
        if episode > 0 and episode % REPLACE_COUNT == 0:
            self.target.load_state_dict(self.network.state_dict())

    def remember(self, transition):
        self.replay_mem.append(transition)

    def remember_trauma(self, transition):
        self.trauma_mem.append(transition)

    def get_action(self, rgb_image, distance, speed):
        FPS = 20
        with torch.no_grad():
            if np.random.random() > self.epsilon:
                q_vals = self.network(rgb_image, distance, speed)
                action = torch.argmax(q_vals).item()
            else:
                action = np.random.choice(self.n_actions)
                time.sleep(1/FPS)
        return action

    def learn(self):
        if len(self.replay_mem) < REPLAY_BATCH \
            or len(self.trauma_mem) < TRAUMA_BATCH:
            return

        batch = random.sample(self.replay_mem, REPLAY_BATCH)
        t_batch = random.sample(self.trauma_mem, TRAUMA_BATCH)
        combined_batch = itertools.zip_longest(batch, t_batch)

        for minibatch, t_minibatch in combined_batch:
            state = minibatch[0]; reward = minibatch[2]
            next_state = minibatch[3]; done = minibatch[4]

            t_state = t_minibatch[0]; t_reward = t_minibatch[2]
            t_next_state = t_minibatch[3]; t_done = t_minibatch[4]

            network_qval = self.network(state)
            target_qval = reward + (1 - done) * GAMMA * self.target(next_state)

            t_network_qval = self.network(t_state)
            t_target_qval = t_reward + (1 - t_done) * GAMMA * self.target(t_next_state)

            self.network.optim.zero_grad()
            loss = self.network.loss(target_qval, network_qval)
            loss += self.network.loss(t_target_qval, t_network_qval)
            loss.backward()
            self.network.optim_step()

    def decrement_epsilon(self):
        self.epsilon *= DECAY_FACTOR
        self.epsilon = max(self.epsilon, MIN_EPSILON)
