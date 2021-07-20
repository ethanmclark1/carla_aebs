import pdb
import time
import torch
import random
import itertools
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from collections import deque
from lane_follower.LF_model import LaneFollower

ALPHA = 5e-3
DECAY_FACTOR = 0.9875
EPSILON = 1.0
GAMMA = 0.95
MIN_EPSILON = 0.01
REPLAY_SIZE = 10_000
REPLAY_BATCH = 64
REPLACE_COUNT = 25
TRAUMA_SIZE = 1_000
TRAUMA_BATCH = 25

class DuelingDQN(nn.Module):
    def __init__(self, state_size=3, n_actions=4):
        super(DuelingDQN, self).__init__()
        self.lane_follow = LaneFollower()
        self.lane_follow.load_state_dict(torch.load('./models/learned_model.pth'))
        self.lane_follow.eval()
        
        self.fc = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.value = nn.Linear(32, 1)
        self.advantage = nn.Linear(32, n_actions)

        self.loss = nn.MSELoss()
        self.optim = Adam(self.fc.parameters(), lr=ALPHA)

    def forward(self, rgb_image, distance, speed):
        rgb_flattened = rgb_image.flatten()
        lane_follow_input = torch.zeros([1,1,200,75])
        for i in range(200):
            for j in range(75):
                lane_follow_input[0][0][i][j] = 1 if rgb_flattened[i + j * 200] > 0 else 0

        with torch.no_grad():
            steering_angle = self.lane_follow(lane_follow_input).to('cpu').squeeze().item()
        
        fc_input = torch.FloatTensor([steering_angle, distance, speed])
        fc_output = self.fc(fc_input)

        value = self.value(fc_output)
        advantage = self.advantage(fc_output)
        q_value = value + (advantage - advantage.mean())
        return q_value

class Agent():
    def __init__(self, state_size=3, n_actions=4):
        self.epsilon = EPSILON
        self.n_actions = n_actions
        self.state_size = state_size
        self.network = DuelingDQN(state_size, n_actions)
        self.target = DuelingDQN(state_size, n_actions)
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
        with torch.no_grad():
            if np.random.random() > self.epsilon:
                q_vals = self.network(rgb_image, distance, speed)
                action = torch.argmax(q_vals).item()
            else:
                action = np.random.choice(self.n_actions)
        return action

    def learn(self):
        if len(self.replay_mem) < REPLAY_BATCH \
            or len(self.trauma_mem) < TRAUMA_BATCH:
            return

        record_loss = []
        batch = random.sample(self.replay_mem, REPLAY_BATCH)
        trauma_batch = random.sample(self.trauma_mem, TRAUMA_BATCH)
        combined_batch = itertools.zip_longest(batch, trauma_batch)

        for minibatch, trauma_minibatch in combined_batch:
            self.network.optim.zero_grad()

            rgb_image = minibatch[0][0]; distance = minibatch[0][1]; velocity = minibatch[0][2]
            reward = minibatch[2]
            next_rgb_image = minibatch[3][0]; next_distance = minibatch[3][1]; next_velocity = minibatch[3][2]
            done = minibatch[4]

            network_qval = self.network(rgb_image, distance, velocity)
            target_qval = reward + (1 - done) * GAMMA * self.target(next_rgb_image, next_distance, next_velocity)
            loss = self.network.loss(target_qval, network_qval)

            if trauma_minibatch:
                trauma_rgb_image = trauma_minibatch[0][0]; trauma_distance = trauma_minibatch[0][1]; trauma_velocity = trauma_minibatch[0][2]
                trauma_reward = trauma_minibatch[2]
                trauma_next_rgb_image = trauma_minibatch[3][0]; trauma_next_distance = trauma_minibatch[3][1]; trauma_next_velocity = trauma_minibatch[3][2]
                trauma_done = trauma_minibatch[4]

                t_network_qval = self.network(trauma_rgb_image, trauma_distance, trauma_velocity)
                t_target_qval = trauma_reward + (1 - trauma_done) * GAMMA * self.target(trauma_next_rgb_image, trauma_next_distance, trauma_next_velocity)
                loss += self.network.loss(t_target_qval, t_network_qval)
            
            record_loss.append(loss)
            loss.backward()
            self.network.optim.step()

        return record_loss

    def decrement_epsilon(self):
        self.epsilon *= DECAY_FACTOR
        self.epsilon = max(self.epsilon, MIN_EPSILON)