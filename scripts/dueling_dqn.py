import pdb
import torch
import random
import itertools
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from collections import deque
from lane_follower.LF_model import LaneFollower

ALPHA = 8e-3
BATCH_SIZE = 32
CRASH_DISTANCE = 1.35
DECAY_FACTOR = 0.9875
EPSILON = 1.0
GAMMA = 0.95
MIN_EPSILON = 0.01
MEM_SIZE = 5_000
REPLACE_COUNT = 50

class DuelingDQN(nn.Module):
    def __init__(self, state_size=3, n_actions=4):
        super(DuelingDQN, self).__init__()
        self.lane_follow = LaneFollower()
        self.lane_follow.load_state_dict(torch.load('./models/learned_model.pth'))
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
        self.optim = Adam([*self.fc.parameters(), *self.value.parameters(), *self.advantage.parameters()], lr=ALPHA)

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

class Agent():
    def __init__(self, state_size=3, n_actions=4):
        self.epsilon = EPSILON
        self.n_actions = n_actions
        self.state_size = state_size
        self.network = DuelingDQN(state_size, n_actions)
        self.target = DuelingDQN(state_size, n_actions)
        self.replay_mem = deque(maxlen=MEM_SIZE)
        self.trauma_mem = deque(maxlen=MEM_SIZE)
    
    def update_target(self, episode):
        if episode > 0 and episode % REPLACE_COUNT == 0:
            self.target.load_state_dict(self.network.state_dict())

    def remember(self, transition, distance):
        if distance < CRASH_DISTANCE:
            self.trauma_mem.append(transition)
        else:
            self.replay_mem.append(transition)

    def get_action(self, state):
        with torch.no_grad():
            if np.random.random() > self.epsilon:
                q_vals = self.network(state)
                action = torch.argmax(q_vals).item()
            else:
                action = np.random.choice(self.n_actions)
        return action

    def learn(self):
        if len(self.replay_mem) < BATCH_SIZE \
            or len(self.trauma_mem) < BATCH_SIZE:
            return

        record_loss = []
        batch = random.sample(self.replay_mem, BATCH_SIZE)
        trauma_batch = random.sample(self.trauma_mem, BATCH_SIZE)
        combined_batch = itertools.zip_longest(batch, trauma_batch)

        for minibatch, trauma_minibatch in combined_batch:
            self.network.optim.zero_grad()

            state = minibatch[0]; reward = minibatch[2]; 
            next_state = minibatch[3]; done = minibatch[4]
            trauma_state = trauma_minibatch[0]; trauma_reward = trauma_minibatch[2]; 
            trauma_next_state = trauma_minibatch[3]; trauma_done = trauma_minibatch[4]

            network_qval = self.network(state)
            target_qval = reward + (1 - done) * GAMMA * self.target(next_state)
            loss = self.network.loss(target_qval, network_qval)
            trauma_network_qval = self.network(trauma_state)
            trauma_target_qval = trauma_reward + (1 - trauma_done) * GAMMA * self.target(trauma_next_state)
            loss += self.network.loss(trauma_target_qval, trauma_network_qval)
                
            record_loss.append(loss)
            loss.backward()
            self.network.optim.step()

        return record_loss

    def decrement_epsilon(self):
        self.epsilon *= DECAY_FACTOR
        self.epsilon = max(self.epsilon, MIN_EPSILON)