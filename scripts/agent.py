import torch
import random
import itertools
import numpy as np
from collections import deque
from architecture.emergency_braking import EmergencyBraking

ACTION_THRESHOLD = 3.15
CRASH_DISTANCE = 1.35

DECAY_FACTOR = 0.9875
EPSILON = 1.0
GAMMA = 0.95
MIN_EPSILON = 0.01

BATCH_SIZE = 32
MEM_SIZE = 10_000
REPLACE_COUNT = 1_000
TRAUMA_MEM_SIZE = 1_000
TRAUMA_BATCH_SIZE = 10

class Agent():
    def __init__(self, state_size=3, n_actions=4):
        self.epsilon = EPSILON
        self.n_actions = n_actions
        self.state_size = state_size
        self.network = EmergencyBraking(state_size, n_actions)
        self.target = EmergencyBraking(state_size, n_actions)
        self.replay_mem = deque(maxlen=MEM_SIZE)
        self.trauma_mem = deque(maxlen=TRAUMA_MEM_SIZE)
    
    def update_target(self, episode):
        if episode > 0 and episode % REPLACE_COUNT == 0:
            self.target.load_state_dict(self.network.state_dict())

    def remember(self, transition, distance):
        if distance < CRASH_DISTANCE:
            self.trauma_mem.append(transition)
        else:
            self.replay_mem.append(transition)

    def get_action(self, state, inference):
        with torch.no_grad():
            if inference or np.random.random() > self.epsilon:
                q_vals = self.network(state)
                action = torch.argmax(q_vals).item()
            else:
                action = np.random.choice(self.n_actions)
        return action

    def learn(self):
        if len(self.replay_mem) < BATCH_SIZE \
            or len(self.trauma_mem) < TRAUMA_BATCH_SIZE:
            return

        batch = random.sample(self.replay_mem, BATCH_SIZE)
        trauma_batch = random.sample(self.trauma_mem, TRAUMA_BATCH_SIZE)
        combined_batch = itertools.zip_longest(batch, trauma_batch)

        for minibatch, trauma_minibatch in combined_batch:
            self.network.optim.zero_grad()

            state = minibatch[0]; reward = minibatch[2]; 
            next_state = minibatch[3]; done = minibatch[4]
            network_qval = self.network(state)
            target_qval = reward + (1 - done) * GAMMA * self.target(next_state)
            loss = self.network.loss(target_qval, network_qval)

            if trauma_minibatch:
                trauma_state = trauma_minibatch[0]; trauma_reward = trauma_minibatch[2]; 
                trauma_next_state = trauma_minibatch[3]; trauma_done = trauma_minibatch[4]
                trauma_network_qval = self.network(trauma_state)
                trauma_target_qval = trauma_reward + (1 - trauma_done) * GAMMA * self.target(trauma_next_state)
                loss += self.network.loss(trauma_target_qval, trauma_network_qval)   

            loss.backward()
            self.network.optim.step()

    def decrement_epsilon(self):
        self.epsilon *= DECAY_FACTOR
        self.epsilon = max(self.epsilon, MIN_EPSILON)
