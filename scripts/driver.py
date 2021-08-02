import os
import time
import agent
import torch
import environment
import numpy as np
import pandas as pd
from agent import REPLACE_COUNT
import matplotlib.pyplot as plt

def plot(reward, size):
    moving_average = pd.Series(reward).rolling(window=size).mean().iloc[size-1:].values
    plt.plot(reward, label='Reward')
    plt.plot(moving_average, label=f'{size} Episode Moving Average Reward', color='red')
    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('reward_data.png')

def train(transition, distance, episode):
    agent.remember(transition, distance)
    agent.update_target(episode)
    agent.learn()
    agent.decrement_epsilon()

TRAIN_LEN = 10_000
INFERENCE_LEN = 50

if __name__ == '__main__':
    if not os.path.isdir('trained_model'):
        os.makedirs('trained_model')

    actor = agent.Agent()
    env = environment.Env()
    start_time = time.time()

    episode = 0
    total_loss = []
    total_reward = []

    while episode < TRAIN_LEN:
        episodic_return = 0
        state, skip_episode = env.reset()
        if skip_episode:
            continue

        done = False
        while True:
            action = actor.get_action(state, inference=False) 
            next_state, reward, done, skip_episode = env.step(action)
            episodic_return += reward
            episodic_return = np.clip(episodic_return, a_min=-175, a_max=15)

            if (episodic_return < -40 and episodic_return > -100):
                skip_episode = True
            if skip_episode:
                break

            train([state, action, reward, next_state, done], env.distance, episode)
            state = next_state
            
            if done:
                break
            try:
                env.world.tick(7.5)
            except:
                print('WARNING: tick not recieved')

        env.destroy_actor()
        episode += 1

        if skip_episode:
            episode -= 1
        else:
            print(f'Episode: {episode} \t Reward: {episodic_return}')
            total_reward.append(episodic_return)


    end_time = time.time()
    print(f'Training time: {end_time - start_time}')
    PATH = './trained_model/emergency_braking.pth'
    torch.save(actor.network.state_dict(), PATH)
    plot(total_reward, REPLACE_COUNT)
