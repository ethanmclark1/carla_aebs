import os
import pdb
import dqn
import time
import torch
import environment
from threading import Thread

EPISODES = 100
MIN_REWARD = -200
MODEL_NAME = 'AEBS'
AGGREGATE_STATS_EVERY = 10

def init_cuda():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')

    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    
    return device

# TODO: Setup multithread, setup Tensorboard, configure GPU
if __name__ == '__main__':
    device = init_cuda()
    if not os.path.isdir('models'):
        os.makedirs('models')

    total_reward = [-200]
    env = environment.Env()
    agent = dqn.Agent(device)

    for episode in range(EPISODES):
        episodic_return = 0
        depth_image, ego_speed = env.reset()
        done = False

        while True:
            distance = env.get_distance(env.semantic_image)
            action = agent.get_action(depth_image, distance, ego_speed)
            next_depth_image, reward, done, _ = env.step(action, distance)
            episodic_return += reward
            
            state = (depth_image, distance, ego_speed)
            next_state = (next_depth_image, distance, ego_speed)

            if distance < 1.25:
                agent.remember_trauma([state, action, reward, next_state, done])
            else:
                agent.remember([state, action, reward, next_state, done])

            agent.update_target(episode)
            agent.learn()
            if done: 
                break

            depth_image = next_depth_image

        for actor in env.actor_list:
            try:
                if actor.attributes.get('role_name') == 'autopilot':
                    actor.set_autopilot(False)
            except:
                actor.stop()
            actor.destroy()

        total_reward.append(episodic_return)
        agent.decrement_epsilon()


    # PATH = './aebs_network.pth'
    # torch.save(agent.network.state_dict(), PATH)
