import os
import pdb
import dueling_dqn
import time
import torch
import tensorboard
import environment

EPISODES = 5000
AGGREGATE_STATS_EVERY = 100

def is_car_visible(env, semantic_image):
    skip_episode = False
    distance = env.get_distance(semantic_image)
    if distance == None:
        skip_episode = True
    return distance, skip_episode

def train(agent, total_loss):
    if distance < 1.25:
        agent.remember_trauma([state, action, reward, next_state, done])
    else:
        agent.remember([state, action, reward, next_state, done])

    agent.update_target(episode)
    total_loss.append(agent.learn())
    return total_loss

if __name__ == '__main__':
    if not os.path.isdir('models'):
        os.makedirs('models')

    agent = dueling_dqn.Agent()
    env = environment.Env()
    start_time = time.time()

    total_reward = []
    total_loss = []
    episode = 0
    while episode < EPISODES:
        episodic_return = 0
        depth_image, ego_speed, skip_episode = env.reset()
        done = False
        if skip_episode:
            continue

        while True:
            distance, skip_episode = is_car_visible(env, env.semantic_image)
            if skip_episode:
                break

            action = agent.get_action(depth_image, distance, ego_speed)
            next_depth_image, reward, done, _ = env.step(action, distance)
            episodic_return += reward
            
            state = (depth_image, distance, ego_speed)

            next_distance, skip_episode = is_car_visible(env, env.semantic_image)
            if skip_episode:
                break
            
            next_state = (next_depth_image, next_distance, ego_speed)

            total_loss = train(agent, total_loss)
            if done: 
                env.destroy_actor()
                break

            depth_image = next_depth_image
        
        if skip_episode:
            episode -= 1
        else:
            print(f'Episode: {episode} \t Reward: {episodic_return}')
            total_reward.append(episodic_return)

        episode += 1
        env.destroy_actor()
        agent.decrement_epsilon()

    end_time = time.time()
    print(f'Training time: {end_time - start_time}')
    pdb.set_trace()
    PATH = './models/aebs_network.pth'
    torch.save(agent.network.state_dict(), PATH)
