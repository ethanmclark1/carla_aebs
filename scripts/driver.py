import os
import pdb
import dueling_dqn
import time
import torch
import tensorboard
import environment

EPISODES = 4_000

if __name__ == '__main__':
    if not os.path.isdir('models'):
        os.makedirs('models')

    agent = dueling_dqn.Agent()
    env = environment.Env()
    start_time = time.time()

    episode = 0
    total_loss = []
    total_reward = []

    while episode < EPISODES:
        episodic_return = 0
        state, skip_episode = env.reset()
        if skip_episode:
            continue

        done = False
        while True:
            action = agent.get_action(state)
            next_state, reward, done, skip_episode = env.step(action)
            episodic_return += reward

            if (episodic_return < -40 and episodic_return > -100) \
                or episodic_return < -160:
                skip_episode = True
            if skip_episode:
                break

            agent.remember([state, action, reward, next_state, done], env.distance)
            agent.update_target(episode)
            total_loss.append(agent.learn())
            agent.decrement_epsilon()
            state = next_state
            
            if done:
                break

        env.destroy_actor()
        episode += 1

        if skip_episode:
            episode -= 1
        else:
            print(f'Episode: {episode} \t Reward: {episodic_return}')
            total_reward.append(episodic_return)


    end_time = time.time()
    print(f'Training time: {end_time - start_time}')
    PATH = './models/aebs_network.pth'
    torch.save(agent.network.state_dict(), PATH)
    pdb.set_trace()
    a = 3
