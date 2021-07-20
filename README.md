# Advanced Emergency Braking System 
## Introduction
This implementation of an Advanced Emergency Braking System is an ensemble of two networks, there is a CNN based architecture for the lane following. This outputs a steering angle that is grouped with velocity and distance to lead car as the input for a Dueling DQN. The Dueling DQN calculates the Q values for each action and then the argmax is taken to get the most useful action.

## Setup Environment
- [Install Anaconda for Linux](https://docs.anaconda.com/anaconda/install/linux/)
- [Install CARLA for Linux](https://carla.readthedocs.io/en/0.9.11/start_quickstart/)
- `conda create --name <env> --file requirements.txt`

## Run AEBS
- Open a terminal and enter the command: `./CarlaUE.sh -opengl` to run CARLA simulator
- Activate conda environment in another terminal then enter the command: `python3 driver.py`

## References
1. [Chae, Hyunmin, et al. "Autonomous braking system via deep reinforcement learning." 2017 IEEE 20th International conference on intelligent transportation systems (ITSC). IEEE, 2017.](https://arxiv.org/abs/1702.02302)
2. [Bojarski, Mariusz, et al. "End to end learning for self-driving cars." arXiv preprint arXiv:1604.07316 (2016).](https://arxiv.org/abs/1604.07316)
