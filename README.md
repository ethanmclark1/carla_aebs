# Advanced Emergency Braking System 
## Introduction
This implementation of an Advanced Emergency Braking System is an ensemble of two networks, there is a CNN based architecture for lane following that outputs a steering angle which is grouped with velocity and distance as input for the following network, Dueling DQN. The Dueling DQN calculates the Q values for each action and then argmax is taken to get the most useful action.

Using this ensembled network, the ego car is able to apply the brakes on-time to avoid crashing into lead car; it is able to maintain its position in middle of lane.


## Setup Environment
- [Install Anaconda for Linux](https://docs.anaconda.com/anaconda/install/linux/)
- [Install CARLA for Linux](https://carla.readthedocs.io/en/0.9.11/start_quickstart/)
- `conda create --name <env> --file requirements.txt`

## Run AEBS
- Open a terminal and enter command: `./CarlaUE.sh -opengl` to run CARLA simulator
- Activate conda environment in another terminal then enter command: `python3 driver.py`
- To visualize the agent in action, open a third terminal and enter the command: `sudo docker run -it --network="host" -e CARLAVIZ_HOST_IP=localhost -e CARLA_SERVER_IP=localhost -e CARLA_SERVER_PORT=2000 mjxu96/carlaviz:0.9.11` then open a browser and go to localhost: `127.0.0.1:8080/`

## References
1. [Chae, Hyunmin, et al. "Autonomous braking system via deep reinforcement learning." 2017 IEEE 20th International conference on intelligent transportation systems (ITSC). IEEE, 2017.](https://arxiv.org/abs/1702.02302)
2. [Bojarski, Mariusz, et al. "End to end learning for self-driving cars." arXiv preprint arXiv:1604.07316 (2016).](https://arxiv.org/abs/1604.07316)
3. [Wang, Ziyu, et al. "Dueling network architectures for deep reinforcement learning." International conference on machine learning. PMLR, 2016.](https://arxiv.org/abs/1511.06581)
