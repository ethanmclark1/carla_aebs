# Autonomous Emergency Braking System 
## Overview
The Autonomous Emergency Braking System described here integrates two neural networks for enhanced vehicle safety and lane adherence:

**1. Convolutional Neural Network (CNN) for Lane Following**:
- This network is responsible for determining the steering angle.
- The output is based on visual inputs, ensuring the vehicle remains centered within its lane.

**2. Dueling Deep Q-Network (Dueling DQN)**:
- This network receives the steering angle from the CNN, combined with data on velocity and distance.
- It then computes Q-values for potential actions.
- The action with the highest Q-value is selected, guiding the vehicle's next move.

Together, these networks enable the vehicle to autonomously apply brakes as needed, preventing collisions with vehicles ahead while ensuring consistent lane positioning.

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
