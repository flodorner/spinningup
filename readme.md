Spinning Up in Deep RL 
==================================

This repository is forked from the original OpenAI [spinningup repository](https://github.com/openai/spinningup) to implement changes required for the [State Augmented Contstrained RL](https://github.com/flodorner/Augmented_Constrained_RL), as well as the more robust adaptive entropy penalty for SAC.

SpinningUp contains a code repo of the implementation of key Reinforcement Learning algorithms including Soft Actor-Critic, Proximal Policy Optimization and Twin Delayed DDPG used in [Augmented Contstrained RL](https://github.com/flodorner/Augmented_Constrained_RL) project. Visit [spinningup.openai.com](https://spinningup.openai.com)! for more informtaion on Spinning Up

## Installation

First Install OpenMPI:
```
sudo apt-get update && sudo apt-get install libopenmpi-dev
```
Now install the spinningup repo:
```
git clone https://github.com/flodorner/spinningup.git
cd spinningup
pip install -e .
```

## Changes

The following changes were implemented in this fork:

- Added GPU support for PyTorch to SAC, PPO and TD3.

- Implemented adaptive entropy penalty for SAC as an alternative to the fixed entropy regularization parameter. More information about the entropy constraint can be found in this [paper](https://arxiv.org/abs/1812.05905).

- Implemented data augmentation using the known cost dynamics to boost sample efficiency.

- Added adaptive sampling data augmentation to prevent bias in data augmentation.




