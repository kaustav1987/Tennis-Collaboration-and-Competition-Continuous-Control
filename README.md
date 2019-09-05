<img src="https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif">   

### Deep Reinforcement Learning Tennis-Collaboration and Competition Continuous Control   
### Introduction
In this project, I build a reinforcement learning (RL) agent that controls 2 Tennis Playing Agents within Unity's [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment. 
The goal is to get 2 Tennis Playing Agents play tennis in collaboration as long as possible, i.e. the longer the rally the better

The task solved here refers to a collaboration continuous control problem where two agents must be able to play "tennis" in collaboration, that is, the longer the rally goes the higher will be the reward that both agents will earn.

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

2. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file.
### Requeriments:
- tensorflow: 1.7.1
- Pillow: 4.2.1
- matplotlib
- numpy: 1.11.0
- pytest: 3.2.2
- docopt
- pyyaml
- protobuf: 3.5.2
- grpcio: 1.11.0
- torch: 0.4.1
- pandas
- scipy
- ipykernel
- jupyter: 5.6.0

I have also created a requirement.txt. You may do a pip install requirement.txt to install all the required packages.

### Execution 
To execute my code enter the command 'python main.py'

## Summary of Environment
- Set-up: 2 Table Tennis playing agent who want to learn playing table tennis rally as long as possible.
- Goal: Each agent must move its hand to the goal location, to hit the ball on the board over the net and inside the bound.
- Agents: The environment contains 2 agents linked to a single Brain.
- Agent Reward Function :
  - If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.
  After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.This yields a single score for each episode.
- Brains: One Brain with the following observation/action space.
  - Vector Observation space: 24 variables corresponding to position, rotation, velocity, and angular velocities of the ball.
  - Vector Action space: (Continuous) Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. Every entry in the action vector should be a number between -1 and 1.
  - Visual Observations: None.
- Benchmark Mean Reward: +0.5

## Approach
Here are the high-level steps taken in building an agent that solves this environment.

1. Evaluate the state and action space.
1. Establish performance baseline using a random action policy.
1. Select an appropriate algorithm and begin implementing it.
1. Run experiments, make revisions, and retrain the agent until the performance threshold is reached.
## Solution:
To get started, there are a few high-level architecture decisions we need to make. First, we need to determine which types of algorithms are most suitable for the Reacher environment. Second, we need to determine how many "brains" we want controlling the actions of our agents.

There are 2 main differences in the Reacher Environment:
1> Continuous Actions
2> Multi Agent

The value based methods (DQN etc) is not suitable for continuous action space.
Policy based menthods are well suited for this purpose.

But Policy based menthods uses Monte Carlo menthods. THis increases variance and we had to wait for the entire episode to complete to do the training. So I wanted to use a Policy based menthod where we will Temporal difference so that we can training for all the timeframe and the convergence is fast.
The algorithm I choose is Deep Deterministic Policy Gradient (DDPG). It is a model-free, off-policy actor-critic algorithm using deep function approximators that can learn policies in high-dimensional, continuous action spaces.

Using a policy-based approach, the agent (actor) learns how to act by directly estimating the optimal policy.Using a value-based approach, the agent (critic) learns how to estimate the Q value (reward). So we moved ahead of Monte Carlo which leads to too much ocillation.

The agent has different continous values as actions(between -1 and 1),corresponding to movement toward (or away from) the net, and jumping. We cant use So we used Ornstein-Uhlenbeck process. Else the noise would only be in 1 direction directly corellated to the previous direction. Even the random Standard normal noise with a small sd does not help. 

I have used 2 Actor. 1 shared memory is used, so that the batch size gets filled earlier and the learning can start early. Moreover both of them can learn from each other's experience. Since the State is 24 length vector, it doesnt matter the side from which one Actor is playing. The Action is just a vector of 2 values. So one critic should be good enough to tell us the Q values for the action selection. I tried with 2 critics. It did not gave me good result.In my code implementation, I have given the ability to use or not use shared memory. You can control it using shared memory flag.The Noise can also be turned on or off using the noise parameter of the Actor.

### The hyperparameters: 
- Learning Rate: 1e-4 (in both Actor)  - Learning Rate: 3e-4 (in Critic)  - Batch Size: 128   - Replay Buffer: 1e6   - Gamma: 0.99   - Tau: 2e-1   For the neural models:       - Actor         - Hidden: (input, 512)  - ReLU     - Hidden: (512,256)    - ReLU     - Output: (256,2)  - TanH. The action output is clipped between -1 and 1.   - Critic     - Hidden: (input, 512)              - ReLU     - Hidden: (512 + action_size, 256)  - ReLU     - Output: (256, 1)

### Score vs Episode with noise and Separate Memory for each Actor 
<img src="https://github.com/kaustav1987/Tennis-Collaboration-and-Competition-Continuous-Control/blob/master/with%20noise%20and%20individual%20memory.png"> 
It took me 800+ episodes to get the desired score

### Score vs Episode without noise and Individual Memory for each Actor
<img src="https://github.com/kaustav1987/Tennis-Collaboration-and-Competition-Continuous-Control/blob/master/individual%20memory%20no%20noise.png"> 
It took me 413 episodes to get the desired score 

### Score vs Episode without noise and Shared Memory between Actor
<img src="https://github.com/kaustav1987/Tennis-Collaboration-and-Competition-Continuous-Control/blob/master/shared%20memory%20no%20Noise.png"> 
It took me 380+ episodes to get the desired score 

### Future Improvement
I stil want to check this task with the A2C , D4PG algorithm and discover when and where each of the algorithms (DDPG vs. D4PG) have the best performance. I also want to check if using Advantage Critic benefits this task. I want to explore with experienced replay. We may learn more from rare but important events in that case.I think we may also try N-Step boostrapping instead 1 step for bias-variance tradeoff. I think the reward calculation using GAE may also benefit.



