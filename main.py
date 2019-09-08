from unityagents import UnityEnvironment
import numpy as np
import torch
from actor import Actor
from critic import Critic
from memory import ReplayBuffer
from noise import OUNoise
from collections import namedtuple, deque
import os

def main():
    env = UnityEnvironment(file_name="./Tennis.exe")
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    ##Hyperparameters
    BUFFER_SIZE = int(1e6)  # Total buffer size
    BATCH_SIZE = 128        # minibatch size
    GAMMA = 0.99            # discount factor
    TAU = 2e-1              # for soft update of target parameters
    LR_ACTOR = 1e-4         # learning rate for Actor
    LR_CRITIC = 3e-4        # learning rate for Critic
    WEIGHT_DECAY = 0        # L2 weight decay
    sd= 1e-4                ##Standard deviation of noise not used
    share_memory_flag= True

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    n_agents = num_agents
    # size of each action
    action_size = brain.vector_action_space_size

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]

    shared_memory = ReplayBuffer( action_size,BUFFER_SIZE, BATCH_SIZE,device)
    noise = OUNoise(action_size, 0)

    multi_agents_actor = []
    for i in range(n_agents):
        ##2 actors
        multi_agents_actor.append(Actor(action_size, state_size,BUFFER_SIZE, BATCH_SIZE,LR_ACTOR,LR_CRITIC,device,
                                  WEIGHT_DECAY,TAU,shared_memory,noise,share_memory_flag,seed=i))

    ##One Critic for 2 actors
    multi_agents_critic= Critic(action_size, state_size,BUFFER_SIZE, BATCH_SIZE,LR_ACTOR,LR_CRITIC,device,
                                  WEIGHT_DECAY,TAU,share_memory_flag,seed=0)

    scores_deque = deque(maxlen = 100)
    scores = []
    n_episodes=3000
    best_score = -np.inf

    ##Kaustav

    ##Loaded last trained model
    for i in range(n_agents):
        ##Load 2 actors
        actor_checkpoint_filename = 'actor'+ str(i) + '.pth'

        ##Load the trained actor and critic to continue training
        if os.path.exists(os.path.join('./'+ actor_checkpoint_filename)):
            #multi_agents_actor[i].actor_local.load_state_dict(torch.load(os.path.join('./'+ actor_checkpoint_filename)))
            print('\tAgent',i+1,'.Actor Loaded ', end ="")

    critic_checkpoint_filename = 'critic'+ '0' + '.pth'
    if os.path.exists(os.path.join('./'+ critic_checkpoint_filename)):
        #multi_agents_critic.critic_local.load_state_dict(torch.load(os.path.join('./'+ critic_checkpoint_filename)))
        print('\tCritic Loaded', end= '\n')

    for episode in range(1, n_episodes+1):

        env_info = env.reset(train_mode=True)[brain_name]     # reset the environment
        states = env_info.vector_observations                 # get the current state (for each agent)
        score = np.zeros(num_agents)                          # initialize the score (for each agent)

        ##reset the noise
        for i in range(n_agents):
            multi_agents_actor[i].reset()

        while True:
            #print(episode)
            actions =[]
            ##Deal with each agent separately
            for i in range(n_agents):

                #multistate[i] = states[i]

                #states = states.reshape(1,-1)
                ##action = multi_agents[i].act(states[i].reshape(1,-1))
                action = multi_agents_actor[i].act(states[i], noise= False)
                actions.append(action)

            actions = np.vstack([action for action in actions])
            env_info = env.step(actions.reshape(-1))[brain_name]           # send all actions to tne environment

            next_states = env_info.vector_observations         # get next state (for each agent)
            #next_states = next_states.reshape(1,-1)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished

            ##Deal with each agent separately
            for i in range(n_agents):
                #print(states[i].reshape(1,-1).shape,
                #      next_states[i].reshape(1,-1).shape,actions[i].reshape(1,-1).shape )
                #print(actions[i].shape)

                multi_agents_actor[i].step(states[i].reshape(1,-1), actions[i].reshape(1,-1), rewards[i],
                                     next_states[i].reshape(1,-1), dones[i])
            multi_agents_critic.step(multi_agents_actor[0], shared_memory,GAMMA)
            multi_agents_critic.step(multi_agents_actor[1], shared_memory,GAMMA)
            score += rewards                                    # update the score (for each agent)
            states = next_states                                 # roll over states to next time step
            if np.any( dones ):
                break
            ## End of 1 episode
        scores.append(np.max(score))
        scores_deque.append(np.max(score))
        print('\rEpisode: \t{} \tScore: \t{:.4f} \tAverage Score: \t{:.4f}'.format(episode, np.max(score),
                                                                                       np.mean(scores_deque)), end="")
        if  np.max(score) > best_score:
            best_score = np.max(score)
            ##Save Actor and Critic for each agents
            checkpoint(n_agents, multi_agents_actor, multi_agents_critic)
        if  np.mean(scores_deque) > 0.5:
            print('\nEnvironment solved in episode \t{}'.format(episode) )
            checkpoint(n_agents, multi_agents_actor, multi_agents_critic)
            break

def checkpoint(n_agents, actor, critic):
    for i in range(n_agents):
        actor_checkpoint_filename = 'actor'+ str(i) + '.pth'

        torch.save(actor[i].actor_local.state_dict(), actor_checkpoint_filename)
    critic_checkpoint_filename = 'critic'+ '0' + '.pth'
    torch.save(critic.critic_local.state_dict(), critic_checkpoint_filename)

if __name__ == '__main__':
    main()
