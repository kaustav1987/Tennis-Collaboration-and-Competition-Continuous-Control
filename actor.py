import torch
import torch.nn as nn
from torch.optim import Adam
from memory import ReplayBuffer
from model import ActorNN,CriticNN
from noise import OUNoise
import numpy as np
import torch.nn.functional as F

class Actor():
    def __init__(self, action_size, state_size,buffer_size, batch_size,actor_lr,critic_lr,device,weight_decay, tau,shared_memory,noise,
    share_memory_flag, seed=0):
        self.state_size  = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size  = batch_size
        self.actor_lr = actor_lr
        self.weight_decay = weight_decay
        self.device = device
        self.seed= seed
        self.actor_loss =[]
        #self.critic_loss =[]
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.tau = tau
        self.noise= OUNoise(self.action_size,self.seed)
        #self.noise = noise
        self.share_memory_flag = share_memory_flag
        if self.share_memory_flag:
            self.memory = shared_memory
        else:
            self.memory = ReplayBuffer(action_size, buffer_size, batch_size, self.device)

        ## Actor
        self.actor_local = ActorNN(self.state_size,self.action_size).to(self.device)
        self.actor_target = ActorNN(self.state_size,self.action_size).to(self.device)
        self.actor_optimizer = Adam(self.actor_local.parameters(), lr = self.actor_lr)
        ## Critic
        #self.critic_local = Critic(self.state_size,self.action_size).to(self.device)
        #self.critic_target = Critic(self.state_size,self.action_size).to(self.device)
        #self.critic_optimizer = Adam(self.critic_local.parameters(), lr = self.critic_lr,  weight_decay=self.weight_decay)
        # initialize targets same as original networks
        self.hard_update(self.actor_target, self.actor_local)
        #self.hard_update(self.critic_target, self.critic_local)

    def reset(self):
        self.noise.reset()

    def act(self, state,noise = True,sd=1e-4):
        state = torch.from_numpy(state).float().to(self.device)

        self.actor_local.eval()
        with torch.no_grad():
            #print(state.shape)
            action = self.actor_local(state).cpu().data.numpy()
            ##action.cpu().detach().numpy()
        self.actor_local.train()

        if noise:
            #print(type(action))
            #action += np.random.normal(loc=0.0, scale=sd, size=action.size)
            action += self.noise.sample()
        action = np.clip(action, -1,1).reshape(1,-1)
        return action




    def hard_update(self,target, source):
        """
        Copy network parameters from source to target
        Inputs:
            target (torch.nn.Module): Net to copy parameters to
            source (torch.nn.Module): Net whose parameters to copy
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def step(self, state, action, rewards, next_state, done,GAMMA=1.0):
        ## As per the description we are not supposed to use discount factor
        self.memory.add(state, action, rewards, next_state, done)
