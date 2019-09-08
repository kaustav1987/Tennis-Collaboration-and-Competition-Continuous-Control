import torch
import torch.nn as nn
from torch.optim import Adam
from memory import ReplayBuffer
from model import ActorNN,CriticNN
from noise import OUNoise
import numpy as np
import torch.nn.functional as F

class Critic():
    def __init__(self, action_size, state_size,buffer_size, batch_size,actor_lr,critic_lr,device,weight_decay, tau,share_memory_flag, seed=0):
        self.state_size  = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size  = batch_size
        #self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.weight_decay = weight_decay
        self.device = device
        self.seed= seed
        self.share_memory_flag= share_memory_flag
        ##self.actor_loss =[]
        self.critic_loss =[]
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.tau = tau
        #self.noise= OUNoise(self.action_size,self.seed)

        #self.memory = ReplayBuffer(action_size, buffer_size, batch_size, self.device)
        ## Actor
        #self.actor_local = Actor(self.state_size,self.action_size).to(self.device)
        #self.actor_target = Actor(self.state_size,self.action_size).to(self.device)
        #self.actor_optimizer = Adam(self.actor_local.parameters(), lr = self.actor_lr)
        ## Critic
        self.critic_local = CriticNN(self.state_size,self.action_size).to(self.device)
        self.critic_target = CriticNN(self.state_size,self.action_size).to(self.device)
        self.critic_optimizer = Adam(self.critic_local.parameters(), lr = self.critic_lr,  weight_decay=self.weight_decay)
        # initialize targets same as original networks
        #self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)


    def soft_update(self, local_model, target_model, tau):
        for target_parm, local_parm in zip(target_model.parameters(),local_model.parameters()):
            target_parm.data.copy_(tau*local_parm.data + (1-tau)*target_parm.data)


    def hard_update(self,target, source):
        """
        Copy network parameters from source to target
        Inputs:
            target (torch.nn.Module): Net to copy parameters to
            source (torch.nn.Module): Net whose parameters to copy
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def step(self, actor,memory,GAMMA=1.0):
        ## As per the description we are not supposed to use discount factor

        ##print(state.shape,'**')
        if self.share_memory_flag:

            if (len(memory)) > self.batch_size:
                experiences = memory.sample()
                self.learn(actor,experiences,GAMMA)
        else:
            if (len(actor.memory)) > self.batch_size:
                experiences = actor.memory.sample()
                self.learn(actor,experiences,GAMMA)


    def learn(self,actor, experiences, gamma):
        state, action, rewards,next_state, done = experiences
        ##---------------------------- update critic ---------------------------- #
        # Compute critic loss
        ##Get predicted next state
        ##print('\n\n',state.shape, action.shape, rewards.shape,next_state.shape, done.shape)
        next_action = actor.actor_target(next_state)
        #print(next_state.shape)
        #print(next_action.shape)
        q_target_next = self.critic_target(next_state, next_action)

        #print(q_target_next.shape)
        #print(rewards.shape)
        #print(done.shape)

        q_target = rewards + gamma*q_target_next*(1-done)
        ##Current Q Values
        ##print(action.shape)
        q_current = self.critic_local(state,action)
        critic_loss = F.mse_loss(q_current, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        ##print('Hi')
        self.critic_loss.append(critic_loss.item())

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        action_pred = actor.actor_local(state)
        actor_loss = -self.critic_local(state,action_pred).mean()
        actor.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor.actor_optimizer.step()

        actor.actor_loss.append(actor_loss.item())
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target,self.tau)
        self.soft_update(actor.actor_local, actor.actor_target,self.tau)
