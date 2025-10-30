import torch
import torch.nn as nn
import numpy as np
from base_agent import TD3BaseAgent
from models.CarRacing_model import ActorNetSimple, CriticNetSimple
from environment_wrapper.CarRacingEnv import CarRacingEnvironment
import random
from base_agent import OUNoiseGenerator, GaussianNoise


class CarRacingTD3Agent(TD3BaseAgent):

    def __init__(self, config):
        super(CarRacingTD3Agent, self).__init__(config)
        # initialize environment
        self.env = CarRacingEnvironment(N_frame=4, config=config, test=False)
        self.test_env = CarRacingEnvironment(N_frame=4,
                                             config=config,
                                             test=True)

        # behavior network
        self.actor_net = ActorNetSimple(self.env.observation_space.shape[0],
                                        self.env.action_space.shape[0], 4)
        self.critic_net1 = CriticNetSimple(self.env.observation_space.shape[0],
                                           self.env.action_space.shape[0], 4)
        self.critic_net2 = CriticNetSimple(self.env.observation_space.shape[0],
                                           self.env.action_space.shape[0], 4)
        self.actor_net.to(self.device)
        self.critic_net1.to(self.device)
        self.critic_net2.to(self.device)
        # target network
        self.target_actor_net = ActorNetSimple(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0], 4)
        self.target_critic_net1 = CriticNetSimple(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0], 4)
        self.target_critic_net2 = CriticNetSimple(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0], 4)
        self.target_actor_net.to(self.device)
        self.target_critic_net1.to(self.device)
        self.target_critic_net2.to(self.device)
        self.target_actor_net.load_state_dict(self.actor_net.state_dict())
        self.target_critic_net1.load_state_dict(self.critic_net1.state_dict())
        self.target_critic_net2.load_state_dict(self.critic_net2.state_dict())

        # set optimizer
        self.lra = config["lra"]
        self.lrc = config["lrc"]

        self.actor_opt = torch.optim.Adam(self.actor_net.parameters(),
                                          lr=self.lra)
        self.critic_opt1 = torch.optim.Adam(self.critic_net1.parameters(),
                                            lr=self.lrc)
        self.critic_opt2 = torch.optim.Adam(self.critic_net2.parameters(),
                                            lr=self.lrc)

        # choose Gaussian noise or OU noise

        # noise_mean = np.full(self.env.action_space.shape[0], 0.0, np.float32)
        # noise_std = np.full(self.env.action_space.shape[0], 1.0, np.float32)
        # self.noise = OUNoiseGenerator(noise_mean, noise_std)

        self.noise = GaussianNoise(0.0, self.exploration_noise_std,
                                   self.device)

    def decide_agent_actions(self, state, sigma=0.0, brake_rate=0.015):
        ### TODO ###
        # based on the behavior (actor) network and exploration noise
        state = torch.tensor(state, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor_net(state, brake_rate=brake_rate)
            action += sigma * self.noise.generate(action.shape)
            action = torch.clamp(
                action,
                torch.tensor(self.env.action_space.low, device=self.device),
                torch.tensor(self.env.action_space.high, device=self.device))

        return action.cpu().numpy()[0]

    def update_behavior_network(self):
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self.replay_buffer.sample(
            self.batch_size, self.device)
        ### TODO ###
        ### TD3 ###
        # 1. Clipped Double Q-Learning for Actor-Critic
        # 2. Delayed Policy Updates
        # 3. Target Policy Smoothing Regularization

        ## Update Critic ##
        # critic loss
        q_value1 = self.critic_net1(state, action)
        if self.twin:
            q_value2 = self.critic_net2(state, action)
        with torch.no_grad():
            # select action a_next from target actor network and add noise for smoothing
            a_next = self.target_actor_net(next_state)
            if self.target_policy_smoothing:
                noise = torch.clamp(self.noise.generate(a_next.shape),
                                    -self.noise_clip, self.noise_clip)
                a_next = a_next + noise

            act_low = torch.as_tensor(self.env.action_space.low,
                                      device=self.device,
                                      dtype=a_next.dtype)
            act_high = torch.as_tensor(self.env.action_space.high,
                                       device=self.device,
                                       dtype=a_next.dtype)
            a_next = torch.clamp(a_next, act_low, act_high)

            q_next1 = self.target_critic_net1(next_state, a_next)
            if self.twin:
                q_next2 = self.target_critic_net2(next_state, a_next)
                q_next = torch.min(q_next1, q_next2)
            else:
                q_next = q_next1
            q_target = reward + self.gamma * (1 - done) * q_next

        # critic loss function
        criterion = nn.MSELoss()
        critic_loss1 = criterion(q_value1, q_target)

        self.writer.add_scalar('Loss/critic_loss1', critic_loss1.item(),
                               self.total_time_step)

        if self.twin:
            critic_loss2 = criterion(q_value2, q_target)
            self.writer.add_scalar('Loss/critic_loss2', critic_loss2.item(),
                                   self.total_time_step)
        # optimize critic
        self.critic_net1.zero_grad()
        critic_loss1.backward()
        self.critic_opt1.step()

        if self.twin:
            self.critic_net2.zero_grad()
            critic_loss2.backward()
            self.critic_opt2.step()

        ## Delayed Actor(Policy) Updates ##
        if self.total_time_step % self.update_freq == 0:
            ## update actor ##
            # actor loss
            # select action a from behavior actor network (a is different from sample transition's action)
            # get Q from behavior critic network, mean Q value -> objective function
            # maximize (objective function) = minimize -1 * (objective function)
            action = self.actor_net(state)
            q_value = self.critic_net1(state, action)
            actor_loss = -1 * q_value.mean()

            self.writer.add_scalar('Loss/actor_loss', actor_loss.item(),
                                   self.total_time_step)
            # optimize actor
            self.actor_net.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
