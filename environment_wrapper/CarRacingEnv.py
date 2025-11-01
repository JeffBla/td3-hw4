import argparse
from collections import deque
import itertools
import random
import time
import cv2

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class RewardShaping:

    def __init__(self, config):
        self.w_onroad = config['w_onroad']
        self.w_antigrass = config['w_antigrass']
        self.w_smooth = config['w_smooth']
        self.w_brake = config['w_brake']
        self.w_throttle = config['w_throttle']
        self.time_penalty = config['time_penalty']
        self.heavy_offroad_penalty = config['heavy_offroad_penalty']
        self.road_min_for_ok = config['road_min_for_ok']
        self.brake_thresh = config['brake_thresh']


class CarRacingEnvironment:

    def __init__(self, N_frame=4, config=None, writer=None, test=False):
        self.record_video = config['record_video']
        self.test = test
        self.writer = writer
        if self.record_video:
            self.env = gym.make('CarRacing-v3', render_mode="human")
        else:
            self.env = gym.make('CarRacing-v3')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.ep_len = 0
        self.frames = deque(maxlen=N_frame)
        self.use_shaping = config[
            'use_shaping'] if config is not None else False
        self.reward_shaping = RewardShaping(config)
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.prev_prev_action = np.zeros_like(self.prev_action)

        self.offroad_cd = 0
        self.was_on_road = True

    def check_car_position(self, obs):
        # cut the image to get the part where the car is
        part_image = obs[60:84, 40:60, :]

        road_color_lower = np.array([90, 90, 90], dtype=np.uint8)
        road_color_upper = np.array([120, 120, 120], dtype=np.uint8)
        grass_color_lower = np.array([90, 180, 90], dtype=np.uint8)
        grass_color_upper = np.array([120, 255, 120], dtype=np.uint8)
        road_mask = cv2.inRange(part_image, road_color_lower, road_color_upper)
        grass_mask = cv2.inRange(part_image, grass_color_lower,
                                 grass_color_upper)
        # count the number of pixels in the road and grass
        road_pixel_count = cv2.countNonZero(road_mask)
        grass_pixel_count = cv2.countNonZero(grass_mask)

        # save image for debugging
        # filename = "images/image" + str(self.ep_len) + ".jpg"
        # cv2.imwrite(filename, part_image)

        return road_pixel_count, grass_pixel_count

    def shape_reward(self, base_reward, action, info):
        if self.test:
            return base_reward

        road_px = info["road_pixel_count"]
        grass_px = info["grass_pixel_count"]
        total_px = road_px + grass_px + 1e-6  # avoid div by zero
        road_pixel_ratio = road_px / total_px
        grass_pixel_ratio = grass_px / total_px

        shaped_reward = base_reward

        # on road reward
        shaped_reward += self.reward_shaping.w_onroad * road_pixel_ratio

        # anti grass reward
        shaped_reward += self.reward_shaping.w_antigrass * (-grass_pixel_ratio)

        # smooth steering
        steer = float(action[0])
        dsteer = abs(steer - float(self.prev_action[0]))
        ddsteer = abs(steer - 2 * float(self.prev_action[0]) +
                      float(self.prev_prev_action[0]))
        shaped_reward += -self.reward_shaping.w_smooth * (dsteer +
                                                          0.5 * ddsteer)

        # brake penalty
        shaped_reward += -self.reward_shaping.w_brake * max(
            0.0, action[2] - self.reward_shaping.brake_thresh)

        # throttle reward
        shaped_reward += self.reward_shaping.w_throttle * action[1]

        # time penalty
        shaped_reward += self.reward_shaping.time_penalty * (-1)

        # heavy offroad penalty
        onroad = road_px >= self.reward_shaping.road_min_for_ok
        if not onroad and self.was_on_road and self.offroad_cd == 0:
            shaped_reward -= self.reward_shaping.heavy_offroad_penalty
            self.offroad_cd = 30
        self.was_on_road = onroad
        if self.offroad_cd > 0:
            self.offroad_cd -= 1

        # update previous actions
        self.prev_prev_action = self.prev_action
        self.prev_action = action

        # clip the shaped reward to avoid extreme values
        shaped_reward = float(np.clip(shaped_reward, -5.0, 5.0))

        return shaped_reward

    def step(self, action):
        obs, reward, terminates, truncates, info = self.env.step(action)
        original_reward = reward
        original_terminates = terminates
        self.ep_len += 1
        road_pixel_count, grass_pixel_count = self.check_car_position(obs)
        info["road_pixel_count"] = road_pixel_count
        info["grass_pixel_count"] = grass_pixel_count

        # TODO
        if self.use_shaping:
            reward = self.shape_reward(reward, action, info)
        else:
            if road_pixel_count < 10:
                terminates = True
                reward = -100

        # convert to grayscale
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)  # 96x96

        # save image for debugging
        # filename = "images/image" + str(self.ep_len) + ".jpg"
        # cv2.imwrite(filename, obs)

        # frame stacking
        self.frames.append(obs)
        obs = np.stack(self.frames, axis=0)

        if self.test:
            # enable this line to recover the original reward
            reward = original_reward
            # enable this line to recover the original terminates signal, disable this to accerlate evaluation
            # terminates = original_terminates

        return obs, reward, terminates, truncates, info

    def reset(self):
        obs, info = self.env.reset()
        self.ep_len = 0
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)  # 96x96

        # frame stacking
        for _ in range(self.frames.maxlen):
            self.frames.append(obs)
        obs = np.stack(self.frames, axis=0)

        return obs, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


if __name__ == '__main__':
    env = CarRacingEnvironment(test=True)
    obs, info = env.reset()
    done = False
    total_reward = 0
    total_length = 0
    t = 0
    while not done:
        t += 1
        action = env.action_space.sample()
        action[2] = 0.0
        obs, reward, terminates, truncates, info = env.step(action)
        print(
            f'{t}: road_pixel_count: {info["road_pixel_count"]}, grass_pixel_count: {info["grass_pixel_count"]}, reward: {reward}'
        )
        total_reward += reward
        total_length += 1
        env.render()
        if terminates or truncates:
            done = True

    print("Total reward: ", total_reward)
    print("Total length: ", total_length)
    env.close()
