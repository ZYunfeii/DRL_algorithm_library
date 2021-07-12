#!/usr/bin/python
# -*- coding: utf-8 -*-
from td3 import TD3
import gym
import matplotlib.pyplot as plt
import numpy as np
from time import time
import torch
from Arm_env import ArmEnv

def main1():
    """gym环境"""
    env = gym.make('LunarLanderContinuous-v2')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    td3 = TD3(obs_dim, act_dim)

    MAX_EPISODE = 250
    MAX_STEP = 5000
    update_every = 50
    batch_size = 100
    rewardList = []
    begin = time()
    for episode in range(MAX_EPISODE):
        o = env.reset()
        ep_reward = 0
        for j in range(MAX_STEP):
            if episode > 20:
                a = td3.get_action(o, td3.act_noise)
            else:
                a = env.action_space.sample()
            o2, r, d, _ = env.step(a)
            td3.replay_buffer.store(o, a, r, o2, d)

            if episode >= 20 and j % update_every == 0:
                td3.update(batch_size, update_every)

            o = o2
            ep_reward += r

            if d: break
        print('Episode:', episode, 'Reward:%i' % int(ep_reward), 'Time:%i' % (time() - begin))
        rewardList.append(ep_reward)

    plt.figure()
    plt.plot(np.arange(len(rewardList)), rewardList)
    plt.show()

def main2():
    """Arm环境"""
    env = ArmEnv(mode='hard')
    obs_dim = env.state_dim
    act_dim = env.action_dim
    td3 = TD3(obs_dim, act_dim)

    MAX_EPISODE = 100
    MAX_STEP = 500
    update_every = 50
    batch_size = 100
    rewardList = []
    begin = time()
    for episode in range(MAX_EPISODE):
        o = env.reset()
        ep_reward = 0
        for j in range(MAX_STEP):
            if episode > 20:
                a = td3.get_action(o, td3.act_noise)
            else:
                a = env.sample_action()
            o2, r, d,_ = env.step(a)
            td3.replay_buffer.store(o, a, r, o2, d)
            if episode >= 20 and j % update_every == 0:
                td3.update(batch_size, update_every)
                # env.render()

            o = o2
            ep_reward += r

            if d: break
        print('Episode:', episode, 'Reward:%i' % int(ep_reward), 'Time:%i' % (time() - begin))
        rewardList.append(ep_reward)

    torch.save(td3.ac.state_dict(),'../TrainedModel/td3_pi.pkl')

    plt.figure()
    plt.plot(np.arange(len(rewardList)), rewardList)
    plt.show()

if __name__ == '__main__':
    main2()





