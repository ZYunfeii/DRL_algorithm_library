#!/usr/bin/python
# -*- coding: utf-8 -*-
import gym
from TD3Model import AgentTD3, ReplayBuffer
import matplotlib.pyplot as plt
import numpy as np
import time
from PPO.draw import Painter
if __name__ == "__main__":
    env = gym.make('Pendulum-v0')
    state_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
    td3 = AgentTD3()
    td3.init(256,state_dim,action_dim)
    buffer = ReplayBuffer(int(1e6), state_dim, action_dim, False, True)
    MAX_EPISODE = 100
    MAX_STEP = 500
    batch_size = 100
    gamma = 0.99
    reward_list = []
    time_list = []
    begin = time.time()
    for episode in range(MAX_EPISODE):
        s = env.reset()
        ep_reward = 0
        for j in range(MAX_STEP):
            # if episode > 130: env.render()
            if episode > 20:
                a = td3.select_action(s)*2
            else:
                a = env.action_space.sample()
            s_, r, d, _ = env.step(a)
            mask = 0.0 if d else gamma
            other = (r,mask,*a)
            buffer.append_buffer(s, other)
            if episode > 20 and j % 50 == 0:
                td3.update_net(buffer,50,batch_size,1)
            ep_reward += r
            s = s_
            if d: break
        reward_list.append(ep_reward)
        time_list.append(time.time()-begin)
        print('Episode:', episode, 'Reward:%f' % ep_reward, 'time:%f'%(time_list[-1]))
    # plt.figure()
    # plt.plot(np.arange(len(reward_list)), reward_list)
    # plt.show()

    painter = Painter(load_csv=False)
    painter.addData(reward_list,'TD3',x=time_list)
    painter.drawFigure()

