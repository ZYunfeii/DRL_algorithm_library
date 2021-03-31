#!/usr/bin/python
# -*- coding: utf-8 -*-
from PPOModel import AgentPPO
import gym
import matplotlib.pyplot as plt


if __name__ == "__main__":
    env = gym.make('Pendulum-v0')

    state_dim, action_dim, net_dim = env.observation_space.shape[0],env.action_space.shape[0],256
    ppo = AgentPPO(state_dim, action_dim, net_dim)
    gamma = 0.99
    MAX_EPISODE = 100
    MAX_STEP = 500
    batch_size = 128
    rewardList = []


    for episode in range(MAX_EPISODE):
        rewards, steps = ppo.update_buffer(env, 5000, 1, gamma)
        ppo.update_policy(batch_size, 16)
        print('Episode:', episode, 'Reward:%i' % int(rewards[0]))
        rewardList.append(rewards[0])

    plt.figure()
    plt.plot(np.arange(len(rewardList)),rewardList)
    plt.grid()
    plt.show()