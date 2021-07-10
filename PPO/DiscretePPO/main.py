#!/usr/bin/python
# -*- coding: utf-8 -*-

#!/usr/bin/python
# -*- coding: utf-8 -*-
from Agent import AgentDiscretePPO
from core import ReplayBuffer
import gym
from PPO.draw import Painter
import torch
from copy import deepcopy

def testAgent(test_env,agent,episode):
    ep_reward = 0
    o = test_env.reset()
    for j in range(500):
        if episode > 10:
            test_env.render()
        a_int, a_prob = agent.select_action(o)
        o2, reward, done, _ = test_env.step(a_int)
        ep_reward += reward
        if done:break
        o = o2
    return ep_reward

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    test_env = deepcopy(env)
    obs_dim = env.observation_space.shape[0]
    act_dim = 2
    agent = AgentDiscretePPO()
    agent.init(256,obs_dim,act_dim)
    buffer = ReplayBuffer(2**15,obs_dim,act_dim,True)
    MAX_EPISODE = 100
    MAX_STEP = 500
    batch_size = 256
    update_every = 50
    rewardList = []
    agent.state = env.reset()
    for episode in range(MAX_EPISODE):
        with torch.no_grad():
            trajectory_list = agent.explore_env(env,1024,1,0.99)
        buffer.extend_buffer_from_list(trajectory_list)
        agent.update_net(buffer,batch_size,8,2**-8)
        ep_reward = testAgent(test_env,agent,episode)
        print('Episode:', episode, 'Reward:%f' %ep_reward)
        rewardList.append(ep_reward)

    painter = Painter(load_csv=False, load_dir=None)
    painter.addData(rewardList, 'D3QN')
    painter.drawFigure()


