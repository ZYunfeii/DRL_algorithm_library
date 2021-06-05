#!/usr/bin/python
# -*- coding: utf-8 -*-
from Agent import AgentD3QN
from core import ReplayBuffer
import gym
from PPO.draw import Painter

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    obs_dim = env.observation_space.shape[0]
    act_dim = 2
    agent = AgentD3QN()
    agent.init(256,obs_dim,act_dim)
    buffer = ReplayBuffer(2**15,obs_dim,1,if_on_policy=False,if_gpu=True)  # 离散情况这个buffer 的actdim一定写1（反直觉）
    MAX_EPISODE = 100
    MAX_STEP = 500
    batch_size = 256
    gamma = 0.99
    update_every = 50
    rewardList = []
    for episode in range(MAX_EPISODE):
        o = env.reset()
        ep_reward = 0
        for j in range(MAX_STEP):
            if episode > 5:
                a = agent.select_action(o)
            else:
                a = env.action_space.sample()
            o2, r, d, _ = env.step(a)
            mask = 0.0 if d else gamma
            buffer.append_buffer(o,(r,mask,a))

            if episode >= 5 and j % update_every == 0:
                agent.update_net(buffer,2**10,batch_size,1)
            o = o2
            ep_reward += r
            if d: break
        print('Episode:', episode, 'Reward:%f' %ep_reward)
        rewardList.append(ep_reward)

    painter = Painter(load_csv=False, load_dir=None)
    painter.addData(rewardList, 'D3QN')
    painter.drawFigure()


