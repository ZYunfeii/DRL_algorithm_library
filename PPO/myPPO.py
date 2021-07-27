#!/usr/bin/python
# -*- coding: utf-8 -*-
from PPOModel import *
import gym
import matplotlib.pyplot as plt
from draw import Painter
import random
from time import time

def test(env,model):
    state = env.reset()
    for step_sum in range(500):
        env.render()
        action, log_prob = ppo.select_action((state,))
        next_state, reward, done, _ = env.step(np.tanh(action))
        if done:
            break

        state = next_state
def setup_seed(seed):
    """设置随机数种子函数"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run(seed,env_name):
    setup_seed(seed)
    env = gym.make(env_name)

    state_dim, action_dim, net_dim = env.observation_space.shape[0], env.action_space.shape[0], 256

    ppo = AgentPPO(state_dim, action_dim, net_dim)

    MAX_EPISODE = 30
    batch_size = 128
    rewardList = list()
    timeList = list()
    begin = time()
    for episode in range(MAX_EPISODE):
        # if episode > 90: test(env, ppo)
        rewards, steps = ppo.update_buffer(env, 5000, 1)
        ppo.update_policy(batch_size, 8)
        rewardList.append(rewards)
        timeList.append(time()-begin)
        print('Episode:', episode, 'Reward:%i' % int(rewards), 'time:%i'%timeList[-1])


    painter = Painter(load_csv=True, load_dir='figure1.csv')
    painter.addData(rewardList, 'PPO',x=timeList)
    painter.saveData(save_dir='figure1.csv')
    painter.drawFigure()

if __name__ == "__main__":
    run(2021,'LunarLanderContinuous-v2')
