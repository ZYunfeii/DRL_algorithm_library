#!/usr/bin/python
# -*- coding: utf-8 -*-
from td3 import TD3, ReplayBuffer
import gym
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Process, Manager, Pipe
from threading import Thread
from copy import deepcopy
import torch
from time import time,sleep
from PPO.draw import Painter

def main():
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    td3 = TD3(obs_dim,act_dim)
    """进程初始化"""
    process_num = 4
    pipe_dict = dict((i, (pipe1, pipe2)) for i in range(process_num) for pipe1, pipe2 in (Pipe(),))
    child_process_list = []
    for i in range(process_num):
        pro = Process(target=child_process, args=(pipe_dict[i][1],env_name))
        child_process_list.append(pro)
    child_pi = deepcopy(td3.ac.pi)
    [pipe_dict[i][0].send((child_pi.to("cpu"),)) for i in range(process_num)]

    """进程启动"""
    [p.start() for p in child_process_list]

    MAX_EPISODE = 150
    batch_size = 100

    begin = time()
    rewardList = list()
    timeList = list()
    """预先收集buffer"""
    expected_size = td3.replay_buffer.max_size/1000
    while td3.replay_buffer.size < expected_size:
        for i in range(process_num):
            if pipe_dict[i][0].poll():
                receive = pipe_dict[i][0].recv()
                td3.replay_buffer.store(*receive)
    print('==收集完成!')

    for episode in range(MAX_EPISODE):
        t = Thread(target=child_thread, args=(pipe_dict, process_num, td3)) # 通过线程收集buffer
        t.start()
        child_pi = deepcopy(td3.ac.pi)
        [pipe_dict[i][0].send((child_pi.to("cpu"),)) for i in range(process_num)]
        td3.update(batch_size,100)
        ep_reward = test_pi(deepcopy(td3.ac.pi),env)
        rewardList.append(ep_reward)
        timeList.append(time()-begin)
        print(f"epsiode:{episode} reward:{ep_reward} time:{timeList[-1]}s")
        t.join()

    [p.terminate() for p in child_process_list]

    painter = Painter(load_csv=False)
    painter.addData(rewardList,'MP-TD3',x=timeList)
    painter.drawFigure()

def child_thread(pipe_dict,process_num,td3):
    count = 0
    while True:
        for i in range(process_num):
            if pipe_dict[i][0].poll():
                receice = pipe_dict[i][0].recv()
                td3.replay_buffer.store(*receice)
                count += 1
        if count>200: break         # 每个episode该线程收集样本个数为count



def child_process(pipe,env_name):
    env = gym.make(env_name)
    MAX_STEP = 500
    pi = pipe.recv()[0]
    act_dim = env.action_space.shape[0]
    while True:
        o = env.reset()
        for j in range(MAX_STEP):
            if pipe.poll():
                pi = pipe.recv()[0]
            a = pi(torch.as_tensor(o, dtype=torch.float32, device="cpu")).detach().numpy()
            a += 0.15 * np.random.randn(act_dim)
            a = np.clip(a, -1, 1)*2
            if np.random.rand() < 0.1:
                a = env.action_space.sample()
            o2, r, d, _ = env.step(a)
            pipe.send((o,a,r,o2,d))
            o = o2
            if d: break

def test_pi(pi,env):
    o = env.reset()
    ep_reward = 0
    pi = pi.to("cpu")
    for j in range(500):
        a = pi(torch.as_tensor(o, dtype=torch.float32,device="cpu")).detach().numpy()*2
        o2, r, d, _ = env.step(a)
        o = o2
        ep_reward += r
        if d: break
    return ep_reward

if __name__ == "__main__":
    main()