#!/usr/bin/python
# -*- coding: utf-8 -*-
from PPOModel import *
import gym
import multiprocessing
from PPO.draw import Painter
from copy import deepcopy
from time import time
import random
from Arm_env import ArmEnv

def setup_seed(seed):
    """设置随机数种子函数"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main1():
    """
    第一种实现多进程加速的思路：
        1.多个子进程同步训练网络
        2.在主进程将子进程网络的权重取平均更新到net
        3.再将net传入子进程，回到1
    """
    env = gym.make('LunarLanderContinuous-v2')
    net = GlobalNet(env.observation_space.shape[0],env.action_space.shape[0])

    process_num = 4
    pipe_dict = dict((i,(pipe1,pipe2)) for i in range(process_num) for pipe1,pipe2 in (multiprocessing.Pipe(),))
    child_process_list = []
    for i in range(process_num):
        pro = multiprocessing.Process(target=child_process1, args=(pipe_dict[i][1],))
        child_process_list.append(pro)
    [p.start() for p in child_process_list]

    rewardList = list()
    MAX_EPISODE = 30
    for episode in range(MAX_EPISODE):
        [pipe_dict[i][0].send(net) for i in range(process_num)]
        net_list = list()
        for i in range(process_num):
            net_list.append(pipe_dict[i][0].recv())
        """将子进程网络权重的平均值赋给net，下一个episode主进程再将net传给子进程"""
        act_model_dict = net.act.state_dict()
        cri_model_dict = net.cri.state_dict()
        for k1,k2 in zip(act_model_dict.keys(),cri_model_dict.keys()):
            result1 = torch.zeros_like(act_model_dict[k1])
            result2 = torch.zeros_like(cri_model_dict[k2])
            for j in range(process_num):
                result1 += net_list[j][0].state_dict()[k1]
                result2 += net_list[j][1].state_dict()[k2]
            result1 /= process_num
            result2 /= process_num
            act_model_dict[k1] = result1
            cri_model_dict[k2] = result2
        net.act.load_state_dict(act_model_dict)
        net.cri.load_state_dict(cri_model_dict)

        reward = 0
        for i in range(process_num):
            reward += net_list[i][2]
        reward /= process_num
        rewardList.append(reward)
        print(f'episode:{episode}  reward:{reward}')

    [p.terminate() for p in child_process_list]

    painter = Painter(load_csv=True,load_dir='../figure.csv')
    painter.addData(rewardList,'MP-PPO')
    painter.saveData('../figure.csv')
    painter.drawFigure()

def child_process1(pipe):
    setup_seed(2021)
    env = gym.make('LunarLanderContinuous-v2')
    batch_size = 128
    while True:
        net = pipe.recv()
        ppo = AgentPPO(net)
        rewards, steps = ppo.update_buffer(env, 5000, 1)
        ppo.update_policy(batch_size, 8)
        pipe.send((ppo.act.to("cpu"),ppo.cri.to("cpu"),rewards))

def main2():
    """
    第二种实现多进程训练的思路：
        1.多个进程不训练网络，只是拿到主进程的网络后去探索环境，并将transition通过pipe传回主进程
        2.主进程将所有子进程的transition打包为一个buffer后供网络训练
        3.将更新后的net再传到子进程，回到1
    """
    """gym环境"""
    # env = gym.make('LunarLanderContinuous-v2')
    # net = GlobalNet(env.observation_space.shape[0], env.action_space.shape[0])
    """End"""

    """机械臂环境"""
    env = ArmEnv(mode='hard')
    obs_dim = env.state_dim
    act_dim = env.action_dim
    net = GlobalNet(obs_dim, act_dim)
    """End"""

    ppo = AgentPPO(deepcopy(net))
    process_num = 6
    pipe_dict = dict((i, (pipe1, pipe2)) for i in range(process_num) for pipe1, pipe2 in (multiprocessing.Pipe(),))
    child_process_list = []
    for i in range(process_num):
        pro = multiprocessing.Process(target=child_process2, args=(pipe_dict[i][1],))
        child_process_list.append(pro)
    [pipe_dict[i][0].send(net) for i in range(process_num)]
    [p.start() for p in child_process_list]

    rewardList = list()
    timeList = list()
    begin = time()
    MAX_EPISODE = 150
    batch_size = 128
    max_reward = -np.inf
    for episode in range(MAX_EPISODE):
        reward = 0
        buffer_list = list()
        for i in range(process_num):
            receive = pipe_dict[i][0].recv()        # 这句带同步子进程的功能，收不到子进程的数据就都不会走到for之后的语句
            data = receive[0]
            buffer_list.append(data)
            reward += receive[1]
        ppo.update_policy_mp(batch_size,8,buffer_list)
        net.act.load_state_dict(ppo.act.state_dict())
        net.cri.load_state_dict(ppo.cri.state_dict())
        [pipe_dict[i][0].send(net) for i in range(process_num)]

        reward /= process_num
        rewardList.append(reward)
        timeList.append(time() - begin)
        print(f'episode:{episode}  reward:{reward} time:{timeList[-1]}')

        if reward > max_reward and episode > MAX_EPISODE*2/3:
            max_reward = reward
            torch.save(net.act.state_dict(), '../TrainedModel/act.pkl')

    [p.terminate() for p in child_process_list]

    painter = Painter(load_csv=False)
    painter.addData(rewardList, 'MP-PPO')
    painter.drawFigure()


def child_process2(pipe):
    setup_seed(0)
    # env = gym.make('LunarLanderContinuous-v2')
    env = ArmEnv(mode='hard')

    env.reset()
    while True:
        net = pipe.recv()  # 收主线程的net参数，这句也有同步的功能
        ppo = AgentPPO(net,if_explore=True)
        rewards, steps = ppo.update_buffer(env, 500, 1)
        transition = ppo.buffer.sample_all()
        r = transition.reward
        m = transition.mask
        a = transition.action
        s = transition.state
        log = transition.log_prob
        data = (r,m,s,a,log)
        """pipe不能直接传输buffer回主进程，可能是buffer内有transition，因此将数据取出来打包回传"""
        pipe.send((data,rewards))



if __name__ == "__main__":
    main2()


