# README

This is a reinforcement learning algorithm library. The code takes into account both **performance and simplicity**, with **little dependence**. The algorithm code comes from spinningup and some researchers engaged in reinforcement learning.

## Algorithms

The project covers the following algorithms：

* **DDPG**
* **PPO+GAE**
* **TD3**
* **SAC**
* **MADDPG**

All the algorithms adopt the **pytorch** framework. All the codes are combined in the easiest way to understand, which is suitable for beginners of reinforcement learning, but the code performance is excellent.

## References
This project also provides the reference of these algorithms:

* Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments
* CONTINUOUS CONTROL WITH DEEP REINFORCEMENT
* High-Dimensional Continuous Control Using Generalized Advantage Estimation
* Proximal Policy Optimization
* Soft Actor-Critic Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
* Auto alpha  Soft Actor-Critic Algorithms and Applications
* Addressing Function Approximation Error in Actor-Critic Methods

## Results

Testing environment: **'Pendulum-v0'**. What you just need to do is running the **main.py**. Here are the results of several cases:

spinningup-DDPG reward curve:

![Alt text](./imgs/spin_ddpg.png)

spinningup-TD3 reward curve:

![Alt text](./imgs/spin_td3.png)

spinningup-SAC reward curve:

![Alt text](./imgs/spinSAC.png)

## Project tree

```
.
├─ DDPG
│	├─ DDPG
│	└─ DDPG_spinningup
├─ MADDPG
│	├─ .gitignore
│	├─ .idea
│	├─ __pycache__
│	├─ arguments.py
│	├─ enjoy_split.py
│	├─ logs
│	├─ main_openai.py
│	├─ model.py
│	├─ models
│	└─ replay_buffer.py
├─ PPO
│	├─ .idea
│	├─ PPOModel.py
│	├─ __pycache__
│	├─ core.py
│	└─ myPPO.py
├─ README.md
├─ SAC
│	├─ SAC_demo1
│	└─ SAC_spinningup
├─ TD3
│	├─ TD3
│	└─ TD3_spinningup
├─ imgs
│	├─ spinSAC.png
│	├─ spin_ddpg.png
│	└─ spin_td3.png
└─ reference
 	├─ 多智能体 MADDPG - Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments - 1706.02275.pdf
 	├─ 强化学习 DDPG - CONTINUOUS CONTROL WITH DEEP REINFORCEMENT 1509.02971.pdf
 	├─ 强化学习 GAE High-Dimensional Continuous Control Using Generalized Advantage Estimation 1506.02438.pdf
 	├─ 强化学习 PPO - Proximal Policy Optimization1707.06347.pdf
 	├─ 强化学习 SAC1 - Soft Actor-Critic Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor - 1801.01290.pdf
 	├─ 强化学习 SAC2 auto alpha  Soft Actor-Critic Algorithms and Applications 1812.05905.pdf
 	└─ 强化学习 TD3 - Addressing Function Approximation Error in Actor-Critic Methods 1802.09477.pdf
```

## Requirements

gym==0.10.5

matplotlib==3.2.2

pytorch==1.7.1

numpy==1.19.2



