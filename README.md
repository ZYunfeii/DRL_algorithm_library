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

## Results

Testing environment: **'Pendulum-v0'**. What you just need to do is running the **main.py**. Here are the results of several cases:

spinningup-DDPG reward curve:

<img src="F:\MasterDegree\ReinforcementLearning\DRL-algorithm-library-master\imgs\spin_ddpg.png" alt="spin_ddpg" style="zoom:33%;" />

spinningup-TD3 reward curve:

<img src="F:\MasterDegree\ReinforcementLearning\DRL-algorithm-library-master\imgs\spin_td3.png" alt="spin_td3" style="zoom:33%;" />

spinningup-SAC reward curve:

<img src="F:\MasterDegree\ReinforcementLearning\DRL-algorithm-library-master\imgs\spinSAC.png" alt="spinSAC" style="zoom:33%;" />

## Project tree

```
.
├─ DDPG
│    ├─ DDPG
│    └─ DDPG_spinningup
├─ MADDPG
│    ├─ .gitignore
│    ├─ .idea
│    ├─ README.md
│    ├─ __pycache__
│    ├─ arguments.py
│    ├─ enjoy_split.py
│    ├─ logs
│    ├─ main_openai.py
│    ├─ model.py
│    ├─ models
│    └─ replay_buffer.py
├─ PPO
│    ├─ .idea
│    ├─ PPOModel.py
│    ├─ __pycache__
│    ├─ core.py
│    └─ myPPO.py
├─ README.md
├─ SAC
│    ├─ SAC_demo1
│    └─ SAC_spinningup
├─ TD3
│    ├─ TD3
│    └─ TD3_spinningup
└─ imgs
       ├─ spinSAC.png
       ├─ spin_ddpg.png
       └─ spin_td3.png
```

## Requirements

gym==0.10.5

matplotlib==3.2.2

pytorch==1.7.1

numpy==1.19.2



