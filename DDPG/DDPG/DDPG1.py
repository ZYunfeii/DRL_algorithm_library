import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import gym

def soft_target_update(target, current, tau=5e-3):
    for target_param, param in zip(target.parameters(), current.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
class OrnsteinUhlenbeckProcess:
    """ Don't abuse OU Process
    OU process has too much hyper-parameters.
    Over fine-tuning is meaningless.
    """

    def __init__(self, size, theta=0.15, sigma=0.3, x0=0.0, dt=1e-2):
        """
        Source: https://github.com/slowbull/DDPG/blob/master/src/explorationnoise.py
        I think that:
        It makes Zero-mean Gaussian Noise more stable.
        It helps agent explore better in a inertial system.
        """
        self.theta = theta
        self.sigma = sigma
        self.x0 = x0
        self.dt = dt
        self.size = size

    def __call__(self):
        noise = self.sigma * np.sqrt(self.dt) * rd.normal(size=self.size)
        x = self.x0 - self.theta * self.x0 * self.dt + noise
        self.x0 = x  # update x0
        return x
def print_norm(batch_state, neg_avg=None, div_std=None):  # 2020-10-10
    if isinstance(batch_state, torch.Tensor):
        batch_state = batch_state.cpu().data.numpy()
    assert isinstance(batch_state, np.ndarray)

    ary_avg = batch_state.mean(axis=0)
    ary_std = batch_state.std(axis=0)
    fix_std = ((np.max(batch_state, axis=0) - np.min(batch_state, axis=0)) / 6
               + ary_std) / 2

    if neg_avg is not None:  # norm transfer
        ary_avg = ary_avg - neg_avg / div_std
        ary_std = fix_std / div_std

    print(f"| Replay Buffer: avg, fixed std")
    print(f"avg=np.{repr(ary_avg).replace('dtype=float32', 'dtype=np.float32')}")
    print(f"std=np.{repr(ary_std).replace('dtype=float32', 'dtype=np.float32')}")
class BufferArray:  # 2020-11-11
    def __init__(self, max_len, state_dim, action_dim, if_ppo=False):
        state_dim = state_dim if isinstance(state_dim, int) else np.prod(state_dim)  # pixel-level state

        if if_ppo:  # for Offline PPO
            memo_dim = 1 + 1 + state_dim + action_dim + action_dim
        else:
            memo_dim = 1 + 1 + state_dim + action_dim + state_dim

        self.memories = np.empty((max_len, memo_dim), dtype=np.float32)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.next_idx = 0
        self.is_full = False
        self.max_len = max_len
        self.now_len = self.max_len if self.is_full else self.next_idx

        self.state_idx = 1 + 1 + state_dim  # reward_dim==1, done_dim==1
        self.action_idx = self.state_idx + action_dim

    def append_memo(self, memo_tuple):
        # memo_array == (reward, mask, state, action, next_state)
        self.memories[self.next_idx] = np.hstack(memo_tuple)
        self.next_idx = self.next_idx + 1
        if self.next_idx >= self.max_len:
            self.is_full = True
            self.next_idx = 0

    def extend_memo(self, memo_array):
        # assert isinstance(memo_array, np.ndarray)
        size = memo_array.shape[0]
        next_idx = self.next_idx + size
        if next_idx > self.max_len:
            if next_idx > self.max_len:
                self.memories[self.next_idx:self.max_len] = memo_array[:self.max_len - self.next_idx]
            self.is_full = True
            next_idx = next_idx - self.max_len
            self.memories[0:next_idx] = memo_array[-next_idx:]
        else:
            self.memories[self.next_idx:next_idx] = memo_array
        self.next_idx = next_idx

    def random_sample(self, batch_size):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # indices = rd.choice(self.memo_len, batch_size, replace=False)  # why perform worse?
        # indices = rd.choice(self.memo_len, batch_size, replace=True)  # why perform better?
        # same as:
        indices = rd.randint(self.now_len, size=batch_size)
        memory = torch.tensor(self.memories[indices], device=self.device)

        '''convert array into torch.tensor'''
        tensors = (
            memory[:, 0:1],  # rewards
            memory[:, 1:2],  # masks, mark == (1-float(done)) * gamma
            memory[:, 2:self.state_idx],  # states
            memory[:, self.state_idx:self.action_idx],  # actions
            memory[:, self.action_idx:],  # next_states
        )
        return tensors

    def all_sample(self, device):
        tensors = (
            self.memories[:, 0:1],  # rewards
            self.memories[:, 1:2],  # masks, mark == (1-float(done)) * gamma
            self.memories[:, 2:self.state_idx],  # states
            self.memories[:, self.state_idx:self.action_idx],  # actions
            self.memories[:, self.action_idx:],  # next_states
        )
        if device:
            tensors = [torch.tensor(ary, device=device) for ary in tensors]
        return tensors

    def update_pointer_before_sample(self):
        self.now_len = self.max_len if self.is_full else self.next_idx

    def empty_memories_before_explore(self):
        self.next_idx = 0
        self.is_full = False
        self.now_len = 0

    def print_state_norm(self, neg_avg=None, div_std=None):  # non-essential
        memory_state = self.memories[:, 2:self.state_idx]
        print_norm(memory_state, neg_avg, div_std)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),  # nn.BatchNorm1d(mid_dim),
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, action_dim),
        )

    def forward(self, s):
        return self.net(s).tanh()

    def get__noise_action(self, s, a_std):
        a = self.net(s).tanh()
        noise = (torch.randn_like(a) * a_std).clamp(-0.5, 0.5)
        a = (a + noise).clamp(-1.0, 1.0)
        return a

    def get__noise_action_fix(self, s, a_std):
        a = self.net(s).tanh()
        a_temp = torch.normal(a, a_std)
        mask = ((a_temp < -1.0) + (a_temp > 1.0)).type(torch.int8)  # 2019-12-30

        noise_uniform = torch.rand_like(a)  # , device=self.device)
        a = noise_uniform * mask + a_temp * (-mask + 1)
        return a
class Critic(nn.Module):  # 2020-05-05 fix bug
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, 1), )

    def forward(self, s, a):
        x = torch.cat((s, a), dim=1)
        q = self.net(x)
        return q
class AgentDDPG:  # DEMO (tutorial only, simplify, low effective)
    def __init__(self, state_dim, action_dim, net_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.act = Actor(state_dim, action_dim, net_dim).to(self.device)
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=2e-4)

        self.act_target = Actor(state_dim, action_dim, net_dim).to(self.device)
        self.act_target.load_state_dict(self.act.state_dict())

        self.cri = Critic(state_dim, action_dim, net_dim).to(self.device)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=2e-4)

        self.cri_target = Critic(state_dim, action_dim, net_dim).to(self.device)
        self.cri_target.load_state_dict(self.cri.state_dict())

        self.criterion = nn.MSELoss()

        '''training record'''
        self.step = 0

        '''extension'''
        self.ou_noise = OrnsteinUhlenbeckProcess(size=action_dim, sigma=0.3)
        # I hate OU-Process in RL because of its too much hyper-parameters.

    def select_action(self, state):
        states = torch.tensor((state,), dtype=torch.float32, device=self.device)
        action = self.act(states).cpu().data.numpy()[0]
        return (action + self.ou_noise()).clip(-1, 1)

    def update_policy(self, buffer, max_step, batch_size, repeat_times):
        buffer.update_pointer_before_sample()
        critic_loss = actor_loss = None  # just for print return
        for _ in range(int(max_step * repeat_times)):
            with torch.no_grad():
                reward, mask, state, action, next_state = buffer.random_sample(batch_size)

                next_action = self.act_target(next_state)
                next_q_label = self.cri_target(next_state, next_action)
                q_label = reward + mask * next_q_label

            """critic loss (Supervised Deep learning)
            minimize criterion(q_eval, label) to train a critic
            We input state-action to a critic (policy function), critic will output a q_value estimation.
            A better action will get higher q_value from critic.  
            """
            q_value = self.cri(state, action)
            critic_loss = self.criterion(q_value, q_label)

            self.cri_optimizer.zero_grad()
            critic_loss.backward()
            self.cri_optimizer.step()

            """actor loss (Policy Gradient)
            maximize cri(state, action) is equal to minimize -cri(state, action)
            Accurately, it is more appropriate to call 'actor_loss' as 'actor_objective'.

            We train critic output q_value close to q_label
                by minimizing the error provided by loss function of critic.
            We train actor output action which gets higher q_value from critic
                by maximizing the q_value provided by policy function.
            We call it Policy Gradient (PG). The gradient for actor is provided by a policy function.
                By the way, Generative Adversarial Networks (GANs) is a kind of Policy Gradient.
                The gradient for Generator (Actor) is provided by a Discriminator (Critic).
            """
            action_pg = self.act(state)
            actor_loss = -self.cri(state, action_pg).mean()

            self.act_optimizer.zero_grad()
            actor_loss.backward()
            self.act_optimizer.step()

            """soft target update can obviously stabilize training"""
            soft_target_update(self.act_target, self.act)
            soft_target_update(self.cri_target, self.cri)

        return actor_loss.item(), critic_loss.item()

if __name__ == "__main__":
    env = gym.make('Pendulum-v0')

    state_dim, action_dim, net_dim = env.observation_space.shape[0],env.action_space.shape[0],256
    ddpg = AgentDDPG(state_dim, action_dim, net_dim)
    max_buffer = 10000
    gamma = 0.99
    buffer = BufferArray(max_buffer,state_dim,action_dim)
    MAX_EPISODE = 200
    MAX_STEP = 500
    batch_size = 128
    rewardList = []


    for episode in range(MAX_EPISODE):
        observation = env.reset()
        ep_reward = 0
        for j in range(MAX_STEP):
            # if episode>180:
            #     env.render()
            env.render()
            action = ddpg.select_action(observation) * 2
            observation_,reward,done,_ = env.step(action)
            mask = 0.0 if done else gamma
            buffer.append_memo((reward,mask,observation,action,observation_))

            if buffer.is_full:
                ddpg.update_policy(buffer, 1, batch_size, 1)  # 经验池满了的情况下进行网络参数更新


            observation = observation_
            ep_reward += reward
            if done:
                break
        print('Episode:', episode, 'Reward:%i' % int(ep_reward))
        rewardList.append(ep_reward)

    plt.figure()
    plt.plot(np.arange(len(rewardList)),rewardList)

