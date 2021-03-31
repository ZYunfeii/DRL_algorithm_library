import numpy as np
import numpy.random as rd
import torch
import torch.nn as nn
import gym
import matplotlib.pyplot as plt

def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
def soft_target_update(target, current, tau=5e-3):
    for target_param, param in zip(target.parameters(), current.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
class ActorSAC(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim, use_dn):
        super().__init__()

        if use_dn:  # use DenseNet (DenseNet has both shallow and deep linear layer)
            nn_dense_net = DenseNet(mid_dim)
            self.net__mid = nn.Sequential(
                nn.Linear(state_dim, mid_dim), nn.ReLU(),
                nn_dense_net,
            )
            lay_dim = nn_dense_net.out_dim
        else:  # use a simple network for actor. Deeper network does not mean better performance in RL.
            self.net__mid = nn.Sequential(
                nn.Linear(state_dim, mid_dim), nn.ReLU(),
                nn.Linear(mid_dim, mid_dim),
            )
            lay_dim = mid_dim

        self.net__mean = nn.Linear(lay_dim, action_dim)             #均值
        self.net__std_log = nn.Linear(lay_dim, action_dim)          #log标准差

        layer_norm(self.net__mean, std=0.01)  # net[-1] is output layer for action, it is no necessary.

        self.log_std_min = -20
        self.log_std_max = 2
        self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))         #常数：根号2pi
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, state):  # in fact, noise_std is a boolean
        x = self.net__mid(state)
        a_mean = self.net__mean(x)  # NOTICE! it is a_mean without .tanh()
        return a_mean.tanh()

    def get__noise_action(self, s):
        x = self.net__mid(s)
        a_mean = self.net__mean(x)  # NOTICE! it is a_mean without .tanh()

        a_std_log = self.net__std_log(x).clamp(self.log_std_min, self.log_std_max)
        a_std = a_std_log.exp()
        a_mean = torch.normal(a_mean, a_std)  # NOTICE! it needs .tanh()
        return a_mean.tanh()

    def get__a__log_prob(self, state):
        x = self.net__mid(state)
        a_mean = self.net__mean(x)  # NOTICE! it needs a_mean.tanh()
        a_std_log = self.net__std_log(x).clamp(self.log_std_min, self.log_std_max)
        a_std = a_std_log.exp()

        """add noise to action in stochastic policy"""
        a_noise = a_mean + a_std * torch.randn_like(a_mean, requires_grad=True, device=self.device) #Returns a tensor with the same size as input that is filled with random numbers from a normal distribution with mean 0 and variance 1
        # Can only use above code instead of below, because the tensor need gradients here.
        # a_noise = torch.normal(a_mean, a_std, requires_grad=True)

        '''compute log_prob according to mean and std of action (stochastic policy)'''
        a_delta = ((a_noise - a_mean) / a_std).pow(2) * 0.5
        # self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        log_prob_noise = a_delta + a_std_log + self.constant_log_sqrt_2pi

        # same as below:
        # from torch.distributions.normal import Normal
        # log_prob_noise = Normal(a_mean, a_std).log_prob(a_noise)
        # same as below:
        # a_delta = a_noise - a_mean).pow(2) /(2* a_std.pow(2)
        # log_prob_noise = -a_delta - a_std.log() - np.log(np.sqrt(2 * np.pi))

        a_noise_tanh = a_noise.tanh()
        log_prob = log_prob_noise + (-a_noise_tanh.pow(2) + 1.000001).log()

        # same as below:
        # epsilon = 1e-6
        # log_prob = log_prob_noise - (1 - a_noise_tanh.pow(2) + epsilon).log()
        return a_noise_tanh, log_prob.sum(1, keepdim=True)
class CriticTwin(nn.Module):  # TwinSAC <- TD3(TwinDDD) <- DoubleDQN <- Double Q-learning
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()

        def build_critic_network():
            net = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                nn.Linear(mid_dim, 1), )
            layer_norm(net[-1], std=0.01)  # It is no necessary.
            return net

        self.net1 = build_critic_network()
        self.net2 = build_critic_network()

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        q = torch.min(self.net1(x), self.net2(x))
        return q

    def get__q1_q2(self, state, action):
        x = torch.cat((state, action), dim=1)
        q_value1 = self.net1(x)
        q_value2 = self.net2(x)
        return q_value1, q_value2
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
class AgentSAC():
    def __init__(self, state_dim, action_dim, net_dim):
        self.learning_rate = 3e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.act = ActorSAC(state_dim, action_dim, net_dim, use_dn=False).to(self.device)
        self.act.train() #启用batch normalization和drop out
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)
        # SAC uses target update network for critic only. Not for actor

        self.cri = CriticTwin(state_dim, action_dim, net_dim).to(self.device)
        self.cri.train()
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)

        self.cri_target = CriticTwin(state_dim, action_dim, net_dim).to(self.device)
        self.cri_target.eval()#不启用 BatchNormalization 和 Dropout
        self.cri_target.load_state_dict(self.cri.state_dict())

        self.criterion = nn.MSELoss()

        '''training record'''
        self.state = None  # env.reset()
        self.reward_sum = 0.0
        self.step = 0

        '''extension: auto-alpha for maximum entropy'''
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = torch.optim.Adam((self.log_alpha,), lr=self.learning_rate)
        self.target_entropy = np.log(action_dim)

    def select_action(self, state):
        states = torch.tensor((state,), dtype=torch.float32, device=self.device)
        actions = self.act.get__noise_action(states)
        return actions.cpu().data.numpy()[0]

    def update_policy(self, buffer, max_step, batch_size, repeat_times):
        """Contribution of SAC (Soft Actor-Critic with maximum entropy)
        1. maximum entropy (Soft Q-learning -> Soft Actor-Critic, good idea)
        2. auto alpha (automating entropy adjustment on temperature parameter alpha for maximum entropy)
        3. SAC use TD3's TwinCritics too
        """
        buffer.update_pointer_before_sample()

        log_prob = critic_loss = None  # just for print return

        for i in range(int(max_step * repeat_times)):
            with torch.no_grad():
                reward, mask, state, action, next_s = buffer.random_sample(batch_size)

                next_a_noise, next_log_prob = self.act.get__a__log_prob(next_s)
                next_q_label = self.cri_target(next_s, next_a_noise)
                q_label = reward + mask * (next_q_label + next_log_prob * self.alpha)

            '''critic_loss'''
            q1_value, q2_value = self.cri.get__q1_q2(state, action)  # CriticTwin
            critic_loss = self.criterion(q1_value, q_label) + self.criterion(q2_value, q_label)

            self.cri_optimizer.zero_grad()
            critic_loss.backward()
            self.cri_optimizer.step()

            soft_target_update(self.cri_target, self.cri)  #网络参数之间soft更新

            '''actor_loss'''
            action_pg, log_prob = self.act.get__a__log_prob(state)  # policy gradient
            # auto alpha
            alpha_loss = (self.log_alpha * (log_prob - self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            # policy gradient
            self.alpha = self.log_alpha.exp()
            q_eval_pg = self.cri(state, action_pg)  # policy gradient
            actor_loss = -(q_eval_pg + log_prob * self.alpha).mean()  # policy gradient

            self.Q = q_eval_pg.mean().item()

            self.act_optimizer.zero_grad()
            actor_loss.backward()
            self.act_optimizer.step()

        return log_prob.mean().item(), critic_loss

if __name__ == "__main__":
    env = gym.make('Pendulum-v0')

    state_dim, action_dim, net_dim = env.observation_space.shape[0],env.action_space.shape[0],256
    sac = AgentSAC(state_dim, action_dim, net_dim)
    max_buffer = 10000
    gamma = 0.99
    buffer = BufferArray(max_buffer,state_dim,action_dim)
    MAX_EPISODE = 200
    MAX_STEP = 500
    batch_size = 128
    rewardList = []
    logProb = []
    criticLoss = []
    QList = []

    for episode in range(MAX_EPISODE):
        observation = env.reset()
        ep_reward = 0
        for j in range(MAX_STEP):
            # if episode>180:
            #     env.render()
            # env.render()
            action = sac.select_action(observation) * 2
            observation_,reward,done,_ = env.step(action)
            mask = 0.0 if done else gamma
            buffer.append_memo((reward,mask,observation,action,observation_))

            if buffer.is_full:
                logprob, criticloss = sac.update_policy(buffer, 1, batch_size, 1)  # 经验池满了的情况下进行网络参数更新
                logProb.append(logprob)
                criticLoss.append(criticloss)
                QList.append(sac.Q)

            observation = observation_
            ep_reward += reward
            if done:
                break
        print('Episode:', episode, 'Reward:%i' % int(ep_reward))
        rewardList.append(ep_reward)

    plt.figure()
    plt.plot(np.arange(len(rewardList)),rewardList)
    plt.figure()
    plt.plot(np.arange(len(logProb)),logProb)
    plt.figure()
    plt.plot(np.arange(len(criticLoss)), criticLoss)
    plt.figure()
    plt.plot(np.arange(len(QList)), QList)
    plt.show()

