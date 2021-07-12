import numpy as np
from copy import deepcopy
from torch.optim import Adam
import torch
import core as core
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:   # 输入为size；obs的维度(3,)：这里在内部对其解运算成3；action的维度3
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

    def sample_one(self,index):
        batch = dict(obs=self.obs_buf[index],
                     obs2=self.obs2_buf[index],
                     act=self.act_buf[index],
                     rew=self.rew_buf[index],
                     done=self.done_buf[index])
        return batch

class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = ReplayBuffer(obs_dim,act_dim,capacity)
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data.store(*data)
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0               # replay buffer也会同步将指针指向初始位置

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data.sample_one(data_idx)

    @property
    def total_p(self):
        return self.tree[0]  # the root




class DDPG:
    def __init__(self, obs_dim, act_dim, act_bound, actor_critic=core.MLPActorCritic, seed=0,
                capacity=2**19, gamma=0.99, polyak=0.995, pi_lr=1e-3, q_lr=1e-3, act_noise=0.1,
                 abs_err_upper=10,beta=0.4):   # capacity得是2的整数次方

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_bound = act_bound
        self.gamma = gamma
        self.polyak = polyak
        self.act_noise = act_noise
        self.abs_err_upper = abs_err_upper
        self.capacity = capacity
        self.beta = beta

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.ac = actor_critic(obs_dim, act_dim, act_limit = 2.0).to(device)
        self.ac_targ = deepcopy(self.ac).to(device)

        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(self.ac.q.parameters(), lr=q_lr)

        for p in self.ac_targ.parameters():
            p.requires_grad = False

        self.sumTree = SumTree(capacity,obs_dim,act_dim)

    def compute_loss_q(self, data, leaf_idx_list,ISWeights):   #返回(q网络loss, q网络输出的状态动作值即Q值)
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q = self.ac.q(o,a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = self.ac_targ.q(o2, self.ac_targ.pi(o2))
            backup = r + self.gamma * (1 - d) * q_pi_targ

            # update sum tree
            abs_td_error = torch.abs(q-backup)
            for i,leaf_idx in enumerate(leaf_idx_list):
                p = abs_td_error[i] + 0.01             # 避免过于接近0
                p = torch.clip(p,0,self.abs_err_upper)
                p = torch.pow(p,0.8)
                self.sumTree.update(leaf_idx,p.item())

        # MSE loss against Bellman backup
        loss_q = (torch.mul((q-backup)**2,ISWeights)).mean()

        return loss_q # 这里的loss_q没加负号说明是最小化，很好理解，TD正是用函数逼近器去逼近backup，误差自然越小越好

    def compute_loss_pi(self, data):
        o = data['obs']
        q_pi = self.ac.q(o, self.ac.pi(o))
        return -q_pi.mean()  # 这里的负号表明是最大化q_pi,即最大化在当前state策略做出的action的Q值

    def update(self,batch_size):
        total = self.sumTree.total_p
        seg = total / batch_size
        data = {'obs':[],'act':[],'rew':[],'obs2':[],'done':[]}
        leaf_idx_list = list()
        ISWeights = np.empty(batch_size)
        self.beta = np.min((self.beta+0.001,1))       # beta从初始值趋近于1
        for i in range(batch_size):
            begin, end = seg*i, seg*(i+1)
            v = np.random.uniform(begin, end)
            leaf_idx,p,batch = self.sumTree.get_leaf(v)
            leaf_idx_list.append(leaf_idx)
            data['obs'].append(batch['obs'])
            data['act'].append(batch['act'])
            data['rew'].append(batch['rew'])
            data['obs2'].append(batch['obs2'])
            data['done'].append(batch['done'])
            ISWeights[i] = np.power(p/total,-self.beta)
        ISWeights = torch.as_tensor(ISWeights / np.max(ISWeights),dtype=torch.float32,device=device)
        data = {k: torch.as_tensor(v, dtype=torch.float32,device=device) for k,v in data.items()}

        # First run one gradient descent step for Q.
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data,leaf_idx_list,ISWeights)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for p in self.ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.ac.q.parameters():
            p.requires_grad = True


        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o, noise_scale):
        a = self.ac.act(torch.as_tensor(o, dtype=torch.float32,device=device))
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, self.act_bound[0], self.act_bound[1])

    def store(self,tran):
        max_p = np.max(self.sumTree.tree[-self.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.sumTree.add(max_p,tran)