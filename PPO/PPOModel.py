#!/usr/bin/python
# -*- coding: utf-8 -*-

from core import *

class AgentPPO:
    def __init__(self, state_dim, action_dim, net_dim):
        max_buffer = 2**13
        self.gamma = 0.99
        self.buffer = BufferTupleOnline(max_buffer)
        self.learning_rate = 1e-4  # learning rate of actor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.act = ActorPPO(state_dim, action_dim, net_dim).to(self.device)
        self.act.train()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate, )  # betas=(0.5, 0.99))
        # self.act_optimizer = torch.optim.SGD(self.act.parameters(), momentum=0.9, lr=self.learning_rate, )

        self.cri = CriticAdv(state_dim, net_dim).to(self.device)
        self.cri.train()
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate, )  # betas=(0.5, 0.99))
        # self.cri_optimizer = torch.optim.SGD(self.cri.parameters(), momentum=0.9, lr=self.learning_rate, )

        self.criterion = nn.SmoothL1Loss() # 一种损失函数

    def select_action(self, states):  # CPU array to GPU tensor to CPU array
        states = torch.tensor(states, dtype=torch.float32, device=self.device)

        a_noise, log_prob = self.act.get__a__log_prob(states)
        a_noise = a_noise.cpu().data.numpy()[0]
        log_prob = log_prob.cpu().data.numpy()[0]
        return a_noise, log_prob  # not tanh()

    def update_buffer(self, env, max_step, reward_scale):
        # collect tuple (reward, mask, state, action, log_prob, )
        self.buffer.storage_list = list()  # PPO is an online policy RL algorithm.
        # PPO (or GAE) should be an online policy.
        # Don't use Offline for PPO (or GAE). It won't speed up training but slower

        rewards = list()
        steps = list()

        step_counter = 0
        while step_counter < self.buffer.max_memo:
            state = env.reset()

            reward_sum = 0
            step_sum = 0

            for step_sum in range(max_step):
                # env.render()
                action, log_prob = self.select_action((state,))

                next_state, reward, done, _ = env.step(np.tanh(action))
                reward_sum += reward

                mask = 0.0 if done else self.gamma

                reward_ = reward * reward_scale
                self.buffer.push(reward_, mask, state, action, log_prob, )

                if done:
                    break

                state = next_state

            rewards.append(reward_sum)
            steps.append(step_sum)

            step_counter += step_sum
        return np.array(rewards).mean(), steps

    def update_policy(self, batch_size, repeat_times):
        self.act.train()
        self.cri.train()
        clip = 0.25  # ratio.clamp(1 - clip, 1 + clip)
        lambda_adv = 0.98  # why 0.98? cannot use 0.99
        lambda_entropy = 0.01  # could be 0.02
        # repeat_times = 8 could be 2**3 ~ 2**5

        actor_loss = critic_loss = None  # just for print return

        '''the batch for training'''
        max_memo = len(self.buffer)
        all_batch = self.buffer.sample_all()
        all_reward, all_mask, all_state, all_action, all_log_prob = [
            torch.tensor(ary, dtype=torch.float32, device=self.device)
            for ary in (all_batch.reward, all_batch.mask, all_batch.state, all_batch.action, all_batch.log_prob,)
        ]

        # all__new_v = self.cri(all_state).detach_()  # all new value
        with torch.no_grad():
            b_size = 512
            all__new_v = torch.cat(
                [self.cri(all_state[i:i + b_size])
                 for i in range(0, all_state.size()[0], b_size)], dim=0) # 这句相当于把[tensor1, tensor2...] cat 成了一个长tensor

        '''compute old_v (old policy value), adv_v (advantage value) 
        refer: GAE. ICLR 2016. Generalization Advantage Estimate. 
        https://arxiv.org/pdf/1506.02438.pdf'''
        all__delta = torch.empty(max_memo, dtype=torch.float32, device=self.device)
        all__old_v = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # old policy value
        all__adv_v = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # advantage value

        prev_old_v = 0  # old q value
        prev_new_v = 0  # new q value
        prev_adv_v = 0  # advantage q value
        for i in range(max_memo - 1, -1, -1):
            all__delta[i] = all_reward[i] + all_mask[i] * prev_new_v - all__new_v[i]
            all__old_v[i] = all_reward[i] + all_mask[i] * prev_old_v
            all__adv_v[i] = all__delta[i] + all_mask[i] * prev_adv_v * lambda_adv

            prev_old_v = all__old_v[i]
            prev_new_v = all__new_v[i]
            prev_adv_v = all__adv_v[i]

        all__adv_v = (all__adv_v - all__adv_v.mean()) / (all__adv_v.std() + 1e-5)  # advantage_norm:

        '''mini batch sample'''
        sample_times = int(repeat_times * max_memo / batch_size)
        for _ in range(sample_times):
            '''random sample'''
            # indices = rd.choice(max_memo, batch_size, replace=True)  # False)
            indices = rd.randint(max_memo, size=batch_size)

            state = all_state[indices]
            action = all_action[indices]
            advantage = all__adv_v[indices]
            old_value = all__old_v[indices].unsqueeze(1)
            old_log_prob = all_log_prob[indices]

            """Adaptive KL Penalty Coefficient
            loss_KLPEN = surrogate_obj + value_obj * lambda_value + entropy_obj * lambda_entropy
            loss_KLPEN = (value_obj * lambda_value) + (surrogate_obj + entropy_obj * lambda_entropy)
            loss_KLPEN = (critic_loss) + (actor_loss)
            """

            '''critic_loss'''
            new_log_prob = self.act.compute__log_prob(state, action)  # it is actor_loss
            new_value = self.cri(state)

            critic_loss = (self.criterion(new_value, old_value)) / (old_value.std() + 1e-5)
            self.cri_optimizer.zero_grad()
            critic_loss.backward()
            self.cri_optimizer.step()

            '''actor_loss'''
            # surrogate objective of TRPO
            ratio = torch.exp(new_log_prob - old_log_prob)
            surrogate_obj0 = advantage * ratio
            surrogate_obj1 = advantage * ratio.clamp(1 - clip, 1 + clip)
            surrogate_obj = -torch.min(surrogate_obj0, surrogate_obj1).mean()
            loss_entropy = (torch.exp(new_log_prob) * new_log_prob).mean()  # policy entropy

            actor_loss = surrogate_obj + loss_entropy * lambda_entropy
            self.act_optimizer.zero_grad()
            actor_loss.backward()
            self.act_optimizer.step()

        self.act.eval()
        self.cri.eval()
        return actor_loss.item(), critic_loss.item()

class AgentPPO_Mod:
    def __init__(self, state_dim, action_dim):
        max_buffer = 2**13
        self.buffer = BufferTupleOnline(max_buffer)
        self.net_dim = 256
        self.gamma = 0.99
        self.learning_rate = 1e-4  # learning rate of actor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.act = ActorPPO(state_dim, action_dim, self.net_dim).to(self.device)
        self.act.train()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate, )  # betas=(0.5, 0.99))
        # self.act_optimizer = torch.optim.SGD(self.act.parameters(), momentum=0.9, lr=self.learning_rate, )

        self.cri1 = CriticAdv(state_dim, self.net_dim).to(self.device)
        self.cri2 = CriticAdv(state_dim, self.net_dim).to(self.device)
        self.cri1.train()
        self.cri2.train()
        self.cri1_optimizer = torch.optim.Adam(self.cri1.parameters(), lr=self.learning_rate, )  # betas=(0.5, 0.99))
        self.cri2_optimizer = torch.optim.Adam(self.cri2.parameters(), lr=self.learning_rate, )

        self.criterion = nn.SmoothL1Loss() # 一种损失函数


    def select_action(self, states):  # CPU array to GPU tensor to CPU array
        states = torch.tensor(states, dtype=torch.float32, device=self.device)

        a_noise, log_prob = self.act.get__a__log_prob(states)
        a_noise = a_noise.cpu().data.numpy()[0]
        log_prob = log_prob.cpu().data.numpy()[0]
        return a_noise, log_prob  # not tanh()

    def update_buffer(self, env, max_step, reward_scale):
        # collect tuple (reward, mask, state, action, log_prob, )
        self.buffer.storage_list = list()  # PPO is an online policy RL algorithm.
        # PPO (or GAE) should be an online policy.
        # Don't use Offline for PPO (or GAE). It won't speed up training but slower

        rewards = list()
        steps = list()

        step_counter = 0
        while step_counter < self.buffer.max_memo:
            state = env.reset()

            reward_sum = 0
            step_sum = 0

            for step_sum in range(max_step):
                # env.render()
                action, log_prob = self.select_action((state,))

                next_state, reward, done, _ = env.step(np.tanh(action))
                reward_sum += reward

                mask = 0.0 if done else self.gamma

                reward_ = reward * reward_scale
                self.buffer.push(reward_, mask, state, action, log_prob, )

                if done:
                    break

                state = next_state

            rewards.append(reward_sum)
            steps.append(step_sum)

            step_counter += step_sum
        return rewards, steps

    def update_policy(self, batch_size, repeat_times):
        self.act.train()
        self.cri1.train()
        self.cri2.train()
        clip = 0.25  # ratio.clamp(1 - clip, 1 + clip)
        lambda_adv = 0.98  # why 0.98? cannot use 0.99
        lambda_entropy = 0.01  # could be 0.02
        # repeat_times = 8 could be 2**3 ~ 2**5

        actor_loss = critic_loss1 = critic_loss2 = None  # just for print return

        '''the batch for training'''
        max_memo = len(self.buffer)
        all_batch = self.buffer.sample_all()
        all_reward, all_mask, all_state, all_action, all_log_prob = [
            torch.tensor(ary, dtype=torch.float32, device=self.device)
            for ary in (all_batch.reward, all_batch.mask, all_batch.state, all_batch.action, all_batch.log_prob,)
        ]

        # all__new_v = self.cri(all_state).detach_()  # all new value
        with torch.no_grad():
            b_size = 512
            all__new_v = torch.cat(
                [torch.min(self.cri1(all_state[i:i + b_size]),self.cri2(all_state[i:i + b_size]))
                 for i in range(0, all_state.size()[0], b_size)], dim=0) # 这句相当于把[tensor1, tensor2...] cat 成了一个长tensor

        '''compute old_v (old policy value), adv_v (advantage value) 
        refer: GAE. ICLR 2016. Generalization Advantage Estimate. 
        https://arxiv.org/pdf/1506.02438.pdf'''
        all__delta = torch.empty(max_memo, dtype=torch.float32, device=self.device)
        all__old_v = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # old policy value
        all__adv_v = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # advantage value

        prev_old_v = 0  # old q value
        prev_new_v = 0  # new q value
        prev_adv_v = 0  # advantage q value
        for i in range(max_memo - 1, -1, -1):
            all__delta[i] = all_reward[i] + all_mask[i] * prev_new_v - all__new_v[i]
            all__old_v[i] = all_reward[i] + all_mask[i] * prev_old_v
            all__adv_v[i] = all__delta[i] + all_mask[i] * prev_adv_v * lambda_adv

            prev_old_v = all__old_v[i]
            prev_new_v = all__new_v[i]
            prev_adv_v = all__adv_v[i]

        all__adv_v = (all__adv_v - all__adv_v.mean()) / (all__adv_v.std() + 1e-5)  # advantage_norm:

        '''mini batch sample'''
        sample_times = int(repeat_times * max_memo / batch_size)
        for _ in range(sample_times):
            '''random sample'''
            # indices = rd.choice(max_memo, batch_size, replace=True)  # False)
            indices = rd.randint(max_memo, size=batch_size)

            state = all_state[indices]
            action = all_action[indices]
            advantage = all__adv_v[indices]
            old_value = all__old_v[indices].unsqueeze(1)
            old_log_prob = all_log_prob[indices]

            """Adaptive KL Penalty Coefficient
            loss_KLPEN = surrogate_obj + value_obj * lambda_value + entropy_obj * lambda_entropy
            loss_KLPEN = (value_obj * lambda_value) + (surrogate_obj + entropy_obj * lambda_entropy)
            loss_KLPEN = (critic_loss) + (actor_loss)
            """

            '''critic_loss'''
            new_log_prob = self.act.compute__log_prob(state, action)  # it is actor_loss
            new_value1 = self.cri1(state)
            new_value2 = self.cri2(state)

            critic_loss1 = (self.criterion(new_value1, old_value)) / (old_value.std() + 1e-5)
            critic_loss2 = (self.criterion(new_value2, old_value)) / (old_value.std() + 1e-5)
            self.cri1_optimizer.zero_grad()
            critic_loss1.backward()
            self.cri1_optimizer.step()
            self.cri2_optimizer.zero_grad()
            critic_loss2.backward()
            self.cri2_optimizer.step()


            adaptive_clip = clip - 0.04 + np.tanh(np.abs(advantage.cpu().numpy())) *0.08

            '''actor_loss'''
            # surrogate objective of TRPO
            ratio = torch.exp(new_log_prob - old_log_prob)
            ratio_after_clamp = ratio.clone()
            surrogate_obj0 = advantage * ratio
            for i,r in enumerate(ratio):
                ratio_after_clamp[i] = r.clamp(1 - adaptive_clip[i], 1 + adaptive_clip[i])
            surrogate_obj1 = advantage * ratio_after_clamp
            # surrogate_obj1 = advantage * ratio.clamp(1 - adaptive_clip, 1 + adaptive_clip)
            surrogate_obj = -torch.min(surrogate_obj0, surrogate_obj1).mean()
            loss_entropy = (torch.exp(new_log_prob) * new_log_prob).mean()  # policy entropy

            actor_loss = surrogate_obj + loss_entropy * lambda_entropy
            self.act_optimizer.zero_grad()
            actor_loss.backward()
            self.act_optimizer.step()

        self.act.eval()
        self.cri1.eval()
        self.cri2.eval()
        return actor_loss.item(), critic_loss1.item() + critic_loss2.item()