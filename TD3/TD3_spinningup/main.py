from td3 import TD3
import gym
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    td3 = TD3(obs_dim,act_dim)

    MAX_EPISODE = 100
    MAX_STEP = 500
    update_every = 50
    batch_size = 100
    rewardList = []
    for episode in range(MAX_EPISODE):
        o = env.reset()
        ep_reward = 0
        for j in range(MAX_STEP):
            if episode > 20:
                a = td3.get_action(o, td3.act_noise)*2
            else:
                a = env.action_space.sample()
            o2, r, d, _ = env.step(a)
            td3.replay_buffer.store(o, a, r, o2, d)

            if episode >= 20 and j % update_every == 0:
                td3.update(batch_size,update_every)

            o = o2
            ep_reward += r

            if d: break
        print('Episode:', episode, 'Reward:%i' % int(ep_reward))
        rewardList.append(ep_reward)

    plt.figure()
    plt.plot(np.arange(len(rewardList)),rewardList)
    plt.show()





