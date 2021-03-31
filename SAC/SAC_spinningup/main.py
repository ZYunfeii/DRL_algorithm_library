from SACModel import *
import gym
import matplotlib.pyplot as plt


if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_bound = [-env.action_space.high[0], env.action_space.high[0]]

    sac = SAC(obs_dim, act_dim, act_bound)

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
                # env.render()
                a = sac.get_action(o)
            else:
                a = env.action_space.sample()
            o2, r, d, _ = env.step(a)
            sac.replay_buffer.store(o, a, r, o2, d)

            if episode >= 10 and j % update_every == 0:
                for _ in range(update_every):
                    batch = sac.replay_buffer.sample_batch(batch_size)
                    sac.update(data=batch)

            o = o2
            ep_reward += r

            if d:
                break
        print('Episode:', episode, 'Reward:%i' % int(ep_reward))
        rewardList.append(ep_reward)

    plt.figure()
    plt.plot(np.arange(len(rewardList)),rewardList)
    plt.show()
