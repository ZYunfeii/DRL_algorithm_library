from DDPGModel import *
import gym
import matplotlib.pyplot as plt
from PPO.draw import Painter
import random
from copy import deepcopy

def calcu_reward(new_goal, state, action, mode='her'):
    # direcly use observation as goal

    if mode == 'shaping':
        # shaping reward
        goal_cos, goal_sin, goal_thdot = new_goal[0], new_goal[1], new_goal[2]
        cos_th, sin_th, thdot = state[0], state[1], state[2]
        costs = (goal_cos - cos_th) ** 2 + (goal_sin - sin_th) ** 2 + 0.1 * (goal_thdot - thdot) ** 2
        reward = -costs
    elif mode == 'her':
        # binary reward, no theta now
        tolerance = 0.5
        goal_cos, goal_sin, goal_thdot = new_goal[0], new_goal[1], new_goal[2]
        cos_th, sin_th, thdot = state[0], state[1], state[2]
        costs = (goal_cos - cos_th) ** 2 + (goal_sin - sin_th) ** 2 + 0.1 * (goal_thdot - thdot) ** 2
        reward = 0 if costs < tolerance else -1
    else:
        raise Exception('error for mode!')
    return reward

def generate_goals(i, episode_cache, sample_num, sample_range = 200):
    '''
    Input: current steps, current episode transition's cache, sample number
    Return: new goals sets
    notice here only "future" sample policy
    '''
    end = (i+sample_range) if i+sample_range < len(episode_cache) else len(episode_cache)
    epi_to_go = episode_cache[i:end]
    if len(epi_to_go) < sample_num:
        sample_trans = epi_to_go
    else:
        sample_trans = random.sample(epi_to_go, sample_num)
    return [np.array(trans[3][:3]) for trans in sample_trans]

def gene_new_sas(new_goals, transition):
    state, new_state = transition[0][:3], transition[3][:3]
    action = transition[1]
    state = np.concatenate((state, new_goals))
    new_state = np.concatenate((new_state, new_goals))
    return state, action, new_state

def evaluate(env,agent,episode):
    reward = 0
    pendulum_goal = np.array([1, 0, 0], dtype=np.float32)
    goal = pendulum_goal
    o = env.reset()
    o = np.concatenate((o, goal))
    for _ in range(500):
        # if episode > 80: env.render()
        a = agent.get_action(o, None, deterministic=True)
        o2, r, d, _ = env.step(a)
        o2 = np.concatenate((o2, goal))
        reward += r
        o = o2
        if d: break
    return reward

if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    eval_env = deepcopy(env)
    pendulum_goal = np.array([1.0, 0.0, 0.0], dtype=np.float32) # HER needs goal
    goal = pendulum_goal
    obs_dim = env.observation_space.shape[0] *2 # HER needs more obs_dim
    act_dim = env.action_space.shape[0]
    act_bound = [-env.action_space.high[0], env.action_space.high[0]]

    ddpg = DDPG(obs_dim, act_dim, act_bound)

    MAX_EPISODE = 100
    MAX_STEP = 500
    HER_SAMPLE_NUM = 4
    update_every = 200
    batch_size = 256
    reward_list = []
    her_reward_list = []
    for episode in range(MAX_EPISODE):
        o = env.reset()
        o = np.concatenate((o, goal))  # concatenate
        ep_reward = 0
        episode_cache = []  # HER needs to store the trajectory
        for j in range(MAX_STEP):
            if episode > 5:
                a = ddpg.get_action(o, ddpg.act_noise)
            else:
                a = env.action_space.sample()
            o2, _, d, _ = env.step(a)
            o2 = np.concatenate((o2,goal)) # concatenate
            r = calcu_reward(goal, o, a)  # HER has two kinds of reward
            ep_reward += r
            episode_cache.append((o,a,r,o2))
            ddpg.replay_buffer.store(o, a, r, o2, d)
            o = o2
            if d: break


        # Hindsight replay: Important operation of HER
        for i, transition in enumerate(episode_cache):
            new_goals = generate_goals(i, episode_cache, HER_SAMPLE_NUM)
            for new_goal in new_goals:
                o, a = transition[0], transition[1]
                r = calcu_reward(new_goal, o, a)
                o, a, o2 = gene_new_sas(new_goal, transition)
                ddpg.replay_buffer.store(o, a, r, o2, False)

        if episode >= 5:
            for _ in range(update_every):
                batch = ddpg.replay_buffer.sample_batch(batch_size)
                ddpg.update(data=batch)

        costheta = np.random.rand()  # The paper suggests to use multiple goals
        sintheta = np.sqrt(1-costheta**2)
        w = 2 * np.random.rand()
        goal = np.array([costheta,sintheta,w])

        eval_reward = evaluate(eval_env,ddpg,episode)
        reward_list.append(eval_reward)
        her_reward_list.append(ep_reward)
        print('Episode:', episode, 'HER Reward:%i' % int(ep_reward),'Eval Reward:%i'% int(eval_reward))

    painter = Painter(load_csv=True,load_dir='HER.csv')
    painter.addData(reward_list,'DDPG-HER')
    painter.saveData(save_dir='HER.csv')
    painter.drawFigure()





