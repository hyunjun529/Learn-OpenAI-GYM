import numpy as np
import gym

import time

# Gym
env = gym.make('CartPole-v0')
env._max_episode_steps = 10004
max_episodes = 1500
num_observation = env.observation_space.shape[0]
num_actions = env.action_space.n

# run Env
for episode in range(max_episodes):
    done = False
    cnt_step = 0
    ary_state = []
    ary_action = []
    ary_reward = []
    ob = env.reset()
    while not done:
        #env.render()

        # take a random action
        action = env.action_space.sample()

        ob_next, reward, done, _ = env.step(action)
        if done:  # big penalty
            reward = -100.0

        cnt_step += 1
        ary_state.append(ob.tolist())
        ary_reward.append(reward)
        ary_action.append(action)

        ob = ob_next

    # episode done
    print("========================================================")
    print("episode {}, cnt_step = {} : ".format(episode, cnt_step))
    for i in range(cnt_step):
        print(ary_state[i], ", ", ary_action[i], ", ", ary_reward[i])
    print("========================================================")

    env.reset()
