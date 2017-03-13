import numpy as np
import gym

import time

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam


# Gym
env = gym.make('CartPole-v0')
env._max_episode_steps = 10004
max_episodes = 1500
num_observation = env.observation_space.shape[0]
num_actions = env.action_space.n

# Kears
model = Sequential()
model.add(Flatten(input_shape=(1,) + (num_observation,)))
model.add(Dense(num_actions))
model.add(Activation('linear'))
model.compile(Adam(lr=1e-2), loss='mean_absolute_error', metrics=['mae'])
print(model.summary())

# run Env
for episode in range(max_episodes):
    done = False
    cnt_step = 0
    ary_state = []
    ary_action = []
    ary_reward = []
    e = 1. / ((episode / 10) + 1)
    ob = env.reset()
    while not done:
        #env.render()

        if np.random.rand(1) < e:
            # take a random action
            action = env.action_space.sample()
        else:
            # Choose an action by greedily from the Q-network
            action = np.argmax(model.predict(ob))

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

    batch_state = np.array(ary_state)
    batch_action = np.array(ary_action)
    batch_reward = np.array(ary_reward)
    batch_q = model.predict_on_batch(batch_state)

    #model.fit(ary_state, ary_action, nb_epoch=10, batch_size=100)
    #model.train_on_batch(ary_state, ary_action)

    #model.train_on_batch([batch_action] + batch_state, batch_reward)

    env.reset()

env.close()
