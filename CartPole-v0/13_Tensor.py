import numpy as np
import tensorflow as tf
import gym

import time

# Gym
env = gym.make('CartPole-v0')
env._max_episode_steps = 501
max_episodes = 1500
num_observation = env.observation_space.shape[0]
num_actions = env.action_space.n

# TensorFlow
# https://www.tensorflow.org/get_started/mnist/pros
#https://github.com/hunkim/ReinforcementZeroToAll/blob/master/08_2_softmax_pg_cartpole.py
hidden_layer = 10
learning_rate = 1e-2
gamma = .99

X = tf.placeholder(tf.float32, [None, num_observation], name="input_x")

W1 = tf.Variable(tf.zeros([num_observation, hidden_layer]), name="W1")
layer1 = tf.nn.relu(tf.matmul(X, W1))

W2 = tf.Variable(tf.zeros([hidden_layer, num_actions]), name="W2")
action_pred = tf.nn.softmax(tf.matmul(layer1, W2))

Y = tf.placeholder(tf.float32, [None, num_actions], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal")

log_lik = -Y * tf.log(action_pred)
log_lik_adv = log_lik * advantages
loss = tf.reduce_mean(tf.reduce_sum(log_lik_adv, axis=1))

train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


# run TensorFlow and TensorBoard
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# run Gym
for episode in range(max_episodes):

    ary_state = []
    ary_action = []
    ary_reward = []

    done = False
    cnt_step = 0
    ob = env.reset()
    
    while not done:
        # env.render()

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

env.close()
