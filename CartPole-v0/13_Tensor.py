import logging
import numpy as np
import sys

import tensorflow as tf

import gym
from gym import wrappers


# logging
gym.undo_logger_setup()
logger = logging.getLogger()
formatter = logging.Formatter('[%(asctime)s] %(message)s')
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# Gym
env = gym.make('CartPole-v0')
env._max_episode_steps = 200

outdir = './log/13'
env = wrappers.Monitor(env, directory=outdir, force=True)
env.seed(0)

max_episodes = 1500
num_observation = env.observation_space.shape[0]
num_action = env.action_space.n


# TensorFlow
# https://www.tensorflow.org/get_started/mnist/pros
#https://github.com/hunkim/ReinforcementZeroToAll/blob/master/08_2_softmax_pg_cartpole.py
hidden_layer = 10
learning_rate = 1e-2
gamma = .99

X = tf.placeholder(tf.float32, [None, num_observation], name="input_x")

W1 = tf.get_variable("W1", shape=[num_observation, hidden_layer],
                     initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(X, W1))

W2 = tf.get_variable("W2", shape=[hidden_layer, num_action],
                     initializer=tf.contrib.layers.xavier_initializer())
action_pred = tf.nn.softmax(tf.matmul(layer1, W2))

Y = tf.placeholder(tf.float32, [None, num_action], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal")

log_lik = -Y * tf.log(action_pred)
log_lik_adv = log_lik * advantages
loss = tf.reduce_mean(tf.reduce_sum(log_lik_adv, axis=1))

train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


# dicount reward function
def discount_rewards(r, gamma=0.99):
    """Takes 1d float array of rewards and computes discounted reward
    e.g. f([1, 1, 1], 0.99) -> [1, 0.99, 0.9801] -> [1.22 -0.004 -1.22]
    """
    d_rewards = np.array([val * (gamma ** i) for i, val in enumerate(r)])

    # Normalize/standardize rewards
    d_rewards -= d_rewards.mean()
    d_rewards /= d_rewards.std()
    return d_rewards


# run TensorFlow and TensorBoard
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# run Gym
for episode in range(max_episodes):

    ary_state = np.empty(0).reshape(0, num_observation)
    ary_action = np.empty(0).reshape(0, num_action)
    ary_reward = np.empty(0).reshape(0, 1)

    done = False
    cnt_step = 0
    ob = env.reset()

    while not done:
        # env.render()

        x = np.reshape(ob, [1, num_observation])
        ary_state = np.vstack([ary_state, x])

        action_prob = sess.run(action_pred, feed_dict={X: x})
        action_prob = np.squeeze(action_prob)
        random_noise = np.random.uniform(0, 1, num_action)
        if np.random.rand(1) < (1 - episode / max_episodes):
            action_prob = action_prob + random_noise
        action = np.argmax(action_prob)

        y = np.eye(num_action)[action:action + 1]
        ary_action = np.vstack([ary_action, y])

        ob, reward, done, _ = env.step(action)
        cnt_step += reward
        ary_reward = np.vstack([ary_reward, reward])

        if cnt_step >= 200:
            done = True

    discounted_rewards = discount_rewards(ary_reward)

    ll, la, l, _ = sess.run(
        [log_lik, log_lik_adv, loss, train],
        feed_dict={X: ary_state, Y: ary_action, advantages: discounted_rewards})

    logger.info(str(episode) +  "\t: " +  str(int(cnt_step)) +  "\t: " +  str(l))


input("Y?")

# result
'''
ob = env.reset()
reward_sum = 0
while True:
    env.render()

    x = np.reshape(ob, [1, num_observation])
    action_prob = sess.run(action_pred, feed_dict={X: x})
    action = np.argmax(action_prob)
    ob, reward, done, _ = env.step(action)
    reward_sum += reward
    if done:
        print("Total score: {}".format(reward_sum))
        break
'''

env.close()
gym.upload(outdir)
