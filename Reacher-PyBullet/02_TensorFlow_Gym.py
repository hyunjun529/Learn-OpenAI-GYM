import gym
from gym import wrappers

import numpy as np
import tensorflow as tf


# init Gym
env = gym.make('Reacher-v1')
env.seed(0)
env.reset()
env.render()

# init Variables
max_episodes = 100000 
batch_size = 1000
outdir = './log/'
num_observation = env.observation_space.shape[0]
num_action = env.action_space.shape[0]


# start Monitor
env = wrappers.Monitor(env, directory=outdir, force=True)


# TensorFlow
# https://www.tensorflow.org/get_started/mnist/pros
#https://github.com/hunkim/ReinforcementZeroToAll/blob/master/08_2_softmax_pg_cartpole.py
hidden_layer = 1000
learning_rate = 1e-3
gamma = .995

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
    #return r


# run TensorFlow and TensorBoard
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# run Gym
ary_state = np.empty(0).reshape(0, num_observation)
ary_action = np.empty(0).reshape(0, num_action)
ary_reward = np.empty(0).reshape(0, 1)

for episode in range(max_episodes):

    done = False
    cnt_step = 0
    ob = env.reset()

    '''
    for t in range(100):
        env.render()

        action = env.action_space.sample()

        ob, reward, done, info = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    '''

    while not done:
        if episode % (batch_size/4) == 0:
            env.render()

        x = np.reshape(ob, [1, num_observation])
        ary_state = np.vstack([ary_state, x])

        action_prob = sess.run(action_pred, feed_dict={X: x})
        action_prob = np.squeeze(action_prob)
        random_noise = np.random.uniform(-1, 1, num_action)
        if np.random.rand(1) < (1 - episode / max_episodes):
            action_prob = random_noise
        action = action_prob

        # y = np.eye(num_action)[action:action + 1]
        ary_action = np.vstack([ary_action, action_prob])

        ob, reward, done, _ = env.step(action)
        cnt_step += reward
        ary_reward = np.vstack([ary_reward, reward])

        if episode % (batch_size/4) == 0:
            print("[{:04d}] {} : {}".format(episode, action, reward))

    if episode % batch_size == 0:

        discounted_rewards = discount_rewards(ary_reward)

        ll, la, l, _ = sess.run(
            [log_lik, log_lik_adv, loss, train],
            feed_dict={X: ary_state, Y: ary_action, advantages: discounted_rewards})

        ary_state = np.empty(0).reshape(0, num_observation)
        ary_action = np.empty(0).reshape(0, num_action)
        ary_reward = np.empty(0).reshape(0, 1)

env.close()
