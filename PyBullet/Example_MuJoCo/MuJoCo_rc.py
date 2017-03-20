import pybullet as p
import tensorflow as tf
import time

hidden_layer = 10
learning_rate = 1e-2
gamma = .99

num_observation = 2
num_action = 2

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

sess = tf.Session()
sess.run(tf.global_variables_initializer())


timeStep = 0.01
c = -1

if (c<0):
    c = p.connect(p.GUI)

p.resetSimulation()
p.setTimeStep(timeStep)
p.loadMJCF("Reacher.xml")
p.setGravity(0,0,-10)

while True:
    p.stepSimulation()
    time.sleep(timeStep)
    input(":D")
