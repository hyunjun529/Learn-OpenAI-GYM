import gym
from gym import wrappers
import ddpg
import tensorflow as tf


env = gym.make('Pendulum-v0')
env.reset()
env.render()

outdir = './log/'
f_act = open(outdir + 'log_act.txt', 'w')
f_obs = open(outdir + 'log_obs.txt', 'w')
f_rwd = open(outdir + 'log_rwd.txt', 'w')
f_info = open(outdir + 'log_info.txt', 'w')

env = wrappers.Monitor(env, directory=outdir, force=True)

num_observation = env.observation_space.shape[0]
num_action = env.action_space.shape[0]

for i_episode in range(101):
    observation = env.reset()

    for t in range(10000):
        env.render()

        # action selection
        action = env.action_space.sample()

        # take the action and observe the reward and next state
        observation, reward, done, info = env.step(action)

        # print observation
        f_act.write(str(action) + "\n")
        f_obs.write(str(observation) + "\n")
        f_rwd.write(str(reward) + "\n")
        f_info.write(str(info) + "\n")

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.monitor.close()
