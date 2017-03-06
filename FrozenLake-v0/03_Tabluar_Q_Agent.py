# https://github.com/openai/gym/blob/master/examples/agents/random_agent.py
import argparse
import logging
import sys
import numpy as np

import gym
from gym import wrappers


class TabularQAgent(object):
    """The world's simplest agent!"""
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.Q = np.zeros([self.observation_space.n, self.action_space.n])
        self.learning_rate = 0.85
        self.dis = 0.99

    def act(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='FrozenLake-v0', help='Select the environment to run')
    args = parser.parse_args()

    # Call `undo_logger_setup` if you want to undo Gym's logger setup
    # and configure things manually. (The default should be fine most
    # of the time.)
    gym.undo_logger_setup()
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # You can set the level to logging.DEBUG or logging.WARN if you
    # want to change the amount of output.
    logger.setLevel(logging.INFO)

    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = './log/03'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = TabularQAgent(env.observation_space, env.action_space)

    # h529 : 총 실행되는 epsiod의 수
    episode_count = 10000
    reward = 0
    done = False

    rList = []
    for i in range(episode_count):
        ob = env.reset()
        rAll = 0
        done = False
        while not done:
            action = np.argmax(agent.Q[ob, :] + np.random.randn(1, env.action_space.n) / (i + 1))
            ob2, reward, done, _ = env.step(action)
            agent.Q[ob, action] = reward + agent.dis * np.max(agent.Q[ob2, :])
            ob = ob2
            rAll += reward
        rList.append(rAll)

    logger.info("Score over time" + str(sum(rList) / episode_count))
    logger.info("Final Q-Table Values")
    logger.info(agent.Q)

    # Close the env and write monitor result info to disk
    env.close()

    # Upload to the scoreboard. We could also do this from another
    # process if we wanted.
    logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    #gym.upload(outdir)
