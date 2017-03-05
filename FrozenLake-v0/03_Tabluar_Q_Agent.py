# https://github.com/openai/gym/blob/master/examples/agents/random_agent.py
import argparse
import logging
import sys
import numpy as np
from collections import defaultdict

import gym
from gym import wrappers


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


class TabularQAgent(object):
    """
    Agent implementing tabular Q-learning.
    """

    def __init__(self, observation_space, action_space, **userconfig):
        #if not isinstance(observation_space, discrete.Discrete):
        #    raise UnsupportedSpace('Observation space {} incompatible with {}. (Only supports Discrete observation spaces.)'.format(observation_space, self))
        #if not isinstance(action_space, discrete.Discrete):
        #    raise UnsupportedSpace('Action space {} incompatible with {}. (Only supports Discrete action spaces.)'.format(action_space, self))
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_n = action_space.n
        self.config = {
            "init_mean" : 0.0,      # Initialize Q values with this mean
            "init_std" : 0.0,       # Initialize Q values with this standard deviation
            "learning_rate" : 0.1,
            "eps": 0.05,            # Epsilon in epsilon greedy policies
            "discount": 0.95,
            "n_iter": 10000}        # Number of iterations
        self.config.update(userconfig)
        self.q = defaultdict(lambda: self.config["init_std"] * np.random.randn(self.action_n) + self.config["init_mean"])

    def act(self, observation, eps=None):
        if eps is None:
            eps = self.config["eps"]
        # epsilon greedy.
        action = np.argmax(self.q[observation.item()]) if np.random.random() > eps else self.action_space.sample()
        return action

    def learn(self, env):
        config = self.config
        obs = env.reset()
        q = self.q
        for t in range(config["n_iter"]):
            action, _ = self.act(obs)
            obs2, reward, done, _ = env.step(action)
            future = 0.0
            if not done:
                future = np.max(q[obs2.item()])
            q[obs.item()][action] -= \
                self.config["learning_rate"] * (q[obs.item()][action] - reward - config["discount"] * future)

            obs = obs2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v0', help='Select the environment to run')
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

    # h529 : create env
    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = './log/03'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)

    # h529 : Agnet Setting
    # agent = RandomAgent(env.action_space)
    agent = TabularQAgent(env.observation_space ,env.action_space)


    # h529 : 총 실행되는 epsiod의 수
    episode_count = 1000
    reward = 0
    done = False

    # h529 : episode
    for i in range(episode_count):
        ob = env.reset()
        while True:
            action = agent.act(ob)
            ob, reward, done, _ = env.step(action)
            if done:
                action.learn(env)
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.close()

    # Upload to the scoreboard. We could also do this from another
    # process if we wanted.
    logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    #gym.upload(outdir)
