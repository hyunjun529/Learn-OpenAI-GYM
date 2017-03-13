import gym
env = gym.make('InvertedPendulum-v1')
env.reset()
for _ in range(100):
    for _ in range(150):
        env.render()
        ob, reward, done, _ = env.step(env.action_space.sample()) # take a random action
        print(ob, ", ", reward, ", ", done)
    env.reset()
