# https://gym.openai.com/evaluations/eval_9Us92R7KQ7OZYAStOjBR0Q
# ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ
# 랜덤으로 돌려놓고 잘 나온 에제가 있으면 그것만 다시 재생해서 업로드함
# w_hat은 white_hat인가
import gym
import numpy as np
env = gym.make('CartPole-v0')
def eval(env,w,render=False,steps = 200):
    observation = env.reset()
    cum_reward = 0.0
    for i in range(steps):
        if render:
            env.render()
        w_sum = np.sum(w*observation)
        action = 1 if w_sum > 0 else 0
        observation, reward, done, info = env.step(action)
        cum_reward += reward
        if done:
            break
    return cum_reward
#searching for the best cumulative reward across random configurations
w_hat = None # best model
best_reward = 0.0
for i_episode in range(10000):
    w = np.random.rand(1,4)
    cum_reward = eval(env,w)
    if cum_reward > best_reward:
            best_reward = cum_reward
            w_hat = w

print("The best reward is {}".format(best_reward))
print(w_hat)
env.monitor.start('/tmp/cartpole-experiment-1')
eval(env,w_hat,render=True,steps = 10000)
env.monitor.close()
