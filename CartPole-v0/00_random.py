import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(100):
    for _ in range(150):
        env.render()

        # https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py#L59
        # about action_space
        # force = self.force_mag if action==1 else -self.force_mag
        # force_mag = 10.0
        # x와 theta 변화량에 직접 작용(temp 62 line)
        ob, reward, done, _ = env.step(env.action_space.sample()) # take a random action


        # https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py#L58
        # about observation
        # x, x_dot, theta, theta_dot = state
        # x = 0을 중심으로 Agent cart의 위치 좌표
        # x_dot = cart의 가속도
        # theta = 수직을 기준으로 pole이 움직인 각도의 변화량
        # theta_dot = pole(theta)의 가속도

        # https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py#L76
        # about reward
        # not done = 1.0
        # done = 0

        # https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py#L70
        # about done (end of episode)
        # x가 밖으로 나가는 경우( x < -2.4 or x > 2.4 )
        # or theta가 약 0.4 이상 기울어진 경우( 12 * 2 * math.pi / 360 )

        # 평균 10~30 episode에서 done이 됨
        print(ob, ", ", reward, ", ", done)

    env.reset()
