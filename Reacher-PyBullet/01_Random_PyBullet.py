import pybullet as p
import numpy as np
import time

from AgentReacher import Reacher

timeStep = 0.01

c = -1
c = p.connect(p.GUI)
if (c<0):
    c = p.connect(p.GUI)

p.resetSimulation()
p.setTimeStep(timeStep)
p.setGravity(0,0,-9.8)

reacher = Reacher()

# startStateLogging/stopStateLogging
# STATE_LOGGING_VIDEO_MP4

while True:
    for i in range(50):
        action = np.random.rand(2) * 2 - 1
        obs, reward, done, info = reacher.step(action)
        print(reacher.reward_dist(), ", ", reacher.reward_ctrl(action), " : ", reward)
        time.sleep(timeStep)
    reacher.reset()
