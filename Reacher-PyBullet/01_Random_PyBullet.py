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

while True:
    for i in range(200):
        reacher.action(np.random.rand(2) * 200 - 100)
        print(reacher.reward_dist())
        reacher.step()
        time.sleep(timeStep)
    reacher.reset()
