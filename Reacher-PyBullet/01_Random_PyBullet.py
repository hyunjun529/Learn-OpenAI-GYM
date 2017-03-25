import pybullet as p
import numpy as np
import time

from AgentReacher import Reacher

timeStep = 0.01

c = p.connect(p.GUI)
p.resetSimulation()
p.setTimeStep(timeStep)
p.setGravity(0,0,-9.8)

reacher = Reacher()

while True:
    for i in range(50):
        action = np.random.rand(2) * 2 - 1
        obs, reward, done, info = reacher.step(action)
        print(obs)
        time.sleep(timeStep)
    reacher.reset()
