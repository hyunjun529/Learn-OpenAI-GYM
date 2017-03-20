import pybullet as p
import time

timeStep = 0.01
c = -1

if (c<0):
    c = p.connect(p.GUI)

p.resetSimulation()
p.setTimeStep(timeStep)
p.loadBullet("multibody.bullet")
p.setGravity(0,0,-10)

p.setRealTimeSimulation(1)

while True:
    continue
