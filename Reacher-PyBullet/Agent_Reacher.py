import pybullet as p

class AgentReacher:
    def __init__(self):
        self.reset()

    def reset(self):
        self.quadruped = p.loadURDF("Reacher.xml", 0, 0, .3)
