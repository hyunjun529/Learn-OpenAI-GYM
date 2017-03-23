import pybullet as p
import numpy as np

class Reacher:
    def __init__(self):
        self.reset()

    def reset(self):
        p.resetSimulation()
        self.reacher = p.loadMJCF("Reacher.xml")
        self.kp = 1
        self.kd = 0.1
        self.maxForce = 100

    def action(self, a):
        p.setJointMotorControl2(
            bodyIndex=6,
            jointIndex=0,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=a[0],
            positionGain=self.kp,
            velocityGain=self.kd,
            force=self.maxForce,
            )
        p.setJointMotorControl2(
            bodyIndex=6,
            jointIndex=2,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=a[1],
            positionGain=self.kp,
            velocityGain=self.kd,
            force=self.maxForce,
            )


    def state(self):
        print("not yet")

    def reward_dist(self):
        finger = np.array(p.getLinkState(6, 4)[4])
        target = np.array(p.getLinkState(7, 2)[4])
        return np.linalg.norm(finger - target)

    def reward(self):
        print("not yet")

    def step(self):
        p.stepSimulation()

'''
#6 == Arm
#6, 0 = Joint0
#6, 2 = Joint1
#6, 4 = FingerTip

#7 == Target
#7, 0 = target_x
#7, 1 = target_y
#7, 2 = targetPosition


>>> p.getJointInfo(6, 0)
(0, b'joint0', 0, 7, 6, 1, 0.0, 0.0)

>>> p.getJointInfo(6, 1)
(1, b'jointfix_2_2', 4, -1, -1, 0, 0.0, 0.0)

>>> p.getJointInfo(6, 2)
(2, b'joint1', 0, 8, 7, 1, 0.0, 0.0)

>>> p.getJointInfo(6, 3)
(3, b'jointfix_1_4', 4, -1, -1, 0, 0.0, 0.0)

>>> p.getJointInfo(6, 4)
(4, b'jointfix_0_3', 4, -1, -1, 0, 0.0, 0.0)


>>> p.getJointInfo(7, 0)
(0, b'target_x', 1, 7, 6, 1, 0.0, 0.0)

>>> p.getJointInfo(7, 1)
(1, b'target_y', 1, 8, 7, 1, 0.0, 0.0)

>>> p.getJointInfo(7, 2)
(2, b'jointfix_3_3', 4, -1, -1, 0, 0.0, 0.0)


>>> p.getJointState(6, 0)
(0.0, 0.0, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0)

>>> p.getJointState(6, 2)
(0.0, 0.0, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0)


>>> p.getLinkState(6, 4)
(
(0.21000000834465027, 0.0, 0.009999999776482582),
(0.0, 0.0, 0.0, 1.0),
(0.0, 0.0, 0.0),
(0.0, 0.0, 0.0, 1.0),
(0.21000000834465027, 0.0, 0.009999999776482582),
(0.0, 0.0, 0.0, 1.0)
)

>>> p.getLinkState(7, 2)
(
(0.10000000149011612, -0.10000000149011612, 0.009999999776482582),
(0.0, 0.0, 0.0, 1.0),
(0.0, 0.0, 0.0),
(0.0, 0.0, 0.0, 1.0),
(0.10000000149011612, -0.10000000149011612, 0.009999999776482582),
(0.0, 0.0, 0.0, 1.0)
)


>>> p.getLinkState(6, 4)[4]
(0.21000000834465027, 0.0, 0.009999999776482582)

>>> p.getLinkState(7, 2)[4]
(0.10000000149011612, -0.10000000149011612, 0.009999999776482582)


>>> import numpy as np
>>> aa = np.array(a)
>>> aa
array([ 0.21000001,  0.        ,  0.01      ])
>>> ab = np.array(b)
>>> ab
array([ 0.1 , -0.1 ,  0.01])


>>> np.linalg.norm(aa - ab)
0.14866069354749017

# vec = self.get_body_com("fingertip")-self.get_body_com("target")


'''
