import numpy as np
from numpy import pi
from numpy.linalg import norm
from Utils.utils import *

import pybullet as p

class UR5_Agent:
    def __init__(self, ur5_start_pos, ur5_start_ori, ur5_reset_joint_pos, ur5_reset_joint_vel,
                 action_dim, ur5_control_mode, ur5_force_limit, urdf_path):
        '''
        :param ur5_start_pos:       [0, -0.4, 0.607]
        :param ur5_start_ori:       [0., 0., 0., 1.]
        :param ur5_reset_joint_pos: [pi/2, -pi/2, pi/2, -pi/2, 0, 0, 0]
        :param ur5_reset_joint_vel: [0, 0, 0, 0, 0, 0]
        :param action_dim:          6
        :param ur5_control_mode:    'velocity'
        :param ur5_force_limit:     [150, 150, 150, 28, 28, 28]
        :param urdf_path:           'urdf/ur5.urdf'
        '''
        self.ur5_start_pos = ur5_start_pos
        self.ur5_start_ori = ur5_start_ori
        self.ur5_reset_joint_pos = ur5_reset_joint_pos
        self.ur5_reset_joint_vel = ur5_reset_joint_vel
        self.action_dim = action_dim
        if ur5_control_mode == 'velocity':
            self.ur5_control_mode = p.VELOCITY_CONTROL
        elif ur5_control_mode == 'position':
            self.ur5_control_mode = p.POSITION_CONTROL
        self.ur5_force_limit = ur5_force_limit
        self.urdf_path = urdf_path

        self.id = p.loadURDF(self.urdf_path, self.ur5_start_pos, self.ur5_start_ori, flags=p.URDF_USE_SELF_COLLISION)

    def reset(self):
        p.resetBasePositionAndOrientation(self.id, self.ur5_start_pos, self.ur5_start_ori)
        for i in range(self.action_dim):
            p.resetJointState(self.id, i, targetValue=self.ur5_reset_joint_pos[i], targetVelocity=self.ur5_reset_joint_vel[i])
            p.setJointMotorControl2(self.id, i, controlMode=self.ur5_control_mode, targetVelocity=self.ur5_reset_joint_vel[i], force=self.ur5_force_limit[i])

    def check_init(self):
        joint_pos = self.get_joint_pos()
        joint_vel = self.get_joint_vel()
        agent_pos = self.get_pos()
        agent_quat = self.get_quat()

        err_joint_pos = norm(np.sin(self.ur5_reset_joint_pos) - np.sin(joint_pos)) + \
                        norm(np.cos(self.ur5_reset_joint_pos) - np.cos(joint_pos))
        err_joint_vel = norm(np.array(joint_vel) - np.array(self.ur5_reset_joint_vel))
        err_agent_pos = norm(agent_pos - self.ur5_start_pos)
        err_agent_quat = cal_quat_error(agent_quat, self.ur5_start_ori)
        return (err_joint_pos + err_joint_vel + err_agent_pos + err_agent_quat < 2e-3)

    def make_action(self, action):
        joint_vel = np.hstack((action, np.zeros(self.action_dim - action.shape[0])))
        p.setJointMotorControlArray(self.id, [0, 1, 2, 3, 4, 5], controlMode=self.ur5_control_mode,
                                    targetVelocities=joint_vel.tolist(), forces=self.ur5_force_limit)

    def get_pos(self):
        return np.array(p.getBasePositionAndOrientation(self.id)[0])

    def get_ori(self):
        return np.array(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.id)[1]))

    def get_quat(self):
        return np.array(p.getBasePositionAndOrientation(self.id)[1])

    def get_end_effector_pos(self):
        return np.array(p.getLinkState(self.id, self.action_dim)[0])

    def get_end_effector_ori(self):
        return np.array(p.getQuaternionFromEuler(p.getLinkState(self.id, self.action_dim)[1]))

    def get_end_effector_quat(self):
        return np.array(p.getLinkState(self.id, self.action_dim)[1])

    def get_joint_pos(self):
        x = []
        for i in range(self.action_dim):
            x.append(p.getJointState(self.id, i)[0])
        return x

    def get_joint_vel(self):
        x = []
        for i in range(self.action_dim):
            x.append(p.getJointState(self.id, i)[1])
        return x

class SpaceObject:
    def __init__(self, start_pos, start_ori, urdf_path):
        '''
        :param start_pos:       [0, 0, 0] (satellite) / [-0.031, -0.944, 1.114] (mug)
        :param start_ori:       [0., 0., 0., 1.]
        :param urdf_path:       'urdf/satellite.urdf' (satellite) / 'urdf/mug.urdf' (mug)
        '''
        self.start_pos = start_pos
        self.start_ori = start_ori
        self.urdf_path = urdf_path

        self.id = p.loadURDF(self.urdf_path, self.start_pos, self.start_ori)

    def reset(self, pos, ori):
        self.start_pos = pos
        self.start_ori = ori
        p.resetBasePositionAndOrientation(self.id, self.start_pos, self.start_ori)

    def check_init(self):
        object_pos = self.get_pos()
        object_quat = self.get_quat()

        err_object_pos = norm(object_pos - self.start_pos)
        err_object_quat = cal_quat_error(object_quat, self.start_ori)
        return (err_object_pos + err_object_quat < 1e-4)

    def get_pos(self):
        return np.array(p.getBasePositionAndOrientation(self.id)[0])

    def get_ori(self):
        return np.array(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.id)[1]))

    def get_quat(self):
        return np.array(p.getBasePositionAndOrientation(self.id)[1])
