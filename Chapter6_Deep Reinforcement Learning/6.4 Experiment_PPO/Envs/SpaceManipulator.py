import os
import time
import numpy as np
from numpy import pi
import cv2
from scipy.spatial.transform import Rotation as R
import copy
np.set_printoptions(precision=3, suppress=True)
from numpy.linalg import norm
import platform
import torch

import math
import sys

sys.path.append('..')

from Utils.utils import *
from gym.utils import seeding

import pybullet as p
import pybullet_data

from Envs.objects import UR5_Agent, SpaceObject

_2pi = 2 * pi


# abbreviation
#     pose: pose;
#     pos:  position(s);
#     ori:  orientation;
#     quat: quaternion;
#     vel:  velocity(ies);

class SpaceManipulator:
    def __init__(self, args, plat='wsl'):
        super(SpaceManipulator, self).__init__()
        ''' DATA FROM MAIN CODE '''
        self.plat = plat
        self.headless = args.headless

        self.action_dim = args.action_dim
        self.time_interval = 100

        if self.headless:
            physicsClient = p.connect(p.DIRECT)
        else:
            physicsClient = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # 关闭整个调试GUI，包括网格

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)  # 设置零重力

        self.ur5_start_pos = [0, -0.4, 0.607]
        self.ur5_start_ori = p.getQuaternionFromEuler([0, 0, 0])
        self.ur5_reset_joint_pos = [pi / 2, -pi / 2, pi / 2, -pi / 2, 0, 0, ]
        self.ur5_reset_joint_vel = [0, 0, 0, 0, 0, 0, ]
        self.ur5_force_limit = [150, 150, 150, 28, 28, 28]

        self.satellite_start_pos = [0, 0, 0]
        self.satellite_start_ori = p.getQuaternionFromEuler([0, 0, 0])
        self.mug_start_pos = [0.0, 0.5, 1]
        self.mug_start_ori = p.getQuaternionFromEuler([0, 0, 0])

        # 加载 航天器 / 机械臂 / 目标 模型
        self.dummyShape = p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=(1, 1, 1, 1))
        self.dummyId = p.createMultiBody(baseVisualShapeIndex=self.dummyShape, baseCollisionShapeIndex=-1)
        p.resetBasePositionAndOrientation(self.dummyId, np.array([1, 1, 1]), np.array([0, 0, 0, 1]))
        self.agent = UR5_Agent(
            self.ur5_start_pos,
            self.ur5_start_ori,
            self.ur5_reset_joint_pos,
            self.ur5_reset_joint_vel,
            self.action_dim,
            'velocity',
            self.ur5_force_limit,
            'urdf/ur5.urdf'
        )
        self.floating_base = SpaceObject(self.satellite_start_pos, self.satellite_start_ori, 'urdf/satellite.urdf')
        self.target = SpaceObject(self.mug_start_pos, self.mug_start_ori, 'urdf/mug.urdf')

        # 设置无阻尼和无摩擦力
        p.changeDynamics(self.agent.id, -1, linearDamping=0, angularDamping=0, lateralFriction=0, spinningFriction=0,
                         rollingFriction=0)
        p.changeDynamics(self.floating_base.id, -1, linearDamping=0, angularDamping=0, lateralFriction=0,
                         spinningFriction=0, rollingFriction=0)
        self.constraint = p.createConstraint(
            parentBodyUniqueId=self.floating_base.id,
            parentLinkIndex=-1,  # 连接根链接
            childBodyUniqueId=self.agent.id,
            childLinkIndex=-1,  # 连接根链接
            jointType=p.JOINT_FIXED,  # 旋转关节
            jointAxis=[0, 0, 0],  # 关节绕 Z 轴旋转
            parentFramePosition=self.ur5_start_pos,  # 关节在 cubeA 上的位置
            childFramePosition=[0, 0, 0]  # 关节在 cubeB 上的位置
        )

        p.setTimeStep(self.time_interval / 1000.)  # 100 ms

        # region
        self.max_episode_steps = args.max_episode_steps

        self.is_train = args.is_train

        self.r_max, self.r_min = args.r_max, args.r_min

        self.episode_num = 0
        self.action = np.zeros(self.action_dim)
        self.last_action = self.action

        self.success_interval = args.success_interval

        self.pos_err_queue = []
        self.ori_err_queue = []
        self.pos_err_stable_times = args.pos_err_stable_times
        self.pos_err_thres = args.pos_err_thres
        self.success_sample_rate = args.success_sample_rate
        self.success_min_rate = args.success_min_rate

        self.ori_err_stable_times = args.pos_err_stable_times
        self.ori_err_thres = args.ori_err_thres
        self.ori_penalty_rate = args.ori_penalty_rate

        self.fail_prob_decrease = args.fail_prob_decrease
        self.decrease_threshold = args.decrease_threshold

        self.eval_episodes = 20

        ''' position zero origin define '''
        pos_zero_origins = ['ee', 'ab']
        self.pos_zero_origin = args.pos_zero_origin
        assert (self.pos_zero_origin in pos_zero_origins)

        self.end_effector_axis_list = []

        # env reset
        self.floating_base.reset(self.satellite_start_pos, self.satellite_start_ori)
        self.agent.reset()
        self.target.reset(self.mug_start_pos, self.mug_start_ori)

        ######### get floating_base pos/quat #########
        self.init_floating_base_pos = self.floating_base.get_pos()
        self.init_floating_base_quat = self._ensure_quat_continuity(self.floating_base.get_quat())
        self.init_floating_base_quat_ = self.init_floating_base_quat
        self.floating_base_pos = self.init_floating_base_pos
        self.floating_base_quat = self.init_floating_base_quat
        self.floating_base_quat_ = self.init_floating_base_quat_
        ######### get agent pos/quat #########
        self.init_agent_pos = self.agent.get_pos()
        self.init_agent_quat = self._ensure_quat_continuity(self.agent.get_quat())
        self.init_agent_quat_ = self.init_agent_quat
        self.agent_pos = self.init_agent_pos
        self.agent_quat = self.init_agent_quat
        self.agent_quat_ = self.init_agent_quat_
        ######### get joint pos/vel #########
        self.init_joint_pos = self.agent.get_joint_pos()
        self.init_joint_vel = self.agent.get_joint_vel()

        for i in range(self.action_dim):
            self.init_joint_pos[i] = self.init_joint_pos[i] % _2pi
        self.joint_pos = copy.copy(self.init_joint_pos)
        self.joint_vel = copy.copy(self.init_joint_vel)
        ######### get target pos/quat #########
        self.init_target_pos = self.target.get_pos()
        self.init_target_quat = self._ensure_quat_continuity(self.target.get_quat())
        self.init_target_quat_ = self.init_target_quat
        self.init_target_euler = p.getEulerFromQuaternion(self.init_target_quat)
        self.start_target_pos = self.init_target_pos
        self.start_target_quat = self.init_target_quat
        self.start_target_quat_ = self.init_target_quat_
        self.start_target_euler = self.init_target_euler
        self.target_pos = self.init_target_pos
        self.target_quat = self.init_target_quat
        self.target_quat_ = self.init_target_quat_
        self.target_euler = self.init_target_euler
        ######### get end_effector pos/quat #########
        self.init_end_effector_pos = self.agent.get_end_effector_pos()
        self.init_end_effector_quat = self._ensure_quat_continuity(self.agent.get_quat())
        self.init_end_effector_quat_ = self.init_end_effector_quat
        self.init_end_effector_euler = p.getEulerFromQuaternion(self.init_end_effector_quat)
        self.end_effector_pos = self.init_end_effector_pos
        self.end_effector_quat = self.init_end_effector_quat
        self.end_effector_quat_ = self.init_end_effector_quat_
        self.end_effector_euler = self.init_end_effector_euler
        self.tar_euler_relative_to_ee = self._e2e(self.end_effector_euler, self.target_euler)

        self.target_collision, self.last_target_collision = False, False

        self.last_state = None
        self.eet_ = None
        self.last_err_pos, self.last_err_quat, self.last_err_euler = None, None, None
        self.last_dis_pos, self.last_dis_ori = None, None
        self.pos_reach_list, self.ori_reach_list = [], []
        self.done = False
        self.his_obs = {}
        self._history_observation_reset(self.his_obs)

        self._make_observation()
        self._last_refresh()

    def reset(self, pos=None, ori=None):
        self.pos_err_queue, self.ori_err_queue = [], []
        # history observation
        self.his_obs = {}
        self._history_observation_reset(self.his_obs)

        self.step_count = 0
        self.episode_num += 1
        self.target_collision, self.last_target_collision = True, True
        self.action = np.zeros(self.action_dim)
        self.last_action = self.action

        self.eet_ = None

        self.pos_reach_list, self.ori_reach_list = [], []
        self.done = False

        self._set_floating_base()
        self._set_agent()
        self._set_target()

        self.init_end_effector_pos = self.agent.get_end_effector_pos()
        self.init_agent_pos = self.agent.get_pos()

        self._make_observation()  # Update state
        self._last_refresh()
        p.resetBasePositionAndOrientation(self.dummyId, self.target_pos, self.target_quat)

        return self.state

    def _set_target(self):
        self._target_reinit()
        self.target.reset(self.start_target_pos, self.start_target_quat)

    def _check_target_init(self):
        return self.target.check_init()

    def _set_floating_base(self):
        self.floating_base.reset(self.satellite_start_pos, self.satellite_start_ori)

    def _check_floating_base_init(self):
        return self.floating_base.check_init()

    def _set_agent(self):
        self.agent.reset()

    def _check_agent_init(self):
        return self.agent.check_init()

    def _reset_finish(self):
        return self.r_min < norm(self.target.get_pos() - self.agent.get_pos()) < self.r_max + 0.05

    def _set_max_episode_steps(self, max_episode_steps):
        self.max_episode_steps = max_episode_steps

    def _reach_check(self):
        self.pos_reach_list.append(self.dis_pos < self.pos_err_thres)
        self.ori_reach_list.append(self.dis_ori < self.ori_err_thres)


        self.done = self.pos_reach_list[-1] and self.ori_reach_list[-1]

        while len(self.pos_reach_list) > self.pos_err_stable_times:
            del (self.pos_reach_list[0])
        while len(self.ori_err_queue) > self.ori_err_stable_times:
            del (self.ori_err_queue[0])

    def step(self, action):
        '''
        Step the vrep simulation by one frame, make actions and observations and calculate the resulting
        rewards
        action: angular velocities
        '''
        self.step_count += 1
        self.action = action

        self._make_action(action)
        self.pybullet_step()
        self._make_observation()  # Update state
        self._reach_check()
        reward = self._cal_reward()
        self._last_refresh()

        dw = self.done
        if self.step_count > self.max_episode_steps:
            self.done = True

        self.last_action = self.action
        return self.state, reward, self.done, dw, []

    def _make_observation(self):
        '''
        Make an observation: measure the position and orientation of NAO,
        and get current joint angles from motionProxy.
        '''
        ######### get floating_base pos/quat #########
        self.floating_base_pos = self.floating_base.get_pos()
        self.floating_base_quat = self._ensure_quat_continuity(self.floating_base.get_quat(),
                                                               conto=self.floating_base_quat_)
        self.floating_base_quat_ = self.floating_base_quat
        ######### get agent pos/quat #########
        self.agent_pos = self.agent.get_pos()
        self.agent_quat = self._ensure_quat_continuity(self.agent.get_quat(), conto=self.agent_quat_)
        self.agent_quat_ = self.agent_quat
        ######### get joint pos/vel #########
        self.joint_pos = self.agent.get_joint_pos()
        self.joint_vel = self.agent.get_joint_vel()
        for i in range(len(self.joint_pos)):
            self.joint_pos[i] = self.joint_pos[i] % _2pi
        ######### get target pos/quat #########
        self.target_pos = self.target.get_pos()
        self.target_quat = self._ensure_quat_continuity(self.target.get_quat(), conto=self.target_quat_)
        self.target_quat_ = self.target_quat
        self.target_euler = p.getEulerFromQuaternion(self.target_quat)
        self.target_collision = False
        ######### get end_effector pos/quat #########
        self.end_effector_pos = self.agent.get_end_effector_pos()
        self.end_effector_quat = self._ensure_quat_continuity(self.agent.get_end_effector_quat(),
                                                              conto=self.end_effector_quat_)
        self.end_effector_quat_ = self.end_effector_quat
        self.end_effector_euler = p.getEulerFromQuaternion(self.end_effector_quat)
        self.tar_euler_relative_to_ee = self._e2e(self.end_effector_euler, self.target_euler)

        self._get_state()
        self._save_history_observation_from_make_observation(self.his_obs)

    def _get_state(self):
        if self.pos_zero_origin == 'ee':
            pos_zero_origin = self.init_end_effector_pos
        elif self.pos_zero_origin == 'ab':
            pos_zero_origin = self.init_agent_pos

        end_effector_pos = self.end_effector_pos - pos_zero_origin
        target_pos = self.target_pos - pos_zero_origin
        self.err_pos = target_pos - end_effector_pos
        end_effector_quat = self.end_effector_quat
        target_quat = self.target_quat
        self.err_quat = self._q2q(end_effector_quat, target_quat, self.eet_)

        end_effector_euler = self.end_effector_euler
        target_euler = self.target_euler
        self.err_euler = R.from_quat(self.err_quat).as_euler('xyz')
        self.dis_ori = R.from_quat(self.err_quat).magnitude()

        self.dis_pos = norm(self.err_pos)

        jp = self.joint_pos[:self.action_dim]
        jv = self.joint_vel[:self.action_dim]

        # state:[End{x,y,z}, Tar{x,y,z}, Err{x,y,z}, End{x,y,z,w}, Tar{x,y,z,w}, Err{x,y,z,w},
        # Dis_pos, Dis_ori, Joint_Pos[6], Joint_Vel[6]]
        self.state = np.hstack((end_effector_pos, target_pos, self.err_pos,
                                end_effector_quat, target_quat, self.err_quat,
                                self.dis_pos, self.dis_ori, jp, jv))
        self.eet_ = self.err_quat

    def _history_observation_reset(self, observation):
        ''' FROM _MAKE_OBSERVATION '''
        observation['floating_base_pos'] = []
        observation['floating_base_quat'] = []
        observation['floating_base_quat_'] = []

        observation['agent_pos'] = []
        observation['agent_quat'] = []
        observation['agent_quat_'] = []

        observation['joint_pos'] = []
        observation['joint_vel'] = []

        observation['target_pos'] = []
        observation['target_quat'] = []
        observation['target_quat_'] = []
        observation['target_euler'] = []
        observation['target_collision'] = []

        observation['end_effector_pos'] = []
        observation['end_effector_quat'] = []
        observation['end_effector_quat_'] = []
        observation['end_effector_euler'] = []
        observation['tar_euler_relative_to_ee'] = []

        ''' FROM _GET_STATE '''
        observation['err_pos'] = []
        observation['err_quat'] = []
        observation['err_euler'] = []

        observation['dis_pos'] = []
        observation['dis_ori'] = []

        observation['state'] = []
        observation['eet_'] = []

        ''' FROM STEP '''
        observation['action'] = []

    def _save_history_observation_from_make_observation(self, observation):
        ''' FROM _MAKE_OBSERVATION '''
        observation['floating_base_pos'].append(self.floating_base_pos)
        observation['floating_base_quat'].append(self.floating_base_quat)
        observation['floating_base_quat_'].append(self.floating_base_quat_)

        observation['agent_pos'].append(self.agent_pos)
        observation['agent_quat'].append(self.agent_quat)
        observation['agent_quat_'].append(self.agent_quat_)

        observation['joint_pos'].append(self.joint_pos)
        observation['joint_vel'].append(self.joint_vel)

        observation['target_pos'].append(self.target_pos)
        observation['target_quat'].append(self.target_quat)
        observation['target_quat_'].append(self.target_quat_)
        observation['target_euler'].append(self.target_euler)
        observation['target_collision'].append(self.target_collision)

        observation['end_effector_pos'].append(self.end_effector_pos)
        observation['end_effector_quat'].append(self.end_effector_quat)
        observation['end_effector_quat_'].append(self.end_effector_quat_)
        observation['end_effector_euler'].append(self.end_effector_euler)
        observation['tar_euler_relative_to_ee'].append(self.tar_euler_relative_to_ee)

        ''' FROM _GET_STATE '''
        observation['err_pos'].append(self.err_pos)
        observation['err_quat'].append(self.err_quat)
        observation['err_euler'].append(self.err_euler)

        observation['dis_pos'].append(self.dis_pos)
        observation['dis_ori'].append(self.dis_ori)

        observation['state'].append(self.state)
        observation['eet_'].append(self.eet_)

    def _make_action(self, action):
        '''
        Perform an action - move each joint by a specific amount
        '''
        # Update velocities
        self.agent.make_action(action)

        self.his_obs['action'].append(action)

    def _cal_reward(self):
        pos_err_penalty, ori_err_penalty, smooth_penalty = 0, 0, 0
        ori_decrease_reward, done_reward = 0, 0
        reward = 0

        pos_err = -self.dis_pos
        self.pos_err_queue.append(pos_err)
        while len(self.pos_err_queue) > self.pos_err_stable_times:
            del (self.pos_err_queue[0])

        ori_err = -self.dis_ori
        self.ori_err_queue.append(ori_err)
        while len(self.ori_err_queue) > self.ori_err_stable_times:
            del (self.ori_err_queue[0])

        pos_err_penalty = pos_err
        reward += pos_err_penalty * (1 - self.ori_penalty_rate)

        ori_err_penalty = ori_err * 0.25
        reward += ori_err_penalty * self.ori_penalty_rate

        # reward decrease
        if self.last_state is not None:
            ori_decrease_reward += self.last_dis_ori - self.dis_ori
            ori_decrease_reward *= 0.1
            reward += ori_decrease_reward

        # smooth penalty
        for i in range(len(self.action)):
            target_current_delta = np.abs(self.action[i] - self.joint_vel[i])
            if target_current_delta > 0.5:  # 速度变化过大
                smooth_penalty -= 0.15 * (target_current_delta - 0.5)
        reward += smooth_penalty

        self.last_target_collision = self.target_collision

        # reward for task completion
        if self.pos_reach_list[-1]:
            done_reward += (self.pos_err_thres - abs(pos_err)) / self.pos_err_thres
        if self.ori_reach_list[-1]:
            done_reward += (self.ori_err_thres - abs(ori_err)) / self.ori_err_thres
        reward += done_reward * 0.1

        return reward

    def _last_refresh(self):
        self.last_state = self.state
        self.last_err_pos, self.last_err_quat, self.last_err_euler = self.err_pos, self.err_quat, self.err_euler
        self.last_dis_pos, self.last_dis_ori = self.dis_pos, self.dis_ori

    def _target_reinit(self):
        y_ = self.mug_start_pos + np.random.randn(3) * 0.05
        # self.start_target_quat = self._ensure_quat_continuity(
        #     self._target_quaternion_reinit(), conto=self.start_target_quat_)
        self.start_target_quat_ = self.mug_start_ori

        self.start_target_pos = y_

    @staticmethod
    def _target_quaternion_reinit():
        return R.random().as_quat()

    def _ensure_quat_continuity(self, quaternion, conto=None):
        '''
        确保四元数的方向一致性。
        :param quaternion: 四元数。
        :return: 调整后的四元数序列。
        '''
        # if conto is None:
        #     conto_array = np.array([0.5, 0.5, 0.5, 0.5])
        # else:
        #     conto_array = conto
        conto_array = np.array([0.5, 0.5, 0.5, 0.5]) if conto is None else conto
        if np.dot(quaternion, conto_array) < 0:
            return -quaternion
        return quaternion

    def _q2q(self, q1, q2, conto=None):
        return self._ensure_quat_continuity((R.from_quat(q1).inv() * R.from_quat(q2)).as_quat(), conto)

    def _e2e(self, e1, e2):
        return (R.from_euler('xyz', e1).inv() * R.from_euler('xyz', e2)).as_euler('xyz')

    def set_episode_num(self, episode_num):
        self.episode_num = episode_num

    def set_maintain_save_dir(self, maintain_save_dir):
        self.maintain_save_dir = maintain_save_dir

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def pybullet_step(self, times=1):
        for i in range(times):
            p.stepSimulation()
            time.sleep(1. / 240000.)
