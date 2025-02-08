import os
import time
import numpy as np
from numpy import pi
import cv2
from scipy.spatial.transform import Rotation as R

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

class PybulletSpaceManipulatorReacher:
    def __init__(self, args, headless=None, plat='wsl'):
        super(PybulletSpaceManipulatorReacher, self).__init__()
        ''' DATA FROM MAIN CODE '''
        self.plat = plat
        if self.plat in ['wsl', 'windows']:
            headless = args.headless if headless is None else headless
        else:
            headless = True

        self.action_dim = args.action_dim
        self.time_interval = args.time_interval

        if headless:
            physicsClient = p.connect(p.DIRECT)
        else:
            physicsClient = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # 关闭整个调试GUI，包括网格

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)   # 设置零重力

        self.ur5_start_pos = [0, -0.4, 0.607]
        self.ur5_start_ori = p.getQuaternionFromEuler([0, 0, 0])
        self.ur5_reset_joint_pos = [pi/2, -pi/2, pi/2, -pi/2, 0, 0, ]
        self.ur5_reset_joint_vel = [0, 0, 0, 0, 0, 0, ]
        self.ur5_force_limit = [150, 150, 150, 28, 28, 28]

        self.satellite_start_pos = [0, 0, 0]
        self.satellite_start_ori = p.getQuaternionFromEuler([0, 0, 0])
        self.mug_start_pos = [-0.031, -0.944, 1.114]
        self.mug_start_ori = p.getQuaternionFromEuler([0, 0, 0])

        # 加载 航天器 / 机械臂 / 目标 模型
        self.dummyShape = p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=(1, 1, 1, 1))
        self.dummyId = p.createMultiBody(baseVisualShapeIndex=self.dummyShape, baseCollisionShapeIndex=-1)
        p.resetBasePositionAndOrientation(self.dummyId, np.array([1, 1, 1]), np.array([0, 0, 0, 1]))
        self.agent = UR5_Agent(self.ur5_start_pos, self.ur5_start_ori, self.ur5_reset_joint_pos, self.ur5_reset_joint_vel,
                 self.action_dim, 'velocity', self.ur5_force_limit, 'urdf/ur5.urdf')
        self.floating_base = SpaceObject(self.satellite_start_pos, self.satellite_start_ori, 'urdf/satellite.urdf')
        self.target = SpaceObject(self.mug_start_pos, self.mug_start_ori, 'urdf/mug.urdf')

        # 设置无阻尼和无摩擦力
        p.changeDynamics(self.agent.id, -1, linearDamping=0, angularDamping=0, lateralFriction=0, spinningFriction=0, rollingFriction=0)
        p.changeDynamics(self.floating_base.id, -1, linearDamping=0, angularDamping=0, lateralFriction=0, spinningFriction=0, rollingFriction=0)
        self.constraint = p.createConstraint(
            parentBodyUniqueId=self.floating_base.id,
            parentLinkIndex=-1,  # 连接根链接
            childBodyUniqueId=self.agent.id,
            childLinkIndex=-1,   # 连接根链接
            jointType=p.JOINT_FIXED,  # 旋转关节
            jointAxis=[0, 0, 0],  # 关节绕 Z 轴旋转
            parentFramePosition=self.ur5_start_pos,  # 关节在 cubeA 上的位置
            childFramePosition=[0, 0, 0]  # 关节在 cubeB 上的位置
        )

        p.setTimeStep(self.time_interval / 1000.)    # 100 ms

        # region
        self.max_episode_steps = args.max_episode_steps

        self.is_train = args.is_train
        self.use_ensure_quat_continuity = args.use_ensure_quat_continuity

        target_pos_reinit_methods = ['random', 'sequence', 'distribution']
        self.target_pos_reinit_method = args.target_pos_reinit_method
        assert (self.target_pos_reinit_method in target_pos_reinit_methods)

        # M, N, Q: split r/theta/fai into N pieces
        MNQ = args.MNQ.split()
        self.M, self.N, self.Q = int(MNQ[0]), int(MNQ[1]), int(MNQ[2])
        self.r_max, self.r_min = args.r_max, args.r_min

        self.target_reinit_pos_freq = args.target_reinit_pos_freq

        self.episode_num = 0
        self.action = np.zeros(self.action_dim)
        self.last_action = self.action

        self.success_interval = args.success_interval

        self.pos_err_queue          = []
        self.ori_err_queue          = []
        self.pos_err_stable_times   = args.pos_err_stable_times
        self.pos_err_thres          = args.pos_err_thres
        self.arm_angle_0_to_2pi     = args.arm_angle_0_to_2pi
        self.success_sample_rate    = args.success_sample_rate
        self.success_min_rate       = args.success_min_rate

        self.ori_err_stable_times   = args.pos_err_stable_times
        self.ori_err_thres          = args.ori_err_thres
        self.ori_inherent_rate      = args.ori_inherent_rate
        self.ori_penalty_rate       = args.ori_penalty_rate

        self.pos_related_to_finish  = args.pos_related_to_finish
        self.ori_related_to_finish  = args.ori_related_to_finish

        self.fail_prob_decrease             = args.fail_prob_decrease
        self.decrease_threshold             = args.decrease_threshold
        self.success_sample_rate_idx        = args.success_sample_rate_idx
        self.pos_err_stable_times_increase  = args.pos_err_stable_times_increase
        self.pos_err_thres_idx              = args.pos_err_thres_idx
        self.ori_err_thres_idx              = args.ori_err_thres_idx


        self._sphere_reach_fail_prob_reset()
        self.eval_episodes          = self.M * self.N * self.Q
        self.norm_reach_fail_prob   = (self.sphere_reach_fail_prob / np.sum(self.sphere_reach_fail_prob)).reshape(-1)
        self.norm_reach_fail_prob[-1] += 1

        ''' position error penalty define '''
        self.use_pos_err_penalty    = args.use_pos_err_penalty
        pos_err_types               = ['linear', 'sum']
        self.pos_err_type           = args.pos_err_type
        assert (self.pos_err_type in pos_err_types)

        ''' orientation error penalty define '''
        self.use_ori_err_penalty    = args.use_ori_err_penalty
        ori_err_types               = ['dangle', 'minusdot']
        self.ori_err_type           = args.ori_err_type
        assert (self.ori_err_type in ori_err_types)

        ''' orientation decrease reward define '''
        self.use_ori_decrease_reward    = args.use_ori_decrease_reward
        self.ori_decrease_reward_rate   = args.ori_decrease_reward_rate

        ''' other penalty define '''
        self.use_smooth_penalty     = args.use_smooth_penalty

        ''' done reward define '''
        self.use_done_reward    = args.use_done_reward
        self.done_reward_rate   = args.done_reward_rate

        ''' position zero origin define '''
        pos_zero_origins        = ['ee', 'ab']
        self.pos_zero_origin    = args.pos_zero_origin
        assert (self.pos_zero_origin in pos_zero_origins)

        ''' state space define '''
        state_dim_chooses       = ['eterr3p4qdpopv', 'eterr3p4qpv', 'err3p4qdpopv']
        self.state_dim_choose   = args.state_dim_choose
        assert (self.state_dim_choose in state_dim_chooses)
        # endregion

        self.end_effector_axis_list = []

        # env reset
        self.floating_base.reset(self.satellite_start_pos, self.satellite_start_ori)
        self.agent.reset()
        self.target.reset(self.mug_start_pos, self.mug_start_ori)

        ######### get floating_base pos/quat #########
        self.init_floating_base_pos     = self.floating_base.get_pos()
        self.init_floating_base_quat    = self._ensure_quat_continuity(self.floating_base.get_quat())
        self.init_floating_base_quat_   = self.init_floating_base_quat
        self.floating_base_pos          = self.init_floating_base_pos
        self.floating_base_quat         = self.init_floating_base_quat
        self.floating_base_quat_        = self.init_floating_base_quat_
        ######### get agent pos/quat #########
        self.init_agent_pos     = self.agent.get_pos()
        self.init_agent_quat    = self._ensure_quat_continuity(self.agent.get_quat())
        self.init_agent_quat_   = self.init_agent_quat
        self.agent_pos          = self.init_agent_pos
        self.agent_quat         = self.init_agent_quat
        self.agent_quat_        = self.init_agent_quat_
        ######### get joint pos/vel #########
        self.init_joint_pos = self.agent.get_joint_pos()
        self.init_joint_vel = self.agent.get_joint_vel()
        if self.arm_angle_0_to_2pi:
            for i in range(self.action_dim):
                self.init_joint_pos[i] = self.init_joint_pos[i] % _2pi
        self.joint_pos = copy.copy(self.init_joint_pos)
        self.joint_vel = copy.copy(self.init_joint_vel)
        ######### get target pos/quat #########
        self.init_target_pos    = self.target.get_pos()
        self.init_target_quat   = self._ensure_quat_continuity(self.target.get_quat())
        self.init_target_quat_  = self.init_target_quat
        self.init_target_euler  = p.getEulerFromQuaternion(self.init_target_quat)
        self.start_target_pos   = self.init_target_pos
        self.start_target_quat  = self.init_target_quat
        self.start_target_quat_ = self.init_target_quat_
        self.start_target_euler = self.init_target_euler
        self.target_pos         = self.init_target_pos
        self.target_quat        = self.init_target_quat
        self.target_quat_       = self.init_target_quat_
        self.target_euler       = self.init_target_euler
        ######### get end_effector pos/quat #########
        self.init_end_effector_pos      = self.agent.get_end_effector_pos()
        self.init_end_effector_quat     = self._ensure_quat_continuity(self.agent.get_quat())
        self.init_end_effector_quat_    = self.init_end_effector_quat
        self.init_end_effector_euler    = p.getEulerFromQuaternion(self.init_end_effector_quat)
        self.end_effector_pos           = self.init_end_effector_pos
        self.end_effector_quat          = self.init_end_effector_quat
        self.end_effector_quat_         = self.init_end_effector_quat_
        self.end_effector_euler         = self.init_end_effector_euler
        self.tar_euler_relative_to_ee   = self._e2e(self.end_effector_euler, self.target_euler)

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

        target_pose_init_finish, target_reinit_time, target_use_random_init = False, 0, False
        floating_base_pose_init_finish = False
        agent_init_finish, joint_target_vel_init_finish = False, False
        self.r_, self.theta_, self.fai_ = -1, -1, -1
        reset_finish = False
        self.pos_reach_list, self.ori_reach_list = [], []
        self.done = False

        while not reset_finish:
            target_init_times, floating_base_init_times, agent_init_times = 0, 0, 0
            if self.target_reinit_pos_freq != 0 and self.episode_num % self.target_reinit_pos_freq == 0:
                if self.target_pos_reinit_method == 'distribution':
                    self._reach_fail_prob_maintain()

                self._target_reinit(pos_random_reinit=target_use_random_init)
                target_reinit_time += 1

            while True:
                self._set_floating_base()
                self._set_agent()

                self.target_collision = True
                if pos is not None and ori is not None:
                    self.start_target_pos, self.start_target_quat = pos, ori
                    self.target.reset(self.start_target_pos, self.start_target_quat)
                    self.target_collision = False
                else:
                    while self.target_collision:
                        self._set_target()

                        self.target_collision = False
                        if self.target_collision:
                            self._target_reinit(pos_random_reinit=target_use_random_init)
                            target_reinit_time += 1
                            if target_reinit_time >= 10:
                                target_use_random_init = True
                                print('This position maybe unavailable, reinit position randomly!')
                            continue
                self.last_target_collision = self.target_collision

                for _ in range(5):
                    self.pybullet_step(2)
                    target_pose_init_finish = self._check_target_init()
                    floating_base_pose_init_finish = self._check_floating_base_init()
                    agent_init_finish = self._check_agent_init()
                    if target_pose_init_finish and floating_base_pose_init_finish and agent_init_finish:
                        break

                if target_pose_init_finish and floating_base_pose_init_finish and agent_init_finish:
                    break
                else:
                    if not target_pose_init_finish:
                        target_init_times += 1
                        if target_init_times > 10:
                            print('\033[31mTarget pose init may failure!\033[0m')
                            self._target_reinit(pos_random_reinit=target_use_random_init)
                    if not floating_base_pose_init_finish:
                        floating_base_init_times += 1
                        if floating_base_init_times > 10:
                            print('Floating Base pose init may failure!')
                    if not agent_init_finish:
                        agent_init_times += 1
                        if agent_init_times > 10:
                            print('Agent init may failure!')

            reset_finish = self._reset_finish()

        self.init_end_effector_pos = self.agent.get_end_effector_pos()
        self.init_agent_pos = self.agent.get_pos()

        self._make_observation()  # Update state
        self._last_refresh()
        p.resetBasePositionAndOrientation(self.dummyId, self.target_pos, self.target_quat)

        return self.state

    def _set_target(self):
        self.start_target_pos += self.ur5_start_pos
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
        if self.pos_related_to_finish:
            self.pos_reach_list.append(self.dis_pos < self.pos_err_thres)
        else:
            self.pos_reach_list.append(True)    # Default reach

        if self.ori_related_to_finish:
            self.ori_reach_list.append(self.dis_ori < self.ori_err_thres)
        else:
            self.ori_reach_list.append(True)    # Default reach

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

        if self.done and self.target_pos_reinit_method == 'distribution' \
                and len(self.pos_err_queue) >= self.pos_err_stable_times:
            tmp = self.sphere_reach_fail_prob[self.r_, self.theta_, self.fai_] * self.success_sample_rate
            self.sphere_reach_fail_prob[self.r_, self.theta_, self.fai_] = max(tmp, self.success_min_rate)
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
        self.floating_base_pos   = self.floating_base.get_pos()
        self.floating_base_quat  = self._ensure_quat_continuity(self.floating_base.get_quat(), conto=self.floating_base_quat_)
        self.floating_base_quat_ = self.floating_base_quat
        ######### get agent pos/quat #########
        self.agent_pos   = self.agent.get_pos()
        self.agent_quat  = self._ensure_quat_continuity(self.agent.get_quat(), conto=self.agent_quat_)
        self.agent_quat_ = self.agent_quat
        ######### get joint pos/vel #########
        self.joint_pos = self.agent.get_joint_pos()
        self.joint_vel = self.agent.get_joint_vel()
        if self.arm_angle_0_to_2pi:
            for i in range(len(self.joint_pos)):
                self.joint_pos[i] = self.joint_pos[i] % _2pi
        ######### get target pos/quat #########
        self.target_pos       = self.target.get_pos()
        self.target_quat      = self._ensure_quat_continuity(self.target.get_quat(), conto=self.target_quat_)
        self.target_quat_     = self.target_quat
        self.target_euler     = p.getEulerFromQuaternion(self.target_quat)
        self.target_collision = False
        ######### get end_effector pos/quat #########
        self.end_effector_pos         = self.agent.get_end_effector_pos()
        self.end_effector_quat        = self._ensure_quat_continuity(self.agent.get_end_effector_quat(), conto=self.end_effector_quat_)
        self.end_effector_quat_       = self.end_effector_quat
        self.end_effector_euler       = p.getEulerFromQuaternion(self.end_effector_quat)
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

        if self.state_dim_choose == 'eterr3p4qdpopv':
            # state:[End{x,y,z}, Tar{x,y,z}, Err{x,y,z}, End{x,y,z,w}, Tar{x,y,z,w}, Err{x,y,z,w},
            # Dis_pos, Dis_ori, Joint_Pos[6], Joint_Vel[6]]
            self.state = np.hstack((end_effector_pos, target_pos, self.err_pos,
                                    end_effector_quat, target_quat, self.err_quat,
                                    self.dis_pos, self.dis_ori, jp, jv))
            self.eet_ = self.err_quat
        elif self.state_dim_choose == 'eterr3p4qpv':
            # state:[End{x,y,z}, Tar{x,y,z}, Err{x,y,z}, End{x,y,z,w}, Tar{x,y,z,w}, Err{x,y,z,w},
            # Joint_Pos[6], Joint_Vel[6]]
            self.state = np.hstack((end_effector_pos, target_pos, self.err_pos,
                                    end_effector_quat, target_quat, self.err_quat,
                                    jp, jv))
        elif self.state_dim_choose == 'err3p4qdpopv':
            # state:[Err{x,y,z}, Err{x,y,z,w}, Dis_pos, Dis_ori, Joint_Pos[6], Joint_Vel[6]]
            self.state = np.hstack((self.err_pos, self.err_quat, self.dis_pos, self.dis_ori, jp, jv))
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

        if self.use_pos_err_penalty:
            if self.pos_err_type == 'linear':
                pos_err_penalty = pos_err
            elif self.pos_err_type == 'sum':
                pos_err_penalty = -np.sum(np.abs(self.err_pos))

            reward += pos_err_penalty * (1 - self.ori_penalty_rate)

        if self.use_ori_err_penalty:
            if self.ori_err_type == 'dangle':
                ori_err_penalty = ori_err * self.ori_inherent_rate
            elif self.ori_err_type == 'minusdot':
                ori_err_penalty = -(1 - (np.dot(self.end_effector_quat, self.target_quat) ** 2))

            reward += ori_err_penalty * self.ori_penalty_rate

        if self.use_ori_decrease_reward:
            if self.last_state is not None:
                ori_decrease_reward += self.last_dis_ori - self.dis_ori
                ori_decrease_reward *= self.ori_decrease_reward_rate
                reward += ori_decrease_reward

        if self.use_smooth_penalty:
            for i in range(len(self.action)):
                target_current_delta = np.abs(self.action[i] - self.joint_vel[i])
                if target_current_delta > 0.5:  # 速度变化过大
                    smooth_penalty -= 0.15 * (target_current_delta - 0.5)
            reward += smooth_penalty

        self.last_target_collision = self.target_collision

        if self.use_done_reward:
            if self.pos_reach_list[-1]:
                done_reward += (self.pos_err_thres - abs(pos_err)) / self.pos_err_thres
            if self.ori_reach_list[-1]:
                done_reward += (self.ori_err_thres - abs(ori_err)) / self.ori_err_thres
            reward += done_reward * self.done_reward_rate

        return reward

    def _last_refresh(self):
        self.last_state = self.state
        self.last_err_pos, self.last_err_quat, self.last_err_euler = self.err_pos, self.err_quat, self.err_euler
        self.last_dis_pos, self.last_dis_ori = self.dis_pos, self.dis_ori

    def _target_reinit(self, pos_random_reinit=False):
        if not pos_random_reinit:
            if self.target_pos_reinit_method == 'random':
                y_ = self._target_sphere_random_reinit()
            elif self.target_pos_reinit_method == 'sequence':
                y_ = self._target_sphere_sequence_reinit()
            elif self.target_pos_reinit_method == 'distribution':
                y_ = self._target_sphere_dist_reinit()
        else:
            y_ = self._target_sphere_random_reinit()

        self.start_target_quat = self._ensure_quat_continuity(self._target_quaternion_reinit(), conto=self.start_target_quat_)
        self.start_target_quat_ = self.start_target_quat

        self.start_target_pos = y_

    def _target_sphere_random_reinit(self):
        while 1:
            # r_max, r_min = 0.7, 0.2
            # r: 0.2 ~ 0.7; theta: 0 ~ 90, fai: 0 ~ 360
            r = np.random.uniform(self.r_min, self.r_max)
            theta = np.random.uniform(0, pi / 2)
            fai = np.random.uniform(0, _2pi)
            y_ = np.array([r * np.sin(theta) * np.cos(fai), r * np.sin(theta) * np.sin(fai), r * np.cos(theta) + 0.05])
            # 这里 +0.05 是为了保证 target 不会与 agent 相重合，避免碰撞
            if self.r_min < norm(y_) < (self.r_max + 0.05):
                break
        self.r_ = int(((r - self.r_min) / (self.r_max - self.r_min)) * self.M)
        self.theta_ = int((theta / (pi/2)) * self.N)
        self.fai_ = int((fai / _2pi) * self.Q)
        return y_

    def _target_sphere_sequence_reinit(self):
        # r_max, r_min = 0.7, 0.2
        # 将整个球坐标系分为 (M*N*Q) 份
        # r: 0.2 ~ 0.7; theta: 0 ~ 90, fai: 0 ~ 360
        while 1:
            d_r, d_theta, d_fai = (self.r_max - self.r_min) / self.M, pi / 2 / self.N, _2pi / self.Q
            self.r_ = int(self.episode_num % self.eval_episodes / self.N / self.Q)
            self.theta_ = int(self.episode_num % self.eval_episodes / self.Q % self.N)
            self.fai_ = int(self.episode_num % self.eval_episodes % self.Q)
            r = (self.r_) * d_r + self.r_min  # 生成在一定范围内的距离 r
            theta = (self.theta_) * d_theta
            fai = (self.fai_) * d_fai
            y_ = np.array([r * np.sin(theta) * np.cos(fai), r * np.sin(theta) * np.sin(fai), r * np.cos(theta) + 0.05])
            if self.r_min < norm(y_) < (self.r_max + 0.05):
                break

        return y_

    def _target_sphere_dist_reinit(self):
        # r_min, r_max = 0.7, 0.2
        # 将整个球坐标系分为 (M*N*Q) 份
        times = 0
        while 1:
            d_r, d_theta, d_fai = (self.r_max - self.r_min) / self.M, pi / 2 / self.N, _2pi / self.Q
            tmp = np.random.rand()
            for i in range(len(self.norm_reach_fail_prob)):
                tmp -= self.norm_reach_fail_prob[i]
                if tmp < 0:
                    self.r_ = int(i / self.N / self.Q)
                    self.theta_ = int(i / self.Q % self.N)
                    self.fai_ = int(i % self.Q)
                    break
            r1, r2, r3 = np.random.random(3)
            r = (r1 + self.r_) * d_r + self.r_min  # 生成在一定范围内的距离 r
            theta = (r2 + self.theta_) * d_theta
            fai = (r3 + self.fai_) * d_fai
            y_ = np.array([r * np.sin(theta) * np.cos(fai), r * np.sin(theta) * np.sin(fai), r * np.cos(theta) + 0.05])
            if self.r_min < norm(y_) < (self.r_max + 0.05):
                break
            else:
                times += 1

        return y_

    def _target_quaternion_reinit(self):
        return R.random().as_quat()

    def _reach_fail_prob_maintain(self):
        if self.fail_prob_decrease and self.sphere_reach_fail_prob.max() < self.decrease_threshold:
            self._sphere_reach_fail_prob_reset()
            self.pos_err_stable_times += self.pos_err_stable_times_increase
            # self.success_interval += self.pos_err_stable_times_increase
            self.pos_err_thres *= self.pos_err_thres_idx
            self.ori_err_thres *= self.ori_err_thres_idx
            self.success_sample_rate *= self.success_sample_rate_idx
            with open(os.path.join(self.maintain_save_dir, 'reach_fail_prob_maintain.txt'.format(self.episode_num)), 'a+') as f:
                f.write('In episode {}, total step = {}:\n'.format(
                    self.episode_num, (self.episode_num * (self.max_episode_steps+1) + self.step_count)))
                f.write('pos_err_thres = {}, ori_err_thres = {};\n'.format(
                    self.pos_err_thres, self.ori_err_thres))
                f.write('error_stable_times = {}, ori_penalty_rate = {};\n\n'.format(
                    self.pos_err_stable_times, self.ori_penalty_rate))
        self.norm_reach_fail_prob = (self.sphere_reach_fail_prob / np.sum(self.sphere_reach_fail_prob)).reshape(-1)
        self.norm_reach_fail_prob[-1] += 1

    def _sphere_reach_fail_prob_reset(self):
        self.sphere_reach_fail_prob = np.ones([self.M, self.N, self.Q])
        self.sphere_reach_fail_prob[2, 0, 2:7] = self.decrease_threshold * self.success_sample_rate * 0.1
        self.sphere_reach_fail_prob[0:2, 1, 4:6] = self.decrease_threshold * self.success_sample_rate * 0.1

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
        if self.use_ensure_quat_continuity and np.dot(quaternion, conto_array) < 0:
            return -quaternion
        return quaternion

    def _q2q(self, q1, q2, conto=None):
        return self._ensure_quat_continuity((R.from_quat(q1).inv()*R.from_quat(q2)).as_quat(), conto)

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
