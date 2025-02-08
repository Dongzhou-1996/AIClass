import torch
import numpy as np
np.set_printoptions(precision=3, suppress=True)

from torch.utils.tensorboard import SummaryWriter
import gym
import argparse
from Utils.normalization import Normalization, RewardScaling
from Utils.replaybuffer import Off_Policy_ReplayBuffer
from Utils.utils import *
from Agents.DDPG import DDPG
import os

import sys
sys.path.append('..')

import time

def main(args, env):
    set_seed(args.seed, env)

    args.state_dim = env.state.shape[0]

    print('state_dim={}'.format(args.state_dim))
    print('action_dim={}'.format(args.action_dim))
    print('max_episode_steps={}'.format(args.max_episode_steps))
    print('mode:', 'train' if args.is_train else 'evaluation')

    dirs = create_saving_dirs(args)
    print('Create directory successful!\nROOT FOLDER: >>>> {} <<<<'.format(args.save_dir))

    reinit_freq = args.target_reinit_pos_freq
    reinit_flag = True if (reinit_freq > 0) else False

    model_path = None
    stable_step_num = args.stable_step_num

    agent = DDPG(args)
    # print(agent.actor)

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    mean_rewards_list, stable_rewards_list, final_rewards_list  = [], [], []
    mean_errors_list, stable_errors_list, final_errors_list = [], [], []
    actor_loss_list, critic_loss_list, actor_lr_list, critic_lr_list = [], [], [], []

    if args.is_train:    # train
        total_steps = 0  # Record the total steps during the training
        episode_num = 0
        update_time = 0
        noise_std = args.noise_std_index * args.joint_max_vel

        replay_buffer = Off_Policy_ReplayBuffer(args)

        if args.restore:
            if args.model_name is not None:
                model_path = os.path.join(dirs['log'], args.model_name)
                if os.path.exists(model_path):
                    print('Reload model from', model_path)
                else:
                    print('Cannot find model, starting training!')
            else:
                model_path = None
                if check_files_with_suffix(dirs['log'], '.pth'):
                    model_path = get_latest_model_path(dirs['log'])
                    print('Reload latest model from', model_path)

            if model_path is not None:
                root_data_path = dirs['increment']
                data_path_exist = True
                if not os.path.exists(root_data_path):
                    print('Cannot find data!')
                    data_path_exist = False

                # reload model
                episode_num_, total_steps_, state_norm_, reward_scaling_, reach_fail_prob_, \
                    update_time_ = agent.reload_model(model_path)

                # reload data increment
                if data_path_exist:
                    stable_rewards_list_, stable_errors_list_, \
                        actor_loss_list_, critic_loss_list_, \
                        actor_lr_list_, critic_lr_list_, _, _ = reload_data_stable_increment(root_data_path)

                episode_num = episode_num_
                env.set_episode_num(episode_num)
                # PyTorch 会对保存的 list 序列化为 tuple, 这里是反序列化操作
                if data_path_exist:
                    if isinstance(stable_rewards_list_, tuple):
                        stable_rewards_list, stable_errors_list, actor_loss_list, critic_loss_list, \
                            actor_lr_list, critic_lr_list = stable_rewards_list_[0][:episode_num_ - 1], \
                                                            stable_errors_list_[0][:episode_num_ - 1], \
                                                            actor_loss_list_[0], critic_loss_list_[0], \
                                                            actor_lr_list_[0], critic_lr_list_[0]
                    elif isinstance(stable_rewards_list_, list):
                        stable_rewards_list, stable_errors_list, actor_loss_list, critic_loss_list, \
                            actor_lr_list, critic_lr_list = stable_rewards_list_[:episode_num_ - 1], \
                                                            stable_errors_list_[:episode_num_ - 1], \
                                                            actor_loss_list_, critic_loss_list_, \
                                                            actor_lr_list_, critic_lr_list_
                del (stable_rewards_list_, stable_errors_list_, actor_loss_list_, critic_loss_list_, actor_lr_list_, critic_lr_list_, )
                total_steps, state_norm, reward_scaling, env.sphere_reach_fail_prob, update_time = total_steps_, \
                            state_norm_, reward_scaling_, reach_fail_prob_, update_time_

        total_steps_last = total_steps
        data_length = len(stable_rewards_list)
        loss_length = len(actor_loss_list)

        # time.sleep(2)
        env.set_maintain_save_dir(dirs['save'])
        update_flag = False

        t_last = time.time()
        while total_steps < args.max_train_steps:
            episode_steps = 0
            episode_rewards_list, episode_errors_list = [], []
            s = env.reset()

            episode_errors_list = get_episode_error(episode_errors_list, env, episode_steps)

            if args.use_state_norm:
                s = state_norm(s)
            if args.use_reward_scaling:
                reward_scaling.reset()

            done, success_flag, tracking_success_times = False, False, 0
            episode_num += 1
            while not done:
                episode_steps += 1
                if total_steps < args.random_steps:
                    a = (np.random.rand(args.action_dim)-0.5) * 2 * args.joint_max_vel
                else:
                    a = agent.choose_action(s)
                    a = (a + np.random.normal(0, noise_std, size=args.action_dim)).clip(-args.joint_max_vel, args.joint_max_vel)

                s_, reward, done, dw, img = env.step(a)

                episode_errors_list = get_episode_error(episode_errors_list, env, episode_steps)
                episode_rewards_list.append(reward)

                s_ = state_norm(s_) if args.use_state_norm else s_
                r = reward_scaling(reward) if args.use_reward_scaling else reward

                if done and not args.finish_after_reach and episode_steps <= args.max_episode_steps:
                    success_flag = True

                # Take the 'action'，but store the original 'a'（especially for Beta）
                replay_buffer.store(s, a, r, s_, dw)
                s = s_
                total_steps += 1

                # Take 50 steps, then update the networks 50 times
                if total_steps >= args.random_steps and total_steps % args.update_freq == 0:
                    update_time += 1
                    mean_actor_loss, mean_critic_loss = 0, 0
                    for _ in range(args.update_freq):
                        agent.update(replay_buffer, total_steps, update_time)
                        mean_actor_loss += agent.actor_loss_save
                        mean_critic_loss += agent.critic_loss_save
                    actor_loss_list.append(mean_actor_loss)
                    critic_loss_list.append(mean_critic_loss)
                    actor_lr_list.append(agent.optimizer_actor.param_groups[0]['lr'])
                    critic_lr_list.append(agent.optimizer_critic.param_groups[0]['lr'])

                if done and success_flag:
                    success_flag, done = False, False
                    del (env.pos_err_queue[:env.success_interval], env.ori_err_queue[:env.success_interval])
                    tracking_success_times += 1

            # When the number of transitions in buffer reaches batch_size, then update
            if total_steps % args.save_batch_size == 0:
                reward_scaling = DDPG_train_post_process(args, episode_num, dirs, env, agent, total_steps,
                    state_norm, reward_scaling, model_path, stable_rewards_list,
                    stable_errors_list, actor_loss_list, critic_loss_list, actor_lr_list, critic_lr_list,
                    data_length, loss_length)
                data_length = len(stable_rewards_list)
                loss_length = len(actor_loss_list)

            total_steps_last, t_last = train_post_process(
                               episode_num, episode_steps, tracking_success_times, env, episode_errors_list,
                               stable_rewards_list, episode_rewards_list, stable_step_num, stable_errors_list,
                               args, total_steps, total_steps_last, t_last, dirs)

    else:   # evaluation
        eval_episodes = args.eval_episodes
        success_list = np.zeros(args.eval_episodes)

        while check_files_with_suffix(dirs['log'], '.pth'):
            model_path = None
            if model_path is None:
                model_path = get_latest_model_path(dirs['log'], model_name=args.model_name)
            if os.path.exists(model_path):
                data_path = os.path.join(dirs['log'], 'data.pth')
                if not os.path.exists(data_path):
                    print('Cannot find data!')
                    # exit(-1)
                _, _, state_norm, _, reach_fail_prob, _ = agent.reload_model(model_path)

                env.sphere_reach_fail_prob = reach_fail_prob
                print('Reload model from', model_path)
            else:
                raise ValueError('No model file is found in {}'.format(model_path))

            # time.sleep(2)

            total_rewards_list = []

            for episode_num in range(0, eval_episodes):
                step_jv_list = []
                step_jp_list = []
                step_error_list = []
                step_base_pos_list = []
                step_base_ori_list = []
                image_list = []

                print('\nepisode_number={}'.format(episode_num+1))
                episode_steps = 0
                episode_rewards_list = []
                episode_errors_list = []

                s = env.reset()

                episode_errors_list = get_episode_error(episode_errors_list, env, episode_steps)
                step_error_list = get_episode_error(step_error_list, env, episode_steps)

                step_base_pos_first = env.floating_base.get_pos()
                step_base_ori_first = env.floating_base.get_ori()
                step_base_pos_list.append((env.floating_base.get_pos() - step_base_pos_first).tolist())
                step_base_ori_list.append((env.floating_base.get_ori() - step_base_ori_first).tolist())

                if args.use_state_norm:
                    s = state_norm(s, update=False)

                done, success_flag = False, False
                while not done:
                    episode_steps += 1

                    a = agent.choose_action(s)  # Action and the corresponding log probability
                    s_, reward, done, dw, img = env.step(a)

                    step_error_list = get_episode_error(step_error_list, env, episode_steps)
                    step_jp_list, step_jv_list = get_joint_pos_vel(step_jp_list, step_jv_list, env)
                    step_base_pos_list.append((env.floating_base.get_pos() - step_base_pos_first).tolist())
                    step_base_ori_list.append((env.floating_base.get_ori() - step_base_ori_first).tolist())
                    image_list.append(img)

                    episode_errors_list = get_episode_error(episode_errors_list, env, episode_steps)
                    episode_rewards_list.append(reward)
                    print('reward = \r{}'.format(reward), end='')

                    if args.use_state_norm:
                        s_ = state_norm(s_, update=False)

                    s = s_

                    if done and not args.finish_after_reach and episode_steps <= args.max_episode_steps:
                        success_flag = True
                        success_list[episode_num] = 1

                    if done:
                        average_reward = sum(episode_rewards_list) / episode_steps
                        if success_flag:
                            success_flag = False
                            done = False
                            del (env.pos_err_queue[:env.success_interval])

                basename = os.path.basename(model_path)[:-4]
                # evaluation_step_plot(args, dirs, step_error_list, step_jp_list, step_jv_list, step_base_ori_list,
                #                 basename, episode_num, base_orientation_plot=False)

                if total_rewards_list == []:
                    total_rewards_list = [[item] for item in episode_rewards_list]
                    total_rewards_list.append([np.array(episode_rewards_list[-stable_step_num:-1]).mean()])
                    total_rewards_list.append([np.array(episode_rewards_list).mean()])
                    total_rewards_list.append([-args.pos_err_thres])
                else:
                    for i in range(len(episode_rewards_list)):
                        total_rewards_list[i].append(episode_rewards_list[i])
                    total_rewards_list[-3].append(np.array(episode_rewards_list[-stable_step_num:-1]).mean())
                    total_rewards_list[-2].append(np.array(episode_rewards_list).mean())
                    total_rewards_list[-1].append(-args.pos_err_thres)

                mean_rewards_list.append(np.mean(np.array(episode_rewards_list)))
                stable_rewards_list.append(np.mean(np.array(episode_rewards_list[-stable_step_num:])))
                final_rewards_list.append(episode_rewards_list[-1])

                episode_errors_list = np.array(episode_errors_list)
                if mean_errors_list == []:
                    for j in range(episode_errors_list.shape[1]):
                        mean_errors_list.append([np.mean(episode_errors_list[:, j])])
                        stable_errors_list.append([np.mean(episode_errors_list[-stable_step_num:, j])])
                        final_errors_list.append([np.mean(episode_errors_list[-1, j])])
                else:
                    for j in range(episode_errors_list.shape[1]):
                        mean_errors_list[j].append(np.mean(episode_errors_list[:, j]))
                        stable_errors_list[j].append(np.mean(episode_errors_list[-stable_step_num:, j]))
                        final_errors_list[j].append(np.mean(episode_errors_list[-1, j]))

            # evaluation_episode_plot(args, dirs, mean_rewards_list, stable_rewards_list, final_rewards_list,
            #           mean_errors_list, stable_errors_list, final_errors_list)
            #
            # write_to_csv(os.path.join(dirs['eval_csv'], os.path.basename(model_path)[:-4]+'.csv'), total_rewards_list)
            # radar_plot(total_rewards_list[-3], total_rewards_list[-2], total_rewards_list[-1],
            #            os.path.join(dirs['eval_radar'], model_path.split('/')[-1][:-4]+'.png'))
            print('\nsuccess rate is:{}'.format(np.sum(success_list)/len(success_list)))

            os.rename(model_path, os.path.join(dirs['log_'], os.path.basename(model_path)))

# region
parser = argparse.ArgumentParser('SpaceManipulator')

parser.add_argument('--scene_type', type=str2BoolNone, default='ffsm', choices=['fbsm', 'ffsm'],
                    help='fixed-based space manipulator or free-floating space manipulator')

parser.add_argument('--max_train_steps', type=float2int, default=5e6, help='Maximum number of training steps')

# network structure
parser.add_argument('--hidden_structure', type=str2list, default='[256, 256, 128]',
                    help='The structure of hidden layers of the neural network')


parser.add_argument('--replaybuffer_max_size', type=float2int, default=1e6, help='DDPG replay buffer max size')
parser.add_argument('--random_steps', type=float2int, default=25e3, help='DDPG random steps')
parser.add_argument('--noise_std_index', type=float, default=0.1, help='Noise std index, noise_std = noise_std_index * joint_max_vel')
parser.add_argument('--save_batch_size', type=int, default=60000, help='Save Batch size, use when save model')
parser.add_argument('--use_state_norm', type=str2BoolNone, default=True, help='Trick 2:state normalization')
parser.add_argument('--use_reward_scaling', type=str2BoolNone, default=True, help='Trick 4:reward scaling')

parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index')
parser.add_argument('--gpu_num', type=int, default=2, help='GPU numbers, 3 for ass01 & ass02; 2 for ass03')

# hyper parameters
parser.add_argument('--tau', type=float, default=1e-4, help='Soft update')
parser.add_argument('--update_freq', type=float2int, default=50, help='DDPG update frequency')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--lr_a', type=float, default=2e-4, help='Learning rate of actor')
parser.add_argument('--lr_c', type=float, default=2e-4, help='Learning rate of critic')
parser.add_argument('--gamma', type=float, default=0.94, help='Discount factor')

parser.add_argument('--lr_scheduler', type=str2BoolNone, default='singlecyclic', choices=['const', 'cyclic', 'singlecyclic'],
                    help='Trick 6.1: learning rate scheduler, type = const, cyclic, singlecyclic, exp (useless)')
parser.add_argument('--lr_cycle', type=int, default=15, help='Trick 6.21: learning rate change cycle, for cyclic')
parser.add_argument('--lr_ac_minrate', type=float, default=0.05,
                    help='Trick 6.4.3 lr_a/lr_c minrate: lr_a/c_now = max(lr_a/c_now, lr_a/c * lr_ac_minrate)')
parser.add_argument('--lr_minrate', type=float, default=0.2,
                    help='Trick 6.5 lr minrate: if use cyclic, this can make sure: minimum peak = lr * lr_minrate')


parser.add_argument('--save_dir', type=str, default='ddpgtest', help='Data save path')

parser.add_argument('--action_dim', type=int, default=6, help='Dimension of action')
parser.add_argument('--seed', type=int, default=1, help='The seed for training')
parser.add_argument('--fps_check_times', type=int, default=300, help='FPS check per times')
parser.add_argument('--model_save_freq', type=int, default=1, help='Saving model after __ update')
parser.add_argument('--eval_episodes', type=int, default=300, help='sf_num')
parser.add_argument('--use_ensure_quat_continuity', type=str2BoolNone, default=False, help='Use ensure_quat_continuity')
parser.add_argument('--joint_max_vel', type=int, default=3, help='Joint max velocity')
parser.add_argument('--time_interval', type=float, default=100., help='Simulate environment time interval')

# '''
parser.add_argument('--is_train', type=str2BoolNone, default=True, help='Train or evaluation')
parser.add_argument('--target_pos_reinit_method', type=str2BoolNone, default='distribution', choices=['random', 'sequence', 'distribution'],
                    help='Target position reinit by random/sequence/distribution')
parser.add_argument('--MNQ', type=str, default='5 6 10', help='Sphere distribution reinit numbers')
parser.add_argument('--r_max', type=float, default=0.65, help='Max distance from agent to target')
parser.add_argument('--r_min', type=float, default=0.25, help='Min distance from agent to target')
parser.add_argument('--target_reinit_pos_freq', type=int, default=1, help='After this times, target will reinit position')
'''
parser.add_argument('--is_train', type=str2BoolNone, default=False, help='Train or evaluation')
parser.add_argument('--target_pos_reinit_method', type=str2BoolNone, default='random', choices=['random', 'sequence', 'distribution'],
                    help='Target position reinit by random/sequence/distribution')
parser.add_argument('--MNQ', type=str, default='5 6 10', help='Sphere distribution reinit numbers')
parser.add_argument('--r_max', type=float, default=0.65, help='Max distance from agent to target')
parser.add_argument('--r_min', type=float, default=0.25, help='Min distance from agent to target')
parser.add_argument('--target_reinit_pos_freq', type=int, default=1, help='After this times, target will reinit position')
# '''

parser.add_argument('--headless', type=str2BoolNone, default=True, help='Use CoppeliaSim with headless mode')
parser.add_argument('--restore', type=str2BoolNone, default=True, help='Restore model')
parser.add_argument('--model_name', type=str2BoolNone, default=None, help='Evaluation model name')

parser.add_argument('--finish_after_reach', type=str2BoolNone, default=False, help='Finish episode after reach')
parser.add_argument('--success_interval', type=int, default=5, help='If no finish after reach, interval this to calculate next success')

parser.add_argument('--max_episode_steps', type=int, default=49, help='The max step num per episode')
parser.add_argument('--stable_step_num', type=int, default=10, help='Stable step number')
parser.add_argument('--pos_err_stable_times', type=int, default=5, help='Tracking stable if position error < threshold more than this times')
parser.add_argument('--pos_err_thres', type=float, default=0.2, help='Tracking position error threshold')
parser.add_argument('--arm_angle_0_to_2pi', type=str2BoolNone, default=True, help='Arm angle just limited in 0 to 2pi')
parser.add_argument('--success_sample_rate', type=float, default=0.95, help='Success sample rate')
parser.add_argument('--success_min_rate', type=float, default=0, help='Success minimum rate')

parser.add_argument('--ori_err_thres', type=float, default=0.6, help='Tracking quaternion error threshold')
parser.add_argument('--ori_inherent_rate', type=float, default=0.25, help='The inherent rate between pos_error and ori_error')
parser.add_argument('--ori_penalty_rate', type=float, default=0.5, help='The rate of orientation error penalty')
parser.add_argument('--pos_related_to_finish', type=str2BoolNone, default=True, help='Position error related to finish')
parser.add_argument('--ori_related_to_finish', type=str2BoolNone, default=True, help='Orientation error related to finish')

parser.add_argument('--fail_prob_decrease', type=str2BoolNone, default=True, help='Fail prob decrease')
parser.add_argument('--decrease_threshold', type=float, default=1e-2, help='Decrease if fail prob less than threshold')
parser.add_argument('--success_sample_rate_idx', type=float, default=1, help='Success sample rate decrease index')
parser.add_argument('--pos_err_stable_times_increase', type=int, default=1, help='Stable times increase number')

parser.add_argument('--pos_err_thres_idx', type=float, default=0.80, help='Position error threshold decrease index')
parser.add_argument('--ori_err_thres_idx', type=float, default=0.85, help='Quaternion error threshold decrease index')

# reward function define
parser.add_argument('--use_pos_err_penalty', type=str2BoolNone, default=True, help='Reward use position error penalty')
parser.add_argument('--pos_err_type', type=str2BoolNone, default='linear', choices=['linear', 'sum'],
                    help='Linear type: Pos Err = sqrt(Ex^2+Ey^2+Ez^2); Sum type: Pos Err = sum(Ex+Ey+Ez)')

parser.add_argument('--use_ori_err_penalty', type=str2BoolNone, default=True, help='Reward use orientation error penalty')
parser.add_argument('--ori_err_type', type=str2BoolNone, default='dangle', choices=['dangle', 'minusdot'],
                    help='Dangle type: Ori Err = delta angle;  Minusdot type: Ori Err = 1-dot(ang1*ang2)**2')

parser.add_argument('--use_ori_decrease_reward', type=str2BoolNone, default=True, help='Reward use orientation error decrease reward')
parser.add_argument('--ori_decrease_reward_rate', type=float, default=0.1, help='Ori decrease reward rate')

parser.add_argument('--use_smooth_penalty', type=str2BoolNone, default=True, help='Reward use smooth penalty')

parser.add_argument('--use_done_reward', type=str2BoolNone, default=True, help='Reward use done reward')
parser.add_argument('--done_reward_rate', type=float, default=0.1, help='Done reward rate')

# state space define
parser.add_argument('--state_dim_choose', type=str2BoolNone, default='eterr3p4qdpopv',
                    choices=['eterr3p4qdpopv', 'eterr3p4qpv', 'err3p4qdpopv'],
                    help='E:end, T:target, Err:error;'
                    'eterr3p4qdpopv: State=(Ex~z, Tx~z, Errx~z(position), Ex~w, Tx~w, Errx~w(quaternion), dis_pos, dis_ori, pj, vj)'
                    'eterr3p4qpv: State=(Ex~z, Tx~z, Errx~z(position), Ex~w, Tx~w, Errx~w(quaternion), pj, vj)'
                    'err3p4qdpopv: State=(Errx~z(position), Errx~w(quaternion), dis_pos, dis_ori, pj, vj)')


parser.add_argument('--pos_zero_origin', type=str2BoolNone, default='ee', choices=['ee', 'ab'],
                    help='ee: end effector; ab: agent base')
parser.add_argument('--state_dim', type=int, default=3, help='Useless')

args = parser.parse_args()
# endregion

if __name__ == '__main__':
    from Envs.pybullet_SpaceManipulator_reacher import PybulletSpaceManipulatorReacher

    plat = platform_check()
    get_gpuidx_from_dir(args)
    env = PybulletSpaceManipulatorReacher(args, plat=plat)

    main(args, env)
