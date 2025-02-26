import os
import sys
import time
import numpy as np
np.set_printoptions(precision=3, suppress=True)
from Utils.normalization import Normalization, RewardScaling
from Utils.replaybuffer import Off_Policy_ReplayBuffer
from Utils.utils import *
from Agents.DDPG import DDPG
sys.path.append('..')


def main(args, env):
    set_seed(args.seed, env)

    args.state_dim = env.state.shape[0]

    print('state_dim={}'.format(args.state_dim))
    print('action_dim={}'.format(args.action_dim))
    print('max_episode_steps={}'.format(args.max_episode_steps))
    print('mode:', 'train' if args.is_train else 'evaluation')

    dirs = create_saving_dirs(args)
    print('Create directory successful!\nROOT FOLDER: >>>> {} <<<<'.format(args.save_dir))

    model_path = None
    stable_step_num = args.stable_step_num

    agent = DDPG(args)
    # print(agent.actor)

    state_norm = Normalization(shape=args.state_dim)
    reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    mean_rewards_list, stable_rewards_list, final_rewards_list = [], [], []
    mean_errors_list, stable_errors_list, final_errors_list = [], [], []
    actor_loss_list, critic_loss_list, actor_lr_list, critic_lr_list = [], [], [], []

    if args.is_train:    # train
        total_steps = 0  # Record the total steps during the training
        episode_num = 0
        update_time = 0
        noise_std = 0.1 * args.joint_max_vel

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
                episode_num_, total_steps_, state_norm_, reward_scaling_, \
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
                total_steps, state_norm, reward_scaling, update_time = total_steps_, state_norm_, reward_scaling_, update_time_

        total_steps_last = total_steps
        data_length = len(stable_rewards_list)
        loss_length = len(actor_loss_list)

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

                if done and episode_steps <= args.max_episode_steps:
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
            if total_steps % args.save_interval == 0:
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
        model_path = get_model_paths(dirs['log'])[0]

        if os.path.exists(model_path):
            data_path = os.path.join(dirs['log'], 'data.pth')
            if not os.path.exists(data_path):
                print('Cannot find data!')
                # exit(-1)
            _, _, state_norm, _, _ = agent.reload_model(model_path)

            print('Reload model from', model_path)
        else:
            raise ValueError('No model file is found in {}'.format(model_path))

        # time.sleep(2)

        for episode_num in range(0, eval_episodes):
            step_jv_list = []
            step_jp_list = []
            step_error_list = []
            step_base_pos_list = []
            step_base_ori_list = []
            image_list = []

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

                print('\repisode_number = {}, reward = {}'.format(episode_num+1, reward), end='')

                if args.use_state_norm:
                    s_ = state_norm(s_, update=False)

                s = s_

                if done and episode_steps <= args.max_episode_steps:
                    success_flag = True
                    success_list[episode_num] = 1

                if done:
                    average_reward = sum(episode_rewards_list) / episode_steps
                    if success_flag:
                        success_flag = False
                        done = False
                        del (env.pos_err_queue[:env.success_interval])

            basename = os.path.basename(model_path)[:-4]

            mean_rewards_list.append(np.mean(np.array(episode_rewards_list)))
            stable_rewards_list.append(np.mean(np.array(episode_rewards_list[-stable_step_num:])))
            final_rewards_list.append(episode_rewards_list[-1])

            episode_errors_list = np.array(episode_errors_list)
            if not mean_errors_list:
                for j in range(episode_errors_list.shape[1]):
                    mean_errors_list.append([np.mean(episode_errors_list[:, j])])
                    stable_errors_list.append([np.mean(episode_errors_list[-stable_step_num:, j])])
                    final_errors_list.append([np.mean(episode_errors_list[-1, j])])
            else:
                for j in range(episode_errors_list.shape[1]):
                    mean_errors_list[j].append(np.mean(episode_errors_list[:, j]))
                    stable_errors_list[j].append(np.mean(episode_errors_list[-stable_step_num:, j]))
                    final_errors_list[j].append(np.mean(episode_errors_list[-1, j]))

        print('\nsuccess rate is:{}'.format(np.sum(success_list)/len(success_list)))


# region
parser = argparse.ArgumentParser('SpaceManipulator')
parser.add_argument('--gpu_idx', type=int, default=0,
                    help='GPU index')
parser.add_argument('--seed', type=int, default=1,
                    help='The seed for training')
parser.add_argument('--save_dir', type=str, default='ddpg',
                    help='Data save path')
parser.add_argument('--headless', type=str2BoolNone, default=False,
                    help='Use pybullet with headless mode')
parser.add_argument('--pos_zero_origin', type=str2BoolNone, default='ee', choices=['ee', 'ab'],
                    help='ee: end effector; ab: agent base')

parser.add_argument('--is_train', type=str2BoolNone, default=False,
                    help='Train or Eval')
parser.add_argument('--restore', type=str2BoolNone, default=False,
                    help='Restore model')
parser.add_argument('--model_name', type=str2BoolNone, default=None,
                    help='Evaluation model name')
parser.add_argument('--eval_episodes', type=int, default=50,
                    help='the num of episodes to evaluate')
parser.add_argument('--max_train_steps', type=float2int, default=200000,
                    help='Maximum number of training steps')
parser.add_argument('--max_episode_steps', type=int, default=49,
                    help='The max step num per episode')
parser.add_argument('--replay_buffer_size', type=float2int, default=1e5,
                    help='DDPG replay buffer max size')
parser.add_argument('--random_steps', type=float2int, default=1e4,
                    help='DDPG random steps')
parser.add_argument('--save_interval', type=int, default=10000,
                    help='the interval to save model')
parser.add_argument('--use_state_norm', type=str2BoolNone, default=True,
                    help='Trick 2:state normalization')
parser.add_argument('--use_reward_scaling', type=str2BoolNone, default=True,
                    help='Trick 4:reward scaling')

# hyper parameters
parser.add_argument('--hidden_structure', type=str2list, default='[256, 256, 128]',
                    help='The structure of hidden layers of the neural network')
parser.add_argument('--action_dim', type=int, default=6,
                    help='Dimension of action')
parser.add_argument('--tau', type=float, default=1e-4,
                    help='Soft update')
parser.add_argument('--update_freq', type=float2int, default=50,
                    help='DDPG update frequency')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size')
parser.add_argument('--lr_a', type=float, default=2e-4,
                    help='Learning rate of actor')
parser.add_argument('--lr_c', type=float, default=2e-4,
                    help='Learning rate of critic')
parser.add_argument('--gamma', type=float, default=0.95,
                    help='Discount factor')

parser.add_argument('--joint_max_vel', type=int, default=3,
                    help='Joint max velocity')

parser.add_argument('--r_max', type=float, default=1,
                    help='Max distance from agent to target')
parser.add_argument('--r_min', type=float, default=0.8,
                    help='Min distance from agent to target')

parser.add_argument('--success_interval', type=int, default=5,
                    help='If no finish after reach, interval this to calculate next success')
parser.add_argument('--stable_step_num', type=int, default=10,
                    help='Stable step number')
parser.add_argument('--pos_err_stable_times', type=int, default=5,
                    help='Tracking stable if position error < threshold more than this times')
parser.add_argument('--pos_err_thres', type=float, default=0.2,
                    help='Tracking position error threshold')
parser.add_argument('--success_sample_rate', type=float, default=0.95,
                    help='Success sample rate')
parser.add_argument('--success_min_rate', type=float, default=0,
                    help='Success minimum rate')

parser.add_argument('--ori_err_thres', type=float, default=0.6,
                    help='Tracking quaternion error threshold')
parser.add_argument('--ori_penalty_rate', type=float, default=0.5,
                    help='The rate of orientation error penalty')
parser.add_argument('--fail_prob_decrease', type=str2BoolNone, default=True,
                    help='Fail prob decrease')
parser.add_argument('--decrease_threshold', type=float, default=1e-2,
                    help='Decrease if fail prob less than threshold')

args = parser.parse_args()

if __name__ == '__main__':
    from Envs.SpaceManipulator import SpaceManipulator

    plat = platform_check()
    env = SpaceManipulator(args, plat=plat)

    main(args, env)
