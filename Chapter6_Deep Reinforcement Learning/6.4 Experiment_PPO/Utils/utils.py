import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import glob
import platform
import argparse
import csv
import torch
import ast
import time
import copy

from scipy.spatial.transform import Rotation as R

from Utils.normalization import Normalization, RewardScaling


def str2BoolNone(str):
    if str.lower() == 'true':
        return True
    elif str.lower() == 'false':
        return False
    elif str.lower() == 'none':
        return None
    else:
        return str.lower()


def float2int(f):
    return int(f)


def str2list(str):
    return ast.literal_eval(str)


def get_latest_model_path(log_dir, model_name=None):
    if model_name is None:
        model_paths = glob.glob(os.path.join(log_dir, 'model_ep_*.pth'))
        if len(model_paths) > 0:
            print('=> found {} models in {}'.format(len(model_paths), log_dir))
            created_times = [os.path.getmtime(path) for path in model_paths]
            latest_path = model_paths[np.argmax(created_times)]
            print('=> the latest model path: {}'.format(latest_path))
            return latest_path
        else:
            raise ValueError('No pre-trained model found!')
    else:
        model_path = os.path.join(log_dir, model_name)
        if os.path.exists(model_path):
            return model_path
        else:
            raise ValueError('No pre-trained model found!')


def get_model_paths(log_dir):
    model_paths = glob.glob(os.path.join(log_dir, 'model_ep_*.pth'))
    if len(model_paths) > 0:
        print('=> found {} models in {}'.format(len(model_paths), log_dir))
        return sorted(model_paths, reverse=True)
    else:
        raise ValueError('No pre-trained model found!')

def create_saving_dirs(args):
    save_dir = os.path.join('save_files', args.save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if args.is_train:
        with open(os.path.join(save_dir, 'args.txt'), 'w+') as f:
            for arg in vars(args):
                f.write(f'{arg} {getattr(args, arg)}\n')

    log_dir = os.path.join(save_dir, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    increment_dir = os.path.join(log_dir, 'increment')
    if not os.path.exists(increment_dir):
        os.makedirs(increment_dir)

    log_dir_ = os.path.join(log_dir, '_')
    if not os.path.exists(log_dir_):
        os.makedirs(log_dir_)

    data_dir = os.path.join(save_dir, 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    data_reward_dir = os.path.join(data_dir, 'reward')
    if not os.path.exists(data_reward_dir):
        os.makedirs(data_reward_dir)

    data_pos_err_dir = os.path.join(data_dir, 'position_error')
    if not os.path.exists(data_pos_err_dir):
        os.makedirs(data_pos_err_dir)

    data_ori_err_dir = os.path.join(data_dir, 'orientation_error')
    if not os.path.exists(data_ori_err_dir):
        os.makedirs(data_ori_err_dir)

    data_success_rate_dir = os.path.join(data_dir, 'success_rate')
    if not os.path.exists(data_success_rate_dir):
        os.makedirs(data_success_rate_dir)

    eval_dir = os.path.join(save_dir, 'eval')
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    eval_dir_csv = os.path.join(eval_dir, 'csv_save')
    if not os.path.exists(eval_dir_csv):
        os.makedirs(eval_dir_csv)

    eval_dir_radar = os.path.join(eval_dir, 'radar_save')
    if not os.path.exists(eval_dir_radar):
        os.makedirs(eval_dir_radar)

    curve_dir = os.path.join(save_dir, 'curve')
    if not os.path.exists(curve_dir):
        os.makedirs(curve_dir)

    curve_train_dir = os.path.join(curve_dir, 'train')
    if not os.path.exists(curve_train_dir):
        os.makedirs(curve_train_dir)

    curve_episode_pos_err_dir = os.path.join(curve_train_dir, 'position_error')
    if not os.path.exists(curve_episode_pos_err_dir):
        os.makedirs(curve_episode_pos_err_dir)

    curve_episode_ori_err_dir = os.path.join(curve_train_dir, 'orientation_error')
    if not os.path.exists(curve_episode_ori_err_dir):
        os.makedirs(curve_episode_ori_err_dir)

    curve_episode_reward_dir = os.path.join(curve_train_dir, 'reward')
    if not os.path.exists(curve_episode_reward_dir):
        os.makedirs(curve_episode_reward_dir)

    curve_loss_dir = os.path.join(curve_train_dir, 'loss')
    if not os.path.exists(curve_loss_dir):
        os.makedirs(curve_loss_dir)

    curve_test_dir = os.path.join(curve_dir, 'test')
    if not os.path.exists(curve_test_dir):
        os.makedirs(curve_test_dir)

    curve_step_jv_dir = os.path.join(curve_test_dir, 'step_joint_velocity')
    if not os.path.exists(curve_step_jv_dir):
        os.makedirs(curve_step_jv_dir)

    curve_step_jp_dir = os.path.join(curve_test_dir, 'step_joint_position')
    if not os.path.exists(curve_step_jp_dir):
        os.makedirs(curve_step_jp_dir)

    curve_step_ee_quat_dir = os.path.join(curve_test_dir, 'step_endeffector_quaternion')
    if not os.path.exists(curve_step_ee_quat_dir):
        os.makedirs(curve_step_ee_quat_dir)

    curve_step_pos_err_dir = os.path.join(curve_test_dir, 'step_position_error')
    if not os.path.exists(curve_step_pos_err_dir):
        os.makedirs(curve_step_pos_err_dir)

    curve_step_ori_err_dir = os.path.join(curve_test_dir, 'step_orientation_error')
    if not os.path.exists(curve_step_ori_err_dir):
        os.makedirs(curve_step_ori_err_dir)

    curve_base_ori_dir = os.path.join(curve_test_dir, 'step_base_orientation')
    if not os.path.exists(curve_base_ori_dir):
        os.makedirs(curve_base_ori_dir)

    curve_video_dir = os.path.join(curve_test_dir, 'video')
    if not os.path.exists(curve_video_dir):
        os.makedirs(curve_video_dir)

    curve_eval_episode_reward_dir = os.path.join(curve_test_dir, 'episode_reward_error')
    if not os.path.exists(curve_eval_episode_reward_dir):
        os.makedirs(curve_eval_episode_reward_dir)

    curve_eval_episode_pos_dir = os.path.join(curve_test_dir, 'episode_position_error')
    if not os.path.exists(curve_eval_episode_pos_dir):
        os.makedirs(curve_eval_episode_pos_dir)

    curve_eval_episode_ori_dir = os.path.join(curve_test_dir, 'episode_orientation_error')
    if not os.path.exists(curve_eval_episode_ori_dir):
        os.makedirs(curve_eval_episode_ori_dir)

    dirs = {}
    dirs['save'] = save_dir
    dirs['log'] = log_dir
    dirs['log_'] = log_dir_
    dirs['increment'] = increment_dir
    dirs['data_reward'] = data_reward_dir
    dirs['data_pos_err'] = data_pos_err_dir
    dirs['data_ori_err'] = data_ori_err_dir
    dirs['data_success_rate'] = data_success_rate_dir
    dirs['eval'] = eval_dir
    dirs['eval_csv'] = eval_dir_csv
    dirs['eval_radar'] = eval_dir_radar
    dirs['episode_pos_err'] = curve_episode_pos_err_dir
    dirs['episode_ori_err'] = curve_episode_ori_err_dir
    dirs['episode_reward'] = curve_episode_reward_dir
    dirs['step_jv'] = curve_step_jv_dir
    dirs['step_jp'] = curve_step_jp_dir
    dirs['step_ee_quat'] = curve_step_ee_quat_dir
    dirs['step_pos_err'] = curve_step_pos_err_dir
    dirs['step_ori_err'] = curve_step_ori_err_dir
    dirs['base_ori'] = curve_base_ori_dir
    dirs['video'] = curve_video_dir
    dirs['loss'] = curve_loss_dir
    dirs['eval_episode_reward_err'] = curve_eval_episode_reward_dir
    dirs['eval_episode_pos_err'] = curve_eval_episode_pos_dir
    dirs['eval_episode_ori_err'] = curve_eval_episode_ori_dir
    return dirs


def get_episode_error(episode_errors_list, env, episode_steps):
    episode_errors_list.append(env.err_pos.tolist())
    episode_errors_list[episode_steps].append(env.dis_pos)
    episode_errors_list[episode_steps].extend(env.err_euler.tolist())
    episode_errors_list[episode_steps].append(env.dis_ori)
    episode_errors_list[episode_steps].extend(env.err_quat.tolist())
    return episode_errors_list


def get_joint_pos_vel(step_jp_list, step_jv_list, env):
    step_jp_list.append(env.joint_pos)
    step_jv_list.append(env.joint_vel)

    return step_jp_list, step_jv_list


def platform_check():
    platform_is_wsl = (platform.platform().find('WSL') != -1)
    platform_is_centos = (platform.platform().find('centos') != -1)
    platform_is_ubuntu = (platform.platform().find('generic') != -1)
    platform_is_windows = (platform.platform().find('Windows') != -1)
    assert (platform_is_wsl ^ platform_is_centos ^ platform_is_ubuntu ^ platform_is_windows)
    if platform_is_wsl:
        return 'wsl'
    elif platform_is_centos:
        return 'centos'
    elif platform_is_ubuntu:
        return 'ubuntu'
    elif platform_is_windows:
        return 'windows'


def check_files_with_suffix(directory, suffix):
    '''
        检查指定目录中是否有文件以特定后缀结尾。
        :param directory: 要检查的目录
        :param suffix: 文件后缀，例如 '.pth'
        :return: 如果找到至少一个文件则返回 True，否则返回 False
        '''
    # 遍历指定目录
    for item in os.listdir(directory):
        # 拼接完整的文件路径
        item_path = os.path.join(directory, item)
        # 检查是否为文件并且后缀匹配
        if os.path.isfile(item_path) and item_path.endswith(suffix):
            return True
    return False


def radar_plot(stable_lst, lst, thres_lst, save_path, t=None):
    # 定义数据：每个维度的值
    stable_values = stable_lst
    values = lst
    threshold = thres_lst
    # 定义数据：每个维度的名称
    # categories = ['A', 'B', 'C', 'D']

    # 计算每个维度的角度
    n = len(values)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()

    # 使雷达图闭合
    stable_values += stable_values[:1]
    values += values[:1]
    angles += angles[:1]
    threshold += threshold[:1]

    # 绘图
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # 绘制线条
    ax.fill(angles, stable_values, color='blue', alpha=0.25)
    ax.plot(angles, stable_values, color='blue', linewidth=2)  # 线条的样式
    ax.fill(angles, threshold, color='green', alpha=0.25)
    ax.plot(angles, threshold, color='green', linewidth=2)  # 线条的样式
    ax.fill(angles, values, color='red', alpha=0.25)
    ax.plot(angles, values, color='red', linewidth=2)  # 线条的样式

    # 设置极坐标的范围
    ax.set_ylim(min(np.floor((min(values) * 20)) / 20, -0.05), 0)

    # 添加每个特征的标签
    plt.xticks(angles[:-1])

    # 添加标题
    plt.title('Radar Chart Example')

    # 展示图形
    if t is None:
        plt.savefig(save_path)
    else:
        plt.savefig(save_path + str(t) + '.png')


def plt_plot(lst, title, ylabel, save_dir, use_2nd_xlabel=False, label_rate=1,
             xlabel_2nd='step', xlabel='episode', color='red'):
    x = list(range(len(lst)))

    fig, ax = plt.subplots()

    ax.plot(x, lst, color=color)
    ax.grid()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if use_2nd_xlabel:
        ax2 = ax.secondary_xaxis('top', functions=(lambda x: x * label_rate, lambda x: x / label_rate))
        ax2.set_xlabel(xlabel_2nd)
    ax.set_ylabel(ylabel)

    fig.savefig(save_dir)
    plt.close()


def plt_oneplot(lst, title, ylabel, legend, save_dir, xlabel='step', color=[(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)],
                lw=[1, 1, 1, 2], figsize=(8, 6), fontsize=9, settitle=True, subplot=True,
                zoom_loc=(0.97, 0.38, 1.5, 1), zoom_range_x=(25, 50), zoom_range_y=(-0.05, 0.05), ):
    plt.rcParams['font.size'] = fontsize
    l = len(lst)
    plt.figure(figsize=figsize, dpi=200)

    for i in range(l):
        plt.plot(lst[i], color=color[i], label=legend[i], linewidth=lw[i])

    if settitle:
        plt.title(title)
    plt.xlabel(xlabel), plt.ylabel(ylabel), plt.grid(), plt.legend(loc='upper right')
    plt.subplots_adjust(left=0.15, right=0.9, top=0.95, bottom=0.1)

    if subplot:
        ax_inset = inset_axes(plt.gca(), width=zoom_loc[2], height=zoom_loc[3],
                              bbox_to_anchor=(zoom_loc[0], zoom_loc[1]), bbox_transform=plt.gca().transAxes)
        for i in range(l):
            plt.plot(lst[i], color=color[i], linewidth=lw[i])
        ax_inset.set_xlim(zoom_range_x[0], zoom_range_x[1]), ax_inset.set_ylim(zoom_range_y[0],
                                                                               zoom_range_y[1]), ax_inset.grid()

    plt.savefig(save_dir, dpi=600), plt.close()


def plt_subplot(lst, title, ylabel, save_dir, xlabel='episode', color='red', figsize=(8, 6), fontsize=9,
                settitle=True, legend=None):
    plt.rcParams['font.size'] = fontsize
    l = len(lst)
    hspace = 0.2 if l <= 3 else 0.4
    fig, axs = plt.subplots(l, 1, figsize=figsize, gridspec_kw={'hspace': hspace, 'wspace': 0.5}, dpi=200)
    set_each_xlabel, set_each_ylabel = isinstance(xlabel, list), isinstance(ylabel, list)

    for i in range(l):
        x = list(range(len(lst[i])))
        if legend is not None:
            axs[i].plot(x, lst[i], color=color, label=legend[0] + '{}'.format(i + 1)), axs[i].grid()
        else:
            axs[i].plot(x, lst[i], color=color), axs[i].grid()
        if i != l - 1:
            axs[i].set_xticklabels([])
        if set_each_xlabel:
            axs[i].set_xlabel(xlabel[i])
        if set_each_ylabel:
            axs[i].set_ylabel(ylabel[i])
        if legend is not None:
            axs[i].legend(loc='upper right')

    if settitle:
        axs[0].set_title(title)
    if not set_each_xlabel:
        axs[-1].set_xlabel(xlabel)
    if not set_each_ylabel:
        fig.text(0.05, 0.5, ylabel, va='center', ha='center', rotation='vertical')
        # axs[int(l/2)].set_ylabel(ylabel)

    fig.subplots_adjust(left=0.15, right=0.9, top=0.95, bottom=0.1), plt.savefig(save_dir, dpi=600)
    plt.close()


def plt_multiplot(lst, title, save_dir, xlabel='step', ylabel='Orientation Angle, rad',
                  color=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
                  figsize=(8, 6), fontsize=9, label=['pitch', 'roll', 'yaw'], linestyle=['-', '--', '-.'], lw=[1, 1, 1],
                  l=0.15, r=0.9, t=0.95, b=0.1):
    plt.rcParams['font.size'] = fontsize
    plt.figure(figsize=figsize, dpi=200)
    x = np.arange(len(lst[0]))
    for i in range(len(lst)):
        label_ = label if isinstance(label, str) else label[i]
        color_ = color if isinstance(color, str) else color[i]
        linestyle_ = linestyle if isinstance(linestyle, str) else linestyle[i]
        plt.plot(x, lst[i], label=label_, color=color_, linestyle=linestyle_, linewidth=lw[i])

    plt.grid(), plt.legend(), plt.xlabel(xlabel), plt.ylabel(ylabel)
    # plt.title(title)
    plt.subplots_adjust(left=l, right=r, top=t, bottom=b)
    plt.savefig(save_dir, dpi=600), plt.close()


def plt_combineplot(lst_l, lst_r, title_l, title_r, ylabel_l, ylabel_r, save_dir, use_2nd_xlabel=False, label_rate=1,
                    xlabel_l='episode', xlabel_r='episode', xlabel_2nd='step', color_l='red', color_r='red'):
    '''

    :param lst_l:           左边图片的 list, 通常是 N*3 的 list
    :param lst_r:           右边图片的 list, 通常是 N*1 的 list
    :param title_l:         左边图片的 title
    :param title_r:         右边图片的 title
    :param ylabel_l:        左边的 ylabel
    :param ylabel_r:        右边的 ylabel
    :param save_dir:        保存路径
    :param use_2nd_xlabel:  使用第二根 x 轴
    :param label_rate:      第二根 x 轴倍率
    :param xlabel_l:        左边的 xlabel
    :param xlabel_r:        右边的 xlabel
    :param color_l:         左边的颜色
    :param color_r:         右边的颜色
    :return:                NULL
    '''

    # lst_l 应该是一个二维的 list
    assert (isinstance(np.array(lst_l).shape[1], int))
    # 创建一个图形对象
    lst1_len = len(lst_l)

    set_each_xlabel = isinstance(xlabel_l, list)
    set_each_ylabel = isinstance(ylabel_l, list)
    fig = plt.figure(figsize=(18, 7))
    # 创建 GridSpec 对象，3行2列的网格
    gs = gridspec.GridSpec(lst1_len, 2, width_ratios=[1, 1])
    # 左边 n 张图分别占用 (0,0), (1,0), ..., (n,0) 单元格
    axs = []
    for i in range(lst1_len):
        axs.append(fig.add_subplot(gs[i, 0]))
    # 右边 1 张图占用 (0:n,1) 单元格
    axs.append(fig.add_subplot(gs[:, 1]))

    # 绘制左边的图
    for i in range(lst1_len):
        x = list(range(len(lst_l[i])))
        axs[i].plot(x, lst_l[i], color=color_l)
        axs[i].grid()
        if set_each_xlabel:
            axs[i].set_xlabel(xlabel_l[i])
            if use_2nd_xlabel:
                axs2 = axs[i].secondary_xaxis('top', functions=(lambda x: x * label_rate, lambda x: x / label_rate))
                axs2.set_xlabel(xlabel_2nd)
        if set_each_ylabel:
            axs[i].set_ylabel(ylabel_l[i])

    axs[0].set_title(title_l)
    if not set_each_xlabel:
        axs[-2].set_xlabel(xlabel_l)
        if use_2nd_xlabel:
            axs2 = axs[0].secondary_xaxis('top', functions=(lambda x: x * label_rate, lambda x: x / label_rate))
            axs2.set_xlabel(xlabel_2nd)
    if not set_each_ylabel:
        axs[int(lst1_len / 2)].set_ylabel(ylabel_l)

    # 绘制右边的图
    x = list(range(len(lst_r)))
    axs[-1].plot(x, lst_r, color=color_r)
    axs[-1].grid()
    axs[-1].set_title(title_r)
    axs[-1].set_xlabel(xlabel_r)
    if use_2nd_xlabel:
        axs2 = axs[-1].secondary_xaxis('top', functions=(lambda x: x * label_rate, lambda x: x / label_rate))
        axs2.set_xlabel(xlabel_2nd)
    axs[-1].set_ylabel(ylabel_r)

    # 调整布局
    plt.tight_layout()
    # 显示图形
    plt.savefig(save_dir)
    plt.close()


def smooth_filter(lst, n, mode='valid'):
    if np.array(lst).ndim == 1:
        x = np.convolve(lst, np.ones(n) / n, mode=mode)
    else:
        for i in range(len(lst)):
            if i:
                x = np.vstack((x, np.convolve(lst[i], np.ones(n) / n, mode=mode)))
            else:
                x = np.convolve(lst[i], np.ones(n) / n, mode=mode)
    return x


def abs_smooth_filter(lst, n, mode='valid'):
    if np.array(lst).ndim == 1:
        x = np.convolve(np.abs(lst), np.ones(n) / n, mode=mode)
    else:
        for i in range(len(lst)):
            if i:
                x = np.vstack((x, np.convolve(np.abs(lst[i]), np.ones(n) / n, mode=mode)))
            else:
                x = np.convolve(np.abs(lst[i]), np.ones(n) / n, mode=mode)
    return x


def lT(lst):
    return np.array(lst).T.tolist()


def pm2picont(lst):
    arr = np.array(lst)
    for i in range(1, len(arr)):
        if arr[i] - arr[i - 1] > 3:
            arr[i:] -= np.pi * 2
        elif arr[i] - arr[i - 1] < -3:
            arr[i:] += np.pi * 2
    while np.max(arr) > np.pi:
        arr -= np.pi
    while np.min(arr) < -np.pi:
        arr += np.pi
    return arr.tolist()


def flipthres(lst, thres):
    arr = np.array(lst)
    for i in range(1, len(arr)):
        if np.abs(arr[i] - arr[i - 1]) > thres:
            arr[i:] *= -1
    return arr.tolist()


def cal_quat_error(q1, q2):
    '''
    计算两个四元数的姿态误差。
    :param q1, q2: 四元数。
    :return: 四元数姿态误差。
    '''
    return (R.from_quat(q1).inv() * R.from_quat(q2)).magnitude()


def quat_to_cont_repre(data):
    return R.from_quat(data).as_matrix()[:, :2].T.flatten()


def set_seed(seed, env):
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def train_post_process(episode_num, episode_steps, tracking_success_times, env, episode_errors_list,
                       stable_rewards_list, episode_rewards_list, stable_step_num, stable_errors_list,
                       args, total_steps, total_steps_last, t_last, dirs):
    print('episode_number={}, steps={}'.format(episode_num, episode_steps))
    if tracking_success_times:
        print('Tracking success times is: {}'.format(tracking_success_times))

    episode_errors_list = np.array(episode_errors_list)

    stable_rewards_list.append(np.mean(np.array(episode_rewards_list[-stable_step_num:])))
    if stable_errors_list == []:
        for j in range(episode_errors_list.shape[1]):
            stable_errors_list.append([np.mean(episode_errors_list[-stable_step_num:, j])])
    else:
        for j in range(episode_errors_list.shape[1]):
            stable_errors_list[j].append(np.mean(episode_errors_list[-stable_step_num:, j]))

    print('final reward is:{}'.format(episode_rewards_list[-1]))

    if episode_num % 500 == 0:
        t = time.time()
        fps = (total_steps - total_steps_last) / (t - t_last)
        f_hours = (args.max_train_steps - total_steps) / fps / 3600
        with open(os.path.join(dirs['save'], 'fps.txt'), 'a+') as f:
            f.write('Episode {},\t fps: {:.2f},\tfinish after {:.2f} hours\n'.format(
                episode_num, fps, f_hours))

        total_steps_last, t_last = total_steps, t
    return total_steps_last, t_last


def DDPG_train_post_process(
        args, episode_num, dirs, env, agent, total_steps, state_norm, reward_scaling,
        model_path, stable_rewards_list, stable_errors_list, actor_loss_list, critic_loss_list,
        actor_lr_list, critic_lr_list, data_length, loss_length
):
    update_time = int(total_steps / args.save_interval)

    tmp_reward_scaling = reward_scaling

    if total_steps % args.save_interval == 0:
        model_path = os.path.join(dirs['log'], 'model_ep_{:06d}.pth'.format(episode_num))
        increment_data_path = os.path.join(dirs['increment'], 'data_{:06d}.pth'.format(update_time))
        agent.save_model(episode_num=episode_num, total_steps=total_steps, state_norm=state_norm,
                         reward_scaling=tmp_reward_scaling, update_time=update_time, model_path=model_path, )
        save_data_stable_increment(stable_rewards_list[data_length:],
                                   [sublist[data_length:] for sublist in stable_errors_list],
                                   actor_loss_list[loss_length:], critic_loss_list[loss_length:],
                                   actor_lr_list[loss_length:], critic_lr_list[loss_length:],
                                   data_path=increment_data_path)
        reward_error_plot(args, dirs, episode_num, stable_rewards_list, stable_errors_list, update_time)
        loss_lr_plot(args, dirs, episode_num, actor_loss_list, critic_loss_list,
                     actor_lr_list, critic_lr_list)

    return tmp_reward_scaling


# save and reload data
# region
def save_data_stable_increment(stable_rewards_list, stable_errors_list, actor_loss_list, critic_loss_list,
                               actor_lr_list, critic_lr_list, episode_num_list=0, success_num_list=0, data_path=''):
    print('=> saving data to {} ...'.format(data_path))
    checkpoint = {'stable_rewards_list': stable_rewards_list,
                  'stable_errors_list': stable_errors_list,
                  'actor_loss_list': actor_loss_list,
                  'critic_loss_list': critic_loss_list,
                  'actor_lr_list': actor_lr_list,
                  'critic_lr_list': critic_lr_list,
                  'episode_num_list': episode_num_list,
                  'success_num_list': success_num_list,
                  }
    torch.save(checkpoint, data_path, _use_new_zipfile_serialization=False)
    print('=> data saved!')
    return


def reload_data_stable_increment(root_data_path=''):
    if os.path.exists(root_data_path):
        stable_rewards_list, stable_errors_list, actor_loss_list, critic_loss_list, \
            actor_lr_list, critic_lr_list, episode_num_list, success_num_list = [], [], [], [], [], [], [], []
        print('=> reloading data from {} ...'.format(root_data_path))
        data_path_list = sorted(os.listdir(root_data_path))

        for data_path in data_path_list:
            checkpoint = torch.load(os.path.join(root_data_path, data_path),
                                    map_location=lambda storage, loc: storage.cpu())
            if isinstance(checkpoint['stable_rewards_list'], list):
                stable_rewards_list.extend(checkpoint['stable_rewards_list'])
            elif isinstance(checkpoint['stable_rewards_list'], tuple):
                stable_rewards_list.extend(checkpoint['stable_rewards_list'][0])

            if isinstance(checkpoint['stable_errors_list'], list):
                if len(stable_errors_list) == 0:
                    stable_errors_list.extend(checkpoint['stable_errors_list'])
                else:
                    for i in range(len(stable_errors_list)):
                        stable_errors_list[i].extend(checkpoint['stable_errors_list'][i])
            elif isinstance(checkpoint['stable_errors_list'], tuple):
                if len(stable_errors_list) == 0:
                    stable_errors_list.extend(checkpoint['stable_errors_list'][0])
                else:
                    for i in range(len(stable_errors_list)):
                        stable_errors_list[i].extend(checkpoint['stable_errors_list'][0][i])

            if isinstance(checkpoint['actor_loss_list'], list):
                actor_loss_list.extend(checkpoint['actor_loss_list'])
            elif isinstance(checkpoint['actor_loss_list'], tuple):
                actor_loss_list.extend(checkpoint['actor_loss_list'][0])

            if isinstance(checkpoint['critic_loss_list'], list):
                critic_loss_list.extend(checkpoint['critic_loss_list'])
            elif isinstance(checkpoint['critic_loss_list'], tuple):
                critic_loss_list.extend(checkpoint['critic_loss_list'][0])

            if isinstance(checkpoint['actor_lr_list'], list):
                actor_lr_list.extend(checkpoint['actor_lr_list'])
            elif isinstance(checkpoint['actor_lr_list'], tuple):
                actor_lr_list.extend(checkpoint['actor_lr_list'][0])

            if isinstance(checkpoint['critic_lr_list'], list):
                critic_lr_list.extend(checkpoint['critic_lr_list'])
            elif isinstance(checkpoint['critic_lr_list'], tuple):
                critic_lr_list.extend(checkpoint['critic_lr_list'][0])

            if isinstance(checkpoint['episode_num_list'], list):
                episode_num_list.extend(checkpoint['episode_num_list'])
            elif isinstance(checkpoint['episode_num_list'], tuple):
                episode_num_list.extend(checkpoint['episode_num_list'][0])

            if isinstance(checkpoint['success_num_list'], list):
                success_num_list.extend(checkpoint['success_num_list'])
            elif isinstance(checkpoint['success_num_list'], tuple):
                success_num_list.extend(checkpoint['success_num_list'][0])

    else:
        raise ValueError('No data file is found in {}'.format(data_path))
    return stable_rewards_list, stable_errors_list, actor_loss_list, critic_loss_list, \
        actor_lr_list, critic_lr_list, episode_num_list, success_num_list


# endregion

def reward_error_plot(args, dirs, episode_num, stable_rewards_list, stable_errors_list, update_time):
    sf_num = args.eval_episodes
    label_rate = args.max_episode_steps + 1
    update_episode_len = int(args.buffer_size / (args.max_episode_steps + 1))
    final_episode_num_list = (np.array([1, 2, 5, 10, 20]) * update_episode_len).tolist()

    sf_stable_rewards_list = smooth_filter(stable_rewards_list, sf_num)
    plt_plot(sf_stable_rewards_list, title='Stable Reward', ylabel='Stable Reward',
             save_dir=os.path.join(dirs['episode_reward'], 'StableReward_sf{}_ep{}.png'.format(sf_num, episode_num)),
             use_2nd_xlabel=True, label_rate=label_rate)

    abs_sf_stable_errors_list = abs_smooth_filter(stable_errors_list, sf_num)
    # position
    pos_err_start_num = 0
    pos_err_end_num = 3
    plt_combineplot(
        lst_l=abs_sf_stable_errors_list[pos_err_start_num:pos_err_end_num],
        lst_r=abs_sf_stable_errors_list[pos_err_end_num],
        title_l='StablePositionError', title_r='StablePositionError',
        ylabel_l=['X-Error', 'Y-Error', 'Z-Error', ], ylabel_r='StablePositionError',
        save_dir=os.path.join(dirs['episode_pos_err'],
                              'StablePosErr_sf{}_ep{}.png'.format(sf_num, episode_num)),
        use_2nd_xlabel=True, label_rate=label_rate
    )

    # orientation
    ori_err_start_num = 4
    ori_err_end_num = 7
    plt_combineplot(
        lst_l=abs_sf_stable_errors_list[ori_err_start_num:ori_err_end_num],
        lst_r=abs_sf_stable_errors_list[ori_err_end_num],
        title_l='StableOrientationError', title_r='StableOrientationError',
        ylabel_l=['Pitch-Error', 'Roll-Error', 'Yaw-Error', ], ylabel_r='StableOrientationError',
        save_dir=os.path.join(dirs['episode_ori_err'],
                              'StableOriErr_sf{}_ep{}.png'.format(sf_num, episode_num)),
        use_2nd_xlabel=True, label_rate=label_rate
    )

    i = 1
    # reward
    with open(os.path.join(dirs['data_reward'], 'stable_reward_{}.txt'.format(i)), 'a+') as f:
        f.writelines('Update Time {},\tFinal {} Episode,\tStable Reward = {}\n'.format(
            update_time, i, np.mean(sf_stable_rewards_list[-i:])))

    # axle_X-Y-Z
    with open(os.path.join(dirs['data_pos_err'], 'stable_X_error_{}.txt'.format(i)), 'a+') as f:
        f.writelines('Update Time {},\tFinal {} Episode,\tStable X Error = {}\n'.format(
            update_time, i, np.mean(abs_sf_stable_errors_list[pos_err_start_num + 0][-i:])))
    with open(os.path.join(dirs['data_pos_err'], 'stable_Y_error_{}.txt'.format(i)), 'a+') as f:
        f.writelines('Update Time {},\tFinal {} Episode,\tStable Y Error = {}\n'.format(
            update_time, i, np.mean(abs_sf_stable_errors_list[pos_err_start_num + 1][-i:])))
    with open(os.path.join(dirs['data_pos_err'], 'stable_Z_error_{}.txt'.format(i)), 'a+') as f:
        f.writelines('Update Time {},\tFinal {} Episode,\tStable Z Error = {}\n'.format(
            update_time, i, np.mean(abs_sf_stable_errors_list[pos_err_start_num + 2][-i:])))
    # position_error
    with open(os.path.join(dirs['data_pos_err'], 'stable_position_error_{}.txt'.format(i)), 'a+') as f:
        f.writelines('Update Time {},\tFinal {} Episode,\tStable Position Error = {}\n'.format(
            update_time, i, np.mean(abs_sf_stable_errors_list[pos_err_end_num][-i:])))

    # Pitch-Roll-Yaw
    with open(os.path.join(dirs['data_ori_err'], 'stable_Pitch_error_{}.txt'.format(i)), 'a+') as f:
        f.writelines('Update Time {},\tFinal {} Episode,\tStable Pitch Error = {}\n'.format(
            update_time, i, np.mean(abs_sf_stable_errors_list[ori_err_start_num + 0][-i:])))
    with open(os.path.join(dirs['data_ori_err'], 'stable_Roll_error_{}.txt'.format(i)), 'a+') as f:
        f.writelines('Update Time {},\tFinal {} Episode,\tStable Roll Error = {}\n'.format(
            update_time, i, np.mean(abs_sf_stable_errors_list[ori_err_start_num + 1][-i:])))
    with open(os.path.join(dirs['data_ori_err'], 'stable_Yaw_error_{}.txt'.format(i)), 'a+') as f:
        f.writelines('Update Time {},\tFinal {} Episode,\tStable Yaw Error = {}\n'.format(
            update_time, i, np.mean(abs_sf_stable_errors_list[ori_err_start_num + 2][-i:])))
    # orientation_error
    with open(os.path.join(dirs['data_ori_err'], 'stable_orientation_error_{}.txt'.format(i)), 'a+') as f:
        f.writelines('Update Time {},\tFinal {} Episode,\tStable Orientation Error = {}\n'.format(
            update_time, i, np.mean(abs_sf_stable_errors_list[ori_err_end_num][-i:])))


def loss_lr_plot(args, dirs, episode_num, actor_loss_list, critic_loss_list, actor_lr_list, critic_lr_list):
    plt_plot(actor_loss_list, title='Actor Loss', xlabel='update_times', ylabel='Actor Loss',
             save_dir=os.path.join(dirs['loss'], 'Actor_loss.png'.format(episode_num)))
    plt_plot(critic_loss_list, title='Critic Loss', xlabel='update_times', ylabel='Critic Loss',
             save_dir=os.path.join(dirs['loss'], 'Critic_loss.png'.format(episode_num)))
    plt_multiplot([actor_lr_list, critic_lr_list], title='lr_curve', xlabel='update_times', ylabel='learning_rate',
                  color=['red', 'blue'], label=['lr_a', 'lr_c'], linestyle=['--', '-.'],
                  save_dir=os.path.join(dirs['loss'], 'LearningRate.png'))


def success_rate_plot(args, dirs, episode_num, episode_num_list, success_num_list, update_time):
    success_rate_list = (np.array(success_num_list) / np.array(episode_num_list)).tolist()
    plt_plot(success_rate_list, title='Success Rate', xlabel='update_times', ylabel='Success Rate',
             save_dir=os.path.join(dirs['loss'], 'Success_rate.png'.format(episode_num)))

    # success_rate
    with open(os.path.join(dirs['data_success_rate'], 'success_rate.txt'), 'a+') as f:
        f.writelines('Update Time {},\tSuccess Rate = {}\n'.format(
            update_time, success_rate_list[-1]))


def evaluation_step_plot(args, dirs, step_error_list, step_jp_list, step_jv_list, step_base_ori_list,
                         basename, episode_num, base_orientation_plot=True):
    plt_subplot(lT(step_jp_list), 'Manipulator joint angles', xlabel='step', ylabel='Joint Position, rad',
                save_dir=os.path.join(dirs['step_jp'],
                                      'StepJointPosition_{}_scene{}.png'.format(basename, episode_num)))
    plt_subplot(lT(step_jv_list), 'Manipulator joint angular velocities', xlabel='step', ylabel='Joint Velocity, rad/s',
                save_dir=os.path.join(dirs['step_jv'],
                                      'StepJointVelocity_{}_scene{}.png'.format(basename, episode_num)))
    if base_orientation_plot:
        plt_multiplot(lT(step_base_ori_list), title='Base satellite orientation',
                      save_dir=os.path.join(dirs['base_ori'],
                                            'StepBaseOrientation_{}_scene{}.png'.format(basename, episode_num)),
                      label=['pitch', 'roll', 'yaw', 'Base spacecraft'])
    plt_combineplot(
        lst_l=lT(step_error_list)[:3],
        lst_r=lT(step_error_list)[3],
        title_l='PositionError', title_r='PositionError',
        ylabel_l=['X-Error', 'Y-Error', 'Z-Error', ], ylabel_r='PositionError',
        xlabel_l='step', xlabel_r='step',
        save_dir=os.path.join(dirs['step_pos_err'],
                              'StepPositionError_{}_scene{}.png'.format(basename, episode_num)))

    plt_combineplot(
        lst_l=lT(step_error_list)[4:7],
        lst_r=lT(step_error_list)[7],
        title_l='OrientationError', title_r='OrientationError',
        ylabel_l=['Pitch-Error', 'Roll-Error', 'Yaw-Error'], ylabel_r='OrientationError',
        xlabel_l='step', xlabel_r='step',
        save_dir=os.path.join(dirs['step_ori_err'],
                              'StepOrientationError_{}_scene{}.png'.format(basename, episode_num)))


def evaluation_step_plot_forpaper(args, dirs, step_error_list, step_jp_list, step_jv_list, step_base_pos_list,
                                  step_base_ori_list, basename, episode_num, base_orientation_plot=True):
    plt_subplot([pm2picont(lT(step_jp_list)[0]), pm2picont(lT(step_jp_list)[1]), pm2picont(lT(step_jp_list)[2]),
                 pm2picont(lT(step_jp_list)[3]), pm2picont(lT(step_jp_list)[4]), pm2picont(lT(step_jp_list)[5]), ],
                'Manipulator joint angles', xlabel='step', ylabel='Joint Position, rad',
                save_dir=os.path.join(dirs['step_jp'],
                                      'StepJointPosition_{}_scene{}.png'.format(basename, episode_num)),
                figsize=(5, 6), settitle=False, legend=['joint'], color=(232 / 255, 0, 11 / 255))
    plt_subplot(lT(step_jv_list), 'Manipulator joint angular velocities', xlabel='step', ylabel='Joint Velocity, rad/s',
                save_dir=os.path.join(dirs['step_jv'],
                                      'StepJointVelocity_{}_scene{}.png'.format(basename, episode_num)),
                figsize=(5, 6), settitle=False, legend=['joint'], color=(2 / 255, 62 / 255, 1))

    plt_subplot(lT(step_error_list)[:3], 'Manipulator joint angular velocities', xlabel='step',
                ylabel=['X-Error, m', 'Y-Error, m', 'Z-Error, m', ],
                save_dir=os.path.join(dirs['step_pos_err'],
                                      'StepPositionError_{}_scene{}.png'.format(basename, episode_num)),
                figsize=(5, 5), settitle=False, color=(232 / 255, 0, 11 / 255), )
    # plt_oneplot(lT(step_error_list)[:4], 'Manipulator joint angles', xlabel='step', ylabel='Position Error, m',
    plt_oneplot(lT(step_error_list)[:3], 'Manipulator joint angles', xlabel='step', ylabel='Position Error, m',
                save_dir=os.path.join(dirs['step_pos_err'],
                                      'StepPositionError_OnePlot_{}_scene{}.png'.format(basename, episode_num)),
                legend=['x', 'y', 'z', 'End-effector', ], figsize=(5, 4), settitle=False, lw=[1.5, 1.5, 1.5],
                color=[(232 / 255, 0, 11 / 255), (26 / 255, 201 / 255, 56 / 255), (2 / 255, 62 / 255, 1),
                       (0.15, 0.15, 0.15)])

    plt_subplot([flipthres(lT(step_error_list)[4], 3), flipthres(lT(step_error_list)[5], 3),
                 flipthres(lT(step_error_list)[6], 3)],
                'Manipulator joint angular velocities', xlabel='step',
                ylabel=['Pitch-Error, rad', 'Roll-Error, rad', 'Yaw-Error, rad', ],
                save_dir=os.path.join(dirs['step_ori_err'],
                                      'StepOrientationError_{}_scene{}.png'.format(basename, episode_num)),
                figsize=(5, 5), settitle=False, color=(2 / 255, 62 / 255, 1), )
    plt_oneplot([flipthres(lT(step_error_list)[4], 3), flipthres(lT(step_error_list)[5], 3),
                 # flipthres(lT(step_error_list)[6], 3), flipthres(lT(step_error_list)[7], 3),],
                 flipthres(lT(step_error_list)[6], 3), ],
                'Manipulator joint angles', xlabel='step', ylabel='Orientation Error, rad',
                save_dir=os.path.join(dirs['step_ori_err'],
                                      'StepOrientationError_OnePlot_{}_scene{}.png'.format(basename, episode_num)),
                legend=['Pitch', 'Roll', 'Yaw', 'End-effector', ], figsize=(5, 4), settitle=False, lw=[1.5, 1.5, 1.5],
                color=[(232 / 255, 0, 11 / 255), (26 / 255, 201 / 255, 56 / 255), (2 / 255, 62 / 255, 1),
                       (0.15, 0.15, 0.15)],
                zoom_range_y=(-0.1, 0.1), )
    if base_orientation_plot:
        tmp = step_base_ori_list
        # for i in range(len(tmp)):
        #     tmp[i].append(R.from_euler('xyz', tmp[i]).magnitude())

        plt_multiplot(lT(tmp), title='Base satellite orientation', save_dir=os.path.join(dirs['base_ori'],
                                                                                         'StepBaseOrientation_{}_scene{}.png'.format(
                                                                                             basename, episode_num)),
                      figsize=(5, 3.5),
                      color=[(232 / 255, 64 / 255, 139 / 255), (120 / 255, 201 / 255, 92 / 255),
                             (66 / 255, 126 / 255, 1), (0.15, 0.15, 0.15)],
                      label=['Pitch', 'Roll', 'Yaw', 'Base spacecraft'], linestyle=['-', '-', '-', '-'],
                      lw=[1.5, 1.5, 1.5, 2],
                      t=0.97, b=0.12)

        tmp = step_base_pos_list
        # for i in range(len(tmp)):
        #     tmp[i].append((tmp[i][0]**2 + tmp[i][1]**2 + tmp[i][2]**2) ** 0.5)
        plt_multiplot(lT(tmp), ylabel='Position, m', title='Base satellite position',
                      save_dir=os.path.join(dirs['base_ori'],
                                            'StepBasePosition_{}_scene{}.png'.format(basename, episode_num)),
                      figsize=(5, 4),
                      color=[(232 / 255, 0, 11 / 255), (26 / 255, 201 / 255, 56 / 255), (2 / 255, 62 / 255, 1),
                             (0.15, 0.15, 0.15)],
                      label=['x', 'y', 'z', 'Base spacecraft'], linestyle=['-', '-', '-', '-'], lw=[1, 1, 1, 2])


def evaluation_episode_plot(args, dirs, mean_rewards_list, stable_rewards_list, final_rewards_list,
                            mean_errors_list, stable_errors_list, final_errors_list):
    plt_plot(stable_rewards_list, title='Stable Reward', ylabel='Stable Reward',
             save_dir=os.path.join(dirs['eval_episode_reward_err'], 'StableReward.png'))

    pos_err_start_num = 0
    pos_err_end_num = 3

    plt_combineplot(
        lst_l=stable_errors_list[pos_err_start_num:pos_err_end_num],
        lst_r=stable_errors_list[pos_err_end_num],
        title_l='StablePositionError', title_r='StablePositionError',
        ylabel_l=['X-Error', 'Y-Error', 'Z-Error', ], ylabel_r='StablePositionError',
        save_dir=os.path.join(dirs['eval_episode_pos_err'], 'StablePositionError.png'))

    ori_err_start_num = 4
    ori_err_end_num = 7
    plt_combineplot(
        lst_l=stable_errors_list[ori_err_start_num:ori_err_end_num],
        lst_r=stable_errors_list[ori_err_end_num],
        title_l='StableOrientationError', title_r='StableOrientationError',
        ylabel_l=['Pitch-Error', 'Roll-Error', 'Yaw-Error'], ylabel_r='StableOrientationError',
        save_dir=os.path.join(dirs['eval_episode_ori_err'], 'StableOrientationError.png'))


def write_to_csv(csv_file, *lists):
    with open(csv_file, 'w+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 使用 zip 函数并解包每个列表内的子列表
        for data in zip(*lists):
            writer.writerow([item for sublist in data for item in sublist])
