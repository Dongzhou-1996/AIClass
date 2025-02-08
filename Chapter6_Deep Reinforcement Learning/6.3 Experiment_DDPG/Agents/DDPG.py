import torch
import torch.nn as nn
import sys
sys.path.append('..')
from Models.DDPG_actor_critic import Actor, Critic
import os
import numpy as np
import copy

class DDPG(object):
    def __init__(self, args):
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.joint_max_vel = args.joint_max_vel
        self.batch_size = args.batch_size  # batch size
        self.max_train_steps = args.max_train_steps
        self.gamma = args.gamma  # discount factor
        self.tau = args.tau # Softly update the target network
        self.lr_a = args.lr_a  # actor learning rate
        self.lr_c = args.lr_c  # critic learning rate
        self.device = torch.device('cuda', args.gpu_idx) if 0 <= args.gpu_idx <= torch.cuda.device_count() \
            else torch.device('cpu')

        lr_schedulers = ['const', 'cyclic', 'singlecyclic']
        self.lr_scheduler = args.lr_scheduler
        assert (self.lr_scheduler in lr_schedulers)
        self.lr_cycle = args.lr_cycle

        self.lr_ac_minrate = args.lr_ac_minrate
        self.lr_minrate = args.lr_minrate

        self.lr_cycle = int(args.save_batch_size * self.lr_cycle / self.batch_size)

        if self.lr_cycle % 2:
            self.cyclic_array = np.linspace(1, self.lr_ac_minrate, int((self.lr_cycle + 1) / 2))
            self.cyclic_array = np.hstack(
                (self.cyclic_array, np.linspace(self.cyclic_array[-2], 1, int((self.lr_cycle - 1) / 2))))
        else:
            self.cyclic_array = np.linspace(1, self.lr_ac_minrate, int((self.lr_cycle) / 2))
            self.cyclic_array = np.hstack(
                (self.cyclic_array, np.linspace(self.cyclic_array[-1], 1, int(self.lr_cycle / 2))))
        self.singlecyclic_array = np.linspace(1, self.lr_ac_minrate, self.lr_cycle)

        self.actor = Actor(args).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(args).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

        self.MseLoss = nn.MSELoss()

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a = self.actor(s).data.cpu().detach().numpy().flatten()
        return a

    def update(self, replay_buffer, total_steps, update_time):
        batch_s, batch_a, batch_r, batch_s_, batch_dw = replay_buffer.sample(self.batch_size, self.device)  # Sample a batch

        # learning rate scheduler
        # region
        if update_time % self.lr_cycle == 0:
            tmp = 1 - total_steps * (1 - self.lr_minrate) / self.max_train_steps
            self.singlecyclic_array = np.linspace(tmp, self.lr_ac_minrate, self.lr_cycle)
            if self.lr_cycle % 2:
                self.cyclic_array = np.linspace(tmp, self.lr_ac_minrate, int((self.lr_cycle + 1) / 2))
                self.cyclic_array = np.hstack(
                    (self.cyclic_array, np.linspace(self.cyclic_array[-2], tmp, int((self.lr_cycle - 1) / 2))))
            else:
                self.cyclic_array = np.linspace(tmp, self.lr_ac_minrate, int((self.lr_cycle) / 2))
                self.cyclic_array = np.hstack(
                    (self.cyclic_array, np.linspace(self.cyclic_array[-1], tmp, int(self.lr_cycle / 2))))

        lr_rate = 1
        if self.lr_scheduler == 'const':
            lr_a_now = self.lr_a
            lr_c_now = self.lr_c
        elif self.lr_scheduler == 'cyclic':
            lr_a_now = self.lr_a * self.cyclic_array[update_time % self.lr_cycle]
            lr_c_now = self.lr_c * self.cyclic_array[update_time % self.lr_cycle]
            lr_rate = 1 - self.lr_minrate
        elif self.lr_scheduler == 'singlecyclic':
            lr_a_now = self.lr_a * self.singlecyclic_array[update_time % self.lr_cycle]
            lr_c_now = self.lr_c * self.singlecyclic_array[update_time % self.lr_cycle]
            lr_rate = 1 - self.lr_minrate

        if self.lr_scheduler != 'singlecyclic' and self.lr_scheduler != 'cyclic':
            lr_a_now = lr_a_now * (1 - lr_rate * total_steps / self.max_train_steps)
            lr_c_now = lr_c_now * (1 - lr_rate * total_steps / self.max_train_steps)

        for p in self.optimizer_actor.param_groups:
            p['lr'] = np.max([lr_a_now, self.lr_a * self.lr_ac_minrate])
        for p in self.optimizer_critic.param_groups:
            p['lr'] = np.max([lr_c_now, self.lr_c * self.lr_ac_minrate])
        # endregion

        # Compute the target Q
        with torch.no_grad():  # target_Q has no gradient
            Q_ = self.critic_target(batch_s_, self.actor_target(batch_s_))
            target_Q = batch_r + self.gamma * (1 - batch_dw) * Q_

        # Compute the current Q and the critic loss
        current_Q = self.critic(batch_s, batch_a)
        critic_loss = self.MseLoss(target_Q, current_Q)
        # Optimize the critic
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()
        self.critic_loss_save = critic_loss.detach().cpu().tolist()

        # Freeze critic networks so you don't waste computational effort
        for params in self.critic.parameters():
            params.requires_grad = False

        # Compute the actor loss
        actor_loss = -self.critic(batch_s, self.actor(batch_s)).mean()
        # Optimize the actor
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
        self.actor_loss_save = actor_loss.detach().cpu().tolist()

        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True

        # Softly update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save_model(self, episode_num, total_steps, state_norm, reward_scaling, reach_fail_prob, update_time,
                   model_path=''):
        print('=> saving network to {} ...'.format(model_path))
        checkpoint = {'episode_num': episode_num,
                      'total_steps': total_steps,
                      'actor_net': self.actor.state_dict(),
                      'critic_net': self.critic.state_dict(),
                      'state_norm': state_norm,
                      'reward_scaling': reward_scaling,
                      'reach_fail_prob': reach_fail_prob,
                      'update_time': update_time,
                      }
        torch.save(checkpoint, model_path, _use_new_zipfile_serialization=False)
        print('=> model params is saved!')
        return

    def reload_model(self, model_path=''):
        if os.path.exists(model_path):
            print('=> reloading model from {} ...'.format(model_path))
            checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage.cpu())
            episode_num = checkpoint['episode_num']
            total_steps = checkpoint['total_steps']
            self.actor.load_state_dict(checkpoint['actor_net'])
            self.critic.load_state_dict(checkpoint['critic_net'])
            state_norm = checkpoint['state_norm']
            reward_scaling = checkpoint['reward_scaling']
            reach_fail_prob = checkpoint['reach_fail_prob']
            update_time = checkpoint['update_time']
        else:
            raise ValueError('No model file is found in {}'.format(model_path))
        return episode_num, total_steps, state_norm, reward_scaling, reach_fail_prob, update_time
