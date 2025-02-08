import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import sys
sys.path.append('..')
from Models.PPO_actor_critic import Actor_Gaussian, Critic
import os
import numpy as np

class PPO_continuous():
    def __init__(self, args):
        self.joint_max_vel = args.joint_max_vel
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip

        lr_schedulers = ['const', 'cyclic', 'singlecyclic',]
        self.lr_scheduler = args.lr_scheduler
        assert (self.lr_scheduler in lr_schedulers)
        self.lr_cycle = args.lr_cycle

        self.lr_ac_minrate = args.lr_ac_minrate
        self.lr_minrate = args.lr_minrate

        if self.lr_cycle % 2:
            self.cyclic_array = np.linspace(1, self.lr_ac_minrate, int((self.lr_cycle + 1) / 2))
            self.cyclic_array = np.hstack((self.cyclic_array, np.linspace(self.cyclic_array[-2], 1, int((self.lr_cycle - 1) / 2))))
        else:
            self.cyclic_array = np.linspace(1, self.lr_ac_minrate, int((self.lr_cycle) / 2))
            self.cyclic_array = np.hstack((self.cyclic_array, np.linspace(self.cyclic_array[-1], 1, int(self.lr_cycle / 2))))
        self.singlecyclic_array = np.linspace(1, self.lr_ac_minrate, self.lr_cycle)

        self.use_adv_norm = args.use_adv_norm
        self.gpu_idx = args.gpu_idx

        self.device = torch.device('cuda', self.gpu_idx) if 0 <= self.gpu_idx <= torch.cuda.device_count() \
            else torch.device('cpu')
        print(self.device)
        self.is_train = args.is_train

        self.actor = Actor_Gaussian(args).to(self.device)
        self.critic = Critic(args).to(self.device)

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def evaluate(self, s):  # When evaluating the policy, we only use the mean
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a = self.actor(s).detach().numpy().flatten()
        return a

    def choose_action(self, s):
        # 确保输入张量在GPU上
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)

        with torch.no_grad():
            # 确保模型在GPU上
            self.actor.to(self.device)
            if self.is_train:
                dist = self.actor.get_dist(s)
                a = dist.sample()  # 根据概率分布采样动作
                a = torch.clamp(a, -self.joint_max_vel, self.joint_max_vel)  # [-max,max]
                a_logprob = dist.log_prob(a)  # 动作的对数概率密度
            else:
                a = self.actor(s)
                a = torch.clamp(a, -self.joint_max_vel, self.joint_max_vel)  # [-max,max]

        # 在返回前将张量从GPU移到CPU，并转换为NumPy数组
        if self.device.type == 'cpu':
            if self.is_train:
                return a.numpy().flatten(), a_logprob.numpy().flatten()
            else:
                return a.flatten(), 1
        else:
            if self.is_train:
                return a.detach().cpu().numpy().flatten(), a_logprob.detach().cpu().numpy().flatten()
            else:
                return a.detach().cpu().numpy().flatten(), 1

    def get_logprob(self, s, a):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)
        a = torch.tensor(a, dtype=torch.float).to(self.device)

        with torch.no_grad():
            # 确保模型在GPU上
            self.actor.to(self.device)
            dist = self.actor.get_dist(s)
            a = torch.clamp(a, -self.joint_max_vel, self.joint_max_vel)  # [-max,max]
            a_logprob = dist.log_prob(a)  # 动作的对数概率密度
        # 在返回前将张量从GPU移到CPU，并转换为NumPy数组
        if self.device.type == 'cpu':
            return a_logprob.numpy().flatten()
        else:
            return a_logprob.detach().cpu().numpy().flatten()

    def update(self, replay_buffer, total_steps, update_time):
        self.noa = 1
        s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor()  # Get training data
        '''
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        '''
        s = s.to(self.device)
        a = a.to(self.device)
        a_logprob = a_logprob.to(self.device)
        r = r.to(self.device)
        s_ = s_.to(self.device)
        dw = dw.to(self.device)
        done = done.to(self.device)

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

        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs     # TD Error
            for delta, d in zip(reversed(deltas.cpu().flatten().numpy()), reversed(done.cpu().flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1).to(self.device)
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size * self.noa)), self.mini_batch_size, False):
                dist_now = self.actor.get_dist(s[index])
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index])
                # a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob[index].sum(1, keepdim=True))  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # Trick 5: policy entropy
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                self.actor_loss_save = actor_loss.mean().detach().cpu().tolist()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                self.critic_loss_save = critic_loss.detach().cpu().tolist()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

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
