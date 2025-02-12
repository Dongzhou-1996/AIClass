import os
import cv2
import random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter

# Hyper-parameters
seed = 42
best_model_path = './DQN/best_model.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_episodes = 2000
max_episode_len = 1000
env = gym.make('CartPole-v1', render_mode='rgb_array')
num_state = env.observation_space.shape[0]
num_action = env.action_space.n
torch.manual_seed(seed)

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])


class Memory(object):
    def __init__(self, replay_memory_size=10000):
        self.storage = []
        self.total_count = 0
        self.max_size = replay_memory_size

    def push(self, data: Transition):
        if len(self.storage) >= self.max_size:
            self.storage.pop(0)
            self.total_count -= 1
        self.storage.append(data)
        self.total_count += 1

    def sample(self, batch_size):
        samples = random.sample(self.storage, batch_size)
        states, actions, rewards, n_states, dones = map(np.array, zip(*samples))
        return states, actions, rewards, n_states, dones

    def clear(self):
        self.storage.clear()
        self.total_count = 0

    def __len__(self):
        return len(self.storage)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_state, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_value = self.fc3(x)
        return action_value


class DQN(object):
    capacity = 2000
    learning_rate = 1e-3
    memory_count = 0
    batch_size = 64
    gamma = 0.99
    update_count = 0
    start_epsilon = 0.9
    end_epsilon = 0.1

    def __init__(self, mode: str):
        super(DQN, self).__init__()
        self.mode = mode
        self.target_net, self.eval_net = Net(), Net()
        self.target_net.to(device)
        self.eval_net.to(device)
        self.memory = Memory(replay_memory_size=self.capacity)
        self.optimizer = optim.Adam(self.eval_net.parameters(), self.learning_rate)
        self.loss_func = nn.MSELoss()
        self.writer = SummaryWriter('./DQN/logs')

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        value = self.eval_net(state.to(device))
        action_max_value, index = torch.max(value, 1)
        action = index.item()
        if self.mode == 'train':
            epsilon = max(self.end_epsilon, self.start_epsilon - self.update_count * 1e-5)
            self.writer.add_scalar('epsilon', epsilon, self.update_count)
            if np.random.rand(1) >= epsilon:  # epslion greedy
                action = np.random.choice(range(num_action), 1).item()
        return action

    def store_transition(self, transition):
        self.memory.push(transition)
        return

    def update(self):
        # reward = (reward - reward.mean()) / (reward.std() + 1e-7)
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state = torch.FloatTensor(state).to(device)
        action = torch.LongTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).to(device)

        q_values = self.eval_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_state).max(1)[0]
        target_q_values = reward + (self.gamma * next_q_values * (1 - done))
        loss = self.loss_func(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.writer.add_scalar('loss/value_loss', loss.item(), self.update_count)
        self.update_count += 1
        if self.update_count % 1000 == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

    def mode_transfer(self, mode: str):
        self.mode = mode
        if mode == 'train':
            self.target_net.train()
            self.eval_net.train()
        else:
            self.target_net.eval()
            self.eval_net.eval()

    def save_model(self, model_path=""):
        print('Saving network to {} ...'.format(model_path))
        checkpoint = {
            'steps': self.update_count,
            'qnet': self.eval_net.state_dict(),
            'target_qnet': self.target_net.state_dict(),
        }
        torch.save(checkpoint, model_path, _use_new_zipfile_serialization=False)
        print('Model params is saved!')
        return

    def reload_model(self, model_path='', mode='train'):
        if os.path.exists(model_path):
            print('Restoring model from {} ...'.format(model_path))
            checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage.cpu())
            self.eval_net.load_state_dict(checkpoint['qnet'])
            self.target_net.load_state_dict(checkpoint['target_qnet'])
            self.update_count = checkpoint['steps'] + 1
        else:
            raise ValueError('No model file is found in {}'.format(model_path))
        self.mode_transfer(mode)


def test(agent: DQN, episode_num=10, episode=0, render=True):
    agent.mode_transfer('test')
    episode_lens = []
    episode_rewards = []
    for i_ep in range(episode_num):
        r = 0
        state, _ = env.reset(seed=seed)
        if render:
            img = env.render()
            cv2.imshow('window', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey()
        for t in range(10000):
            action = agent.select_action(state)
            next_state, reward, done, info, _ = env.step(action)
            if render:
                img = env.render()
                cv2.imshow('window', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                cv2.waitKey(10)
            r += reward
            state = next_state
            if done or t >= max_episode_len:
                episode_lens.append(t)
                episode_rewards.append(r)
                if render:
                    cv2.destroyWindow('window')
                break
    print('[Test]: Episode {}: Average length: {}, Average reward: {}'.format(
        episode, np.mean(episode_lens), np.mean(episode_rewards)))
    agent.writer.add_scalar('Test/Episode_Len', np.mean(episode_lens), agent.update_count)
    agent.writer.add_scalar('Test/Episode_Reward', np.mean(episode_rewards), agent.update_count)
    return np.mean(episode_lens), np.mean(episode_rewards)


def initialization(agent: DQN):
    agent.mode_transfer('train')
    print('=> start collect experience ...')
    while agent.memory.total_count < 1000:
        state, _ = env.reset(seed=seed)
        for t in range(10000):
            action = agent.select_action(state)
            next_state, reward, done, info, _ = env.step(action)
            transition = Transition(state, action, reward, next_state, done)
            agent.store_transition(transition)
            state = next_state
            if done or t >= max_episode_len:
                break
    print('=> experience has been collected!')
    return


def train(agent: DQN, render=False):
    agent.mode_transfer('train')
    best_reward = 0
    for ep in range(num_episodes):
        state, _ = env.reset(seed=seed)
        r = 0
        for t in range(10000):
            action = agent.select_action(state)
            next_state, reward, done, info, _ = env.step(action)
            r += reward
            transition = Transition(state, action, reward, next_state, done)
            agent.store_transition(transition)
            agent.update()
            state = next_state
            if done or t >= max_episode_len:
                agent.writer.add_scalar('Train/Episode_Len', t + 1, global_step=ep)
                if ep % 50 == 0:
                    ep_len, ep_reward = test(agent, episode_num=10, episode=ep, render=render)
                    if best_reward <= ep_reward:
                        best_reward = ep_reward
                        agent.save_model(best_model_path)

                    agent.mode_transfer('train')
                    print("[Train]: episodes {}, episode length {}, episode reward: {}".format(ep, t, r))
                break
    return


def main():
    agent = DQN(mode='train')
    # initialization(agent)
    # train(agent, render=False)
    agent.reload_model(model_path=best_model_path, mode='test')
    test(agent, episode_num=20, episode=0, render=True)


if __name__ == '__main__':
    main()
