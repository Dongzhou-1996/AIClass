import cv2
import gym
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

env_name = "CartPole-v1"
env = gym.make(env_name, render_mode='rgb_array')  # a single env
plt.ion()


class ActorCritic(object):
    def __init__(self, input_dim, output_dim, hidden_size, mode='train'):

        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ).to(device)

        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
            nn.Softmax(dim=1),
        ).to(device)

        self.step = 0
        self.mode = mode
        self.log_dir = './A2C/logs'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def select_action(self, x):
        v = self.critic(x)
        p = self.actor(x)
        d = Categorical(p)
        return d, v

    def save_model(self, model_path=""):
        print('Saving network to {} ...'.format(model_path))
        checkpoint = {
            'steps': self.step,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }
        torch.save(checkpoint, model_path, _use_new_zipfile_serialization=False)
        print('Model params is saved!')
        return

    def reload_model(self, model_path='', mode='train'):
        if os.path.exists(model_path):
            print('Restoring model from {} ...'.format(model_path))
            checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage.cpu())
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.step = checkpoint['steps'] + 1
        else:
            raise ValueError('No model file is found in {}'.format(model_path))
        self.mode_transfer(mode)

    def mode_transfer(self, mode: str):
        self.mode = mode
        if mode == 'train':
            self.actor.train()
            self.critic.train()
        else:
            self.actor.eval()
            self.critic.eval()


def test_env(agent: ActorCritic, vis=False):
    agent.mode_transfer('test')
    s, _ = env.reset()
    if vis:
        img = env.render()
        cv2.imshow('windows', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
    d = False
    total_reward = 0
    while not d:
        s = torch.FloatTensor(s).unsqueeze(0).to(device)
        dist, _ = agent.select_action(s)
        n_s, r, d, _, _ = env.step(dist.sample().cpu().numpy()[0])
        s = n_s
        if vis:
            img = env.render()
            cv2.imshow('windows', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        total_reward += r
        if total_reward > 1000:
            break

    if vis: cv2.destroyWindow('windows')
    agent.mode_transfer('train')
    return total_reward


def plot(idx, r):
    plt.plot(r, 'b-')
    plt.title('frame %s. reward: %s' % (idx, r[-1]))
    plt.pause(0.0001)


def train(agent: ActorCritic, global_steps=10000):
    actor_optimizer = optim.Adam(agent.actor.parameters(), lr)
    critic_optimizer = optim.Adam(agent.critic.parameters(), lr)

    step = 0
    test_rewards = []
    best_reward = 0
    entropy = 0
    state, _ = env.reset()
    while step < global_steps:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, value = agent.select_action(state)
        action = dist.sample()

        next_state, reward, done, _, _ = env.step(action.cpu().numpy()[0])
        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()

        state = next_state
        step += 1

        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        _, next_value = agent.select_action(next_state)

        reward = torch.FloatTensor([reward]).to(device)
        done = torch.LongTensor([done]).to(device)
        target_value = reward + 0.99 * next_value * (1 - done)
        advantage = target_value.detach() - value

        actor_loss = -(log_prob * advantage.detach()).mean() - 0.001 * entropy.detach()
        critic_loss = 0.5 * advantage.pow(2).mean()

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        agent.step += 1

        if done or step % 1000 == 0:
            state, _ = env.reset()
            entropy = 0

        if step % 500 == 0:
            ep_reward = np.mean([test_env(agent) for _ in range(10)])
            print("[Train]: Step {}, episode reward: {}".format(step, ep_reward))
            if best_reward <= ep_reward:
                best_reward = ep_reward
                agent.save_model(best_model_path)
            test_rewards.append(ep_reward)
            plot(step, test_rewards)

    return


if __name__ == '__main__':
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.n

    # Hyper params:
    global_steps = 20000
    hidden_size = 16
    lr = 1e-3
    best_model_path = './A2C/best_model.pth'
    agent = ActorCritic(num_inputs, num_outputs, hidden_size, mode='train')

    train(agent, global_steps)
    plt.savefig('./A2C/train_curve.jpg', dpi=300)
    agent.reload_model(best_model_path)
    rewards = []
    for _ in range(20):
        episode_reward = test_env(agent, True)
        rewards.append(episode_reward)
    print('[Test]: Episode reward: {}'.format(np.mean(rewards)))