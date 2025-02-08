import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.hidden_structure = args.hidden_structure
        self.joint_max_vel = args.joint_max_vel
        self.device = torch.device('cuda', args.gpu_idx) if 0 <= args.gpu_idx <= torch.cuda.device_count() \
            else torch.device('cpu')

        self.actor = nn.Sequential()
        self.actor.add_module('L1', nn.Linear(self.state_dim, self.hidden_structure[0]))
        self.actor.add_module('T1', nn.Tanh())

        for i in range(len(self.hidden_structure) - 1):
            self.hidden_layer = nn.Linear(self.hidden_structure[i], self.hidden_structure[i + 1])
            self.actor.add_module('L{}'.format(i + 2), self.hidden_layer)

            self.actor.add_module('T{}'.format(i + 2), nn.Tanh())

        self.out_layer =  nn.Linear(self.hidden_structure[-1], self.action_dim)
        self.actor.add_module('L_out', self.out_layer)

        self.actor.add_module('T_out', nn.Tanh())

    def forward(self, state):
        out = self.actor(state.to(self.device))
        mean = self.joint_max_vel * out

        return mean

class Critic(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self, args):
        super(Critic, self).__init__()
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.hidden_structure = args.hidden_structure

        self.critic = nn.Sequential()
        self.layer = nn.Linear(self.state_dim+self.action_dim, self.hidden_structure[0])
        self.critic.add_module('L1', self.layer)

        self.critic.add_module('T1', nn.Tanh())

        for i in range(len(self.hidden_structure)-1):
            self.layer = nn.Linear(self.hidden_structure[i], self.hidden_structure[i+1])
            self.critic.add_module('L{}'.format(i+2), self.layer)

            self.critic.add_module('T{}'.format(i+2), nn.Tanh())

        self.layer = nn.Linear(self.hidden_structure[-1], 1)
        self.critic.add_module('Le', self.layer)

    def forward(self, state, action):
        out = self.critic(torch.cat([state, action], dim=1))
        return out
