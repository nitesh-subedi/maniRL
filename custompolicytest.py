import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


# Replay buffer to store experience tuples
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = zip(*batch)
        return state, action, reward, next_state

    def __len__(self):
        return len(self.buffer)


# Neural network for the policy (actor)
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        std = torch.exp(log_std)
        return mean, std


# Neural network for the Q-function (critic)
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.relu(self.fc1(torch.cat([state, action], 1)))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Neural network for the value function (critic)
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# SAC algorithm
class SAC:
    def __init__(self, state_dim, action_dim):
        self.policy_net = PolicyNetwork(state_dim, action_dim).cuda()
        self.q_net1 = QNetwork(state_dim, action_dim).cuda()
        self.q_net2 = QNetwork(state_dim, action_dim).cuda()
        self.value_net = ValueNetwork(state_dim).cuda()
        self.target_value_net = ValueNetwork(state_dim).cuda()
        self.target_value_net.load_state_dict(self.value_net.state_dict())

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=3e-4)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=3e-4)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=3e-4)

        self.replay_buffer = ReplayBuffer(capacity=1000000)
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2

    def select_action(self, state):
        if type(state) is np.ndarray:
            state_flattened = np.concatenate(
                [s.flatten() if isinstance(s, np.ndarray) else np.array([s]) for s in state])
            state_tensor = torch.FloatTensor(state_flattened).unsqueeze(0).cuda()
        else:
            state_flattened = np.concatenate(
                [s.flatten() if isinstance(s, np.ndarray) else np.array([s]) for s in state[0]])
            state_tensor = torch.FloatTensor(state_flattened).unsqueeze(0).cuda()
        mean, std = self.policy_net(state_tensor)
        normal = Normal(mean, std)
        action = normal.sample()
        action = torch.tanh(action)
        return action.cpu().detach().numpy()[0]

    def update(self, batch_size):
        state, action, reward, next_state = self.replay_buffer.sample(batch_size)
        print(f"State from update {len(state)}")
        state = torch.FloatTensor(state).cuda()
        action = torch.FloatTensor(action).cuda()
        reward = torch.FloatTensor(reward).unsqueeze(1).cuda()
        next_state = torch.FloatTensor(next_state).cuda()

        # Update Q-networks
        with torch.no_grad():
            next_action, next_log_prob = self.policy_net(next_state)
            next_q_value = torch.min(
                self.q_net1(next_state, next_action),
                self.q_net2(next_state, next_action)
            ) - self.alpha * next_log_prob
            target_q_value = reward + self.gamma * next_q_value

        q1_value = self.q_net1(state, action)
        q2_value = self.q_net2(state, action)
        q_loss1 = torch.nn.functional.mse_loss(q1_value, target_q_value)
        q_loss2 = torch.nn.functional.mse_loss(q2_value, target_q_value)

        self.q_optimizer1.zero_grad()
        q_loss1.backward()
        self.q_optimizer1.step()

        self.q_optimizer2.zero_grad()
        q_loss2.backward()
        self.q_optimizer2.step()

        # Update value network
        with torch.no_grad():
            new_action, log_prob = self.policy_net(state)
            target_value = torch.min(
                self.q_net1(state, new_action),
                self.q_net2(state, new_action)
            ) - self.alpha * log_prob

        value = self.value_net(state)
        value_loss = torch.nn.functional.mse_loss(value, target_value)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Update policy network
        new_action, log_prob = self.policy_net(state)
        q_value = torch.min(
            self.q_net1(state, new_action),
            self.q_net2(state, new_action)
        )
        policy_loss = (self.alpha * log_prob - q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update target value network
        for param, target_param in zip(self.value_net.parameters(), self.target_value_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def train(self, env, num_steps, batch_size):
        state = env.reset()
        for step in range(num_steps):
            action = self.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            self.replay_buffer.push(state, action, reward, next_state)
            state = next_state

            if done:
                state = env.reset()

            if len(self.replay_buffer) > batch_size:
                self.update(batch_size)

            # if step % 1000 == 0:
            #     print(f"Step: {step}")


# Example usage with a Gym environment
import gymnasium as gym
import random

env = gym.make("Pendulum-v1")
sac = SAC(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
sac.train(env, num_steps=100000, batch_size=64)
