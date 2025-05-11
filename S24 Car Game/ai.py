# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import time

# Creating the architecture of the Neural Network


class Network(nn.Module):

    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, 30)
        self.fc4 = nn.Linear(30, nb_action)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_values = self.fc4(x)
        return q_values


# Implementing Experience Replay


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


# Implementing Deep Q Learning


class Dqn:

    def __init__(self, input_size, nb_action, gamma, training_mode: bool = True):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        self.last_save_time = time.time()
        self.save_interval = 60  # Save every 5 minutes
        self.epsilon = 0.6  # Exploration rate
        self.batch_size = 100
        self.learning_rate = 0.001
        self.training_mode = training_mode

        # Initialize target network for stable learning
        self.target_network = Network(input_size, nb_action)
        self.target_network.load_state_dict(self.model.state_dict())
        self.target_update_frequency = 1000
        self.steps = 0

    def select_action(self, state):
        if (random.random() < self.epsilon) and self.training_mode:
            # Exploration: random action
            return random.randint(0, self.model.nb_action - 1)
        else:
            # Exploitation: best action
            with torch.no_grad():
                probs = F.softmax(self.model(state) * 100, dim=1)
            action = probs.multinomial(1)
            return action.data[0, 0]

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = (
            self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        )
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_graph=True)
        self.optimizer.step()

    def update(self, reward, new_signal):
        # Ensure new_signal includes distance to goal
        if len(new_signal) != self.model.input_size:
            raise ValueError(
                f"Expected input size {self.model.input_size}, got {len(new_signal)}"
            )

        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push(
            (
                self.last_state,
                new_state,
                torch.LongTensor([int(self.last_action)]),
                torch.Tensor([self.last_reward]),
            )
        )
        action = self.select_action(new_state)

        # Learn from experience if we have enough samples
        if len(self.memory.memory) > self.batch_size and self.training_mode:
            batch_state, batch_next_state, batch_action, batch_reward = (
                self.memory.sample(self.batch_size)
            )
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)

        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]

        # Periodic saving
        if self.training_mode:
            current_time = time.time()
            if current_time - self.last_save_time > self.save_interval:
                self.save()
                self.last_save_time = current_time

        return action

    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1.0)

    def save(self):
        save_path = "last_brain.pth"
        backup_path = "last_brain_backup.pth"

        # Create backup of previous save if it exists
        if os.path.exists(save_path):
            os.replace(save_path, backup_path)

        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "target_state_dict": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "reward_window": self.reward_window,
                "steps": self.steps,
            },
            save_path,
        )

    def load(self):
        if os.path.isfile("last_brain.pth"):
            print("=> loading checkpoint... ")
            checkpoint = torch.load("last_brain.pth")
            self.model.load_state_dict(checkpoint["state_dict"])
            self.target_network.load_state_dict(
                checkpoint.get("target_state_dict", checkpoint["state_dict"])
            )
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.epsilon = checkpoint.get("epsilon", 1.0)
            self.reward_window = checkpoint.get("reward_window", [])
            self.steps = checkpoint.get("steps", 0)
            print("done !")
        else:
            print("no checkpoint found...")
