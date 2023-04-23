import numpy as np
import random
import math
import time
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import count

from replay_memory import Transition

# hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.02
EPS_DECAY = 1000000
TARGET_UPDATE = 10000
RENDER = False
lr = 1e-4
INITIAL_MEMORY = 10000
MEMORY_SIZE = 10 * INITIAL_MEMORY

class neural_net(nn.Module):
    def __init__(self, in_channels=4, n_actions=14):
        """
        Initialize Deep Q Network

        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(neural_net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, n_actions)
        
    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)

class DQN:
    def __init__(self, policy_net, target_net, optimizer, memory, device, n_channels=4, n_actions=14):
        self.n_channels = n_channels
        self.n_actions = n_actions
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.memory = memory
        self.device = device

        self.steps_done = 0
        self.log = {'steps_done': [],
                    'episode': [],
                    'reward': [],
                    'total_reward': [],
                    'loss': []
        }
        return None

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END)* \
                math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample <= eps_threshold:
            return torch.tensor([[random.randrange(4)]], device=self.device, dtype=torch.long)
        with torch.no_grad():
            return self.policy_net(state.to(self.device)).max(1)[1].view(1,1)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        """
        zip(*transitions) unzips the transitions into
        Transition(*) creates new named tuple
        batch.state - tuple of all the states (each state is a tensor)
        batch.next_state - tuple of all the next states (each state is a tensor)
        batch.reward - tuple of all the rewards (each reward is a float)
        batch.action - tuple of all the actions (each action is an int)    
        """
        batch = Transition(*zip(*transitions))
        
        actions = tuple((map(lambda a: torch.tensor([[a]], device=self.device), batch.action))) 
        rewards = tuple((map(lambda r: torch.tensor([r], device=self.device), batch.reward))) 

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device, dtype=torch.bool)
        
        non_final_next_states = torch.cat(  [s for s in batch.next_state
                                            if s is not None]).to(self.device)
        

        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(actions)
        reward_batch = torch.cat(rewards)
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.detach().cpu().numpy().item()

    def get_state(self, obs):
        state = np.array(obs)
        state = state.transpose((2, 0, 1))
        state = torch.from_numpy(state)
        return state.unsqueeze(0)

    def train(self, env, n_episodes, render=False):
        for episode in range(n_episodes):
            obs, info = env.reset()
            state = self.get_state(obs)
            total_reward = 0.0
            for t in count():
                action = self.select_action(state)

                if render:
                    env.render()

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                total_reward += reward

                next_state = None if done else self.get_state(obs)
                reward = torch.tensor([reward], device=self.device)

                self.memory.push(state, action.to('cpu'), next_state, reward.to('cpu'))
                state = next_state

                if self.steps_done > INITIAL_MEMORY:
                    loss = self.optimize_model()
                    self.log['steps_done'].append(self.steps_done)
                    self.log['episode'].append(episode)
                    self.log['reward'].append(reward.to('cpu').numpy().item())
                    self.log['total_reward'].append(total_reward)
                    self.log['loss'].append(loss)
                    if self.steps_done % TARGET_UPDATE == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())

                if done:
                    break
            if episode % 1 == 0:
                print(
                    f'Total steps: {self.steps_done} \t Episode: {episode}/{t+1} \t Total reward: {total_reward}'
                )
        env.close()
        return

    def test(self, env, n_episodes, policy=None, render=True, name='env_screen.png'):
        for episode in range(n_episodes):
            obs, info = env.reset()
            state = self.get_state(obs)
            total_reward = 0.0
            for _ in count():
                if policy is not None:
                    action = policy(state.to(self.device)).max(1)[1].view(1,1)
                else:
                    action = env.action_space.sample()

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                total_reward += reward

                next_state = None if done else self.get_state(obs)
                state = next_state

                if done:
                    print(f"Finished Episode {episode} with reward {total_reward}")
                    break
        env.reset()
        env.close()
        return
    
    def plot(self, logarithmic=False):  # sourcery skip: extract-duplicate-method
        import matplotlib.pyplot as plt
        log = self.log
        fig, ax = plt.subplots(1, 3, figsize=(20, 5))
        ax[0].plot(log['episode'], log['total_reward'], '.')
        ax[0].set_title('Total Reward')
        ax[0].set_xlabel('Episode #')
        ax[1].plot(log['episode'], log['reward'], '.')
        ax[1].set_title('Reward')
        ax[1].set_xlabel('Episode #')
        ax[2].plot(log['episode'], log['loss'], '.')
        ax[2].set_title('Loss')
        ax[2].set_xlabel('Episode #')
        if logarithmic:
            ax[2].set_yscale('log')
        plt.savefig(f'./figures/training_per_episode_log_{logarithmic}.png', dpi=300, bbox_inches='tight')
        plt.show()

        fig, ax = plt.subplots(1, 3, figsize=(20, 5))
        ax[0].plot(log['steps_done'], log['total_reward'], '.')
        ax[0].set_title('Total Reward')
        ax[0].set_xlabel('Steps Done')
        ax[1].plot(log['steps_done'], log['reward'], '.')
        ax[1].set_title('Reward')
        ax[1].set_xlabel('Steps Done')
        ax[2].plot(log['steps_done'], log['loss'], '.')
        ax[2].set_title('Loss')
        ax[2].set_xlabel('Steps Done')
        if logarithmic:
            ax[0].set_xscale('log')
            ax[1].set_xscale('log')
            ax[2].set_xscale('log')
            ax[2].set_yscale('log')
        plt.savefig(f'./figures/training_per_step_log_{logarithmic}.png', dpi=300, bbox_inches='tight')
        plt.show()
        return
