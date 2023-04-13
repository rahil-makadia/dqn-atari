import numpy as np
import matplotlib.pyplot as plt

import itertools
import torch
import torch.nn as nn
import torch.optim as optim

# write the DQN algorithm here following Mnih et al. (2015) for the following cases:
# With replay, with target Q (i.e., the standard algorithm).
# With replay, without target Q (i.e., the target network is reset after each step).
# Without replay, with target Q (i.e., the size of the replay memory buffer is equal to the size of each minibatch).
# Without replay, without target Q (i.e., the target network is reset after each step and the size of the replay memory buffer is equal to the size of each minibatch).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

class DQN:
    def __init__(self, env, gamma, epsilon, epsilon_min, learning_rate, batch_size, replay_size, init_replay_size, target_update, savefig, target_Q, replay, verbose=False):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.target_update = target_update
        self.target_Q = target_Q
        self.target_counter = 0
        num_hidden = 64
        self.num_hidden = num_hidden
        self.replay = replay
        self.replay_buffer = []
        self.init_replay_size = init_replay_size
        self.num_actions = env.num_actions
        self.num_states = env.num_states
        self.savefig = savefig
        if not self.target_Q:
            self.target_update = 1
        if not self.replay:
            self.replay_size = self.batch_size
        if self.target_Q and self.replay:
            self.save_dir = 'yes_target_yes_replay'
        elif self.target_Q:
            self.save_dir = 'yes_target_no_replay'
        elif self.replay:
            self.save_dir = 'no_target_yes_replay'
        else:
            self.save_dir = 'no_target_no_replay'
        self.verbose = verbose
        if self.replay:
            self.initialize_replay_buffer()
        self.build_model()
        return None

    def initialize_replay_buffer(self):
        while len(self.replay_buffer) < self.init_replay_size:
            s = self.env.reset()
            done = False
            while not done:
                a = np.random.randint(self.env.num_actions)
                (s_, r, done) = self.env.step(a)
                self.replay_buffer.append((s, a, r, s_, done))
                s = s_
        return None

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.env.num_states, self.num_hidden),
            nn.Tanh(),
            nn.Linear(self.num_hidden, self.num_hidden),
            nn.Tanh(),
            nn.Linear(self.num_hidden, self.env.num_actions)
        )
        model.to(device)
        self.model = model
        self.target_model = model
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        return None

    def act(self, state):
        val = np.random.random()
        if val < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.model(torch.from_numpy(state).float()).detach().numpy())

    def observe(self, state, action, reward, next_state, done):
        # if the replay buffer is full, replace the oldest transition
        # if the replay buffer is not full, add the transition to the end of the replay buffer
        buffer = (state, action, reward, next_state, done)
        if len(self.replay_buffer) >= self.replay_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append(buffer)
        return None

    def replay_step(self):
        batch_idx = np.random.choice(len(self.replay_buffer), self.batch_size)
        batch = [self.replay_buffer[i] for i in batch_idx]
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])
        for i in range(self.batch_size):
            q_value_for_all_actions = self.model(torch.FloatTensor(states[i]))
            q_value = q_value_for_all_actions[actions[i]]
            next_q_value = self.target_model(torch.FloatTensor(next_states[i])).detach()
            target_q_value = rewards[i] + (1-dones[i]) * self.gamma * torch.max(next_q_value)
            loss = self.loss_fn(q_value, target_q_value)
            self.optimizer.zero_grad()
            loss.backward()
            # clamp the gradient
            for param in self.model.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
        return None

    def check_target_update(self):
        if self.target_counter % self.target_update == 0:
            if self.verbose: print(f'updating target model at step {self.target_counter}')
            self.target_model.load_state_dict(self.model.state_dict())
        return None

    def _run_one_episode(self):
        s = self.env.reset()
        done = False
        iter_count = 0
        log = {
            't': [0],
            's': [s],
            'a': [0],
            'r': [0],
            'theta': [self.env.x[0]],
            'thetadot': [self.env.x[1]],
        }
        G = 0
        while not done:
            a = self.act(s)
            s_, r, done = self.env.step(a)
            self.observe(s, a, r, s_, done)
            self.replay_step()
            self.target_counter += 1
            G += r*self.gamma**iter_count
            iter_count += 1
            self.check_target_update()
            self.epsilon = max(self.epsilon - 0.9/1e4, self.epsilon_min)
            s = s_
            log['t'].append(log['t'][-1] + 1*self.env.dt)
            log['s'].append(s)
            log['a'].append(a)
            log['r'].append(r)
            log['theta'].append(self.env.x[0])
            log['thetadot'].append(self.env.x[1])
        log['G'] = G
        tau = [self.env._a_to_u(a) for a in log['a']]
        log['tau'] = tau
        value_func = [torch.max(self.model(torch.FloatTensor(s))).item() for s in log['s']]
        log['value_func'] = value_func
        self.G.append(log['G'])
        self.all_theta += log['theta']
        self.all_thetadot += log['thetadot']
        self.all_tau += log['tau']
        self.all_value_func += log['value_func']
        if self.savefig:
            self.plot(log)
        return log

    def run(self, num_episodes):
        self.num_episodes = num_episodes
        self.episode_list = [1]
        self.G = []
        self.all_theta = []
        self.all_thetadot = []
        self.all_tau = []
        self.all_value_func = []
        for i in range(num_episodes):
            log = self._run_one_episode()
            if self.verbose: print(f'Episode {i+1}, Return: {log["G"]}, Epsilon: {self.epsilon}')
            self.episode_list.append(i+2)
        return None

    # def plot_contour(self):
    #     figsize = 6
    #     size = 500
    #     theta_vec = np.linspace(-np.pi, np.pi, size)
    #     thetadot_vec = np.linspace(-self.env.max_thetadot, self.env.max_thetadot, size)
    #     theta_grid, thetadot_grid = np.meshgrid(theta_vec, thetadot_vec)
    #     policy_arr = np.zeros((size, size))
    #     value_func_arr = np.zeros((size, size))
    #     for i, j in itertools.product(range(size), range(size)):
    #         s = np.array([theta_grid[i, j], thetadot_grid[i, j]])
    #         predict = self.model(torch.FloatTensor(s)).detach().numpy()
    #         policy_arr[i, j] = self.env._a_to_u(np.argmax(predict))
    #         value_func_arr[i, j] = np.max(predict)
    #     plt.figure(figsize=(figsize, figsize), dpi=150)
    #     plt.contourf(theta_grid, thetadot_grid, policy_arr, cmap='viridis', levels=50)
    #     plt.colorbar(label=r'$\tau$')
    #     plt.xlim(-np.pi, np.pi)
    #     plt.ylim(-self.env.max_thetadot, self.env.max_thetadot)
    #     plt.xlabel(r'$\theta$')
    #     plt.ylabel(r'$\dot{\theta}$')
    #     plt.title('Policy')
    #     plt.savefig(f'./figures/{self.save_dir}/ctr_policy_{self.num_episodes}_{self.target_update}.png', dpi=300, bbox_inches='tight')
    #     plt.close()
    #     plt.figure(figsize=(figsize, figsize), dpi=150)
    #     plt.contourf(theta_grid, thetadot_grid, value_func_arr, cmap='viridis', levels=50)
    #     plt.colorbar(label='Value Function')
    #     plt.xlim(-np.pi, np.pi)
    #     plt.ylim(-self.env.max_thetadot, self.env.max_thetadot)
    #     plt.xlabel(r'$\theta$')
    #     plt.ylabel(r'$\dot{\theta}$')
    #     plt.title('Value Function')
    #     plt.savefig(f'./figures/{self.save_dir}/ctr_value_func_{self.num_episodes}_{self.target_update}.png', dpi=300, bbox_inches='tight')
    #     plt.close()
    #     return None

    # def plot(self, log=None):
    #     def wrap_pi(x): return ((x + np.pi) % (2 * np.pi)) - np.pi
    #     def wrap_2pi(x): return x % (2 * np.pi)
    #     wrap_func = wrap_pi
    #     if self.savefig and max(self.episode_list) == self.num_episodes:
    #         self.plot_contour()
    #     # save animation
    #     if log is not None and self.savefig and max(self.episode_list) == self.num_episodes:
    #         policy_lambda = lambda s: np.argmax(self.model(torch.FloatTensor(s)).detach().numpy())
    #         self.env.video(policy_lambda, f'./figures/{self.save_dir}/animation_{self.num_episodes}_{self.target_update}.gif', writer='pillow')
    #     figsize = 6
    #     # plot the return and n-episode moving average
    #     n_avg = 10
    #     moving_avg = np.convolve(self.G, np.ones((n_avg,))/n_avg, mode='valid')
    #     plt.figure(figsize=(figsize, figsize), dpi=150)
    #     plt.plot(self.episode_list, self.G, 'o', label='Return', alpha=0.3)
    #     if max(self.episode_list) > n_avg:
    #         plt.plot(self.episode_list[n_avg-1:], moving_avg, label=f'{n_avg}-episode moving average')
    #     plt.xlabel('Episode #')
    #     plt.ylabel('Return')
    #     plt.legend()
    #     plt.title('Return vs. Episode #')
    #     plt.savefig(f'./figures/{self.save_dir}/return_{self.num_episodes}_{self.target_update}.png', dpi=300, bbox_inches='tight')
    #     plt.close()
    #     # plot the trajectory
    #     log['theta'] = [wrap_func(x) for x in log['theta']]
    #     plt.figure(figsize=(figsize, figsize), dpi=150)
    #     plt.plot(log['t'], log['theta'], label=r'$\theta$')
    #     plt.axhline(-np.pi, color='r', linestyle='--')
    #     plt.axhline(np.pi, color='r', linestyle='--', label=r'$\theta=\pm\pi$')
    #     plt.axhline(-0.1*np.pi, color='g', linestyle='--')
    #     plt.axhline(0.1*np.pi, color='g', linestyle='--', label=r'$\theta=\pm0.1\pi$')
    #     plt.plot(log['t'], log['thetadot'], label=r'$\dot{\theta}$')
    #     plt.xlabel('t')
    #     plt.ylabel('theta / thetadot')
    #     plt.legend()
    #     plt.title('Pendulum State vs. Time')
    #     plt.savefig(f'./figures/{self.save_dir}/state_{self.num_episodes}_{self.target_update}.png', dpi=300, bbox_inches='tight')
    #     plt.close()
    #     # plot the policy
    #     plt.figure(figsize=(figsize, figsize), dpi=150)
    #     plt.scatter(self.all_theta, self.all_thetadot, c=self.all_tau, cmap='viridis')
    #     plt.colorbar(label=r'$\tau$')
    #     plt.xlim(-np.pi, np.pi)
    #     plt.ylim(-self.env.max_thetadot, self.env.max_thetadot)
    #     plt.xlabel(r'$\theta$')
    #     plt.ylabel(r'$\dot{\theta}$')
    #     plt.title('Policy')
    #     plt.savefig(f'./figures/{self.save_dir}/policy_{self.num_episodes}_{self.target_update}.png', dpi=300, bbox_inches='tight')
    #     plt.close()
    #     # plot the value function
    #     plt.figure(figsize=(figsize, figsize), dpi=150)
    #     plt.scatter(self.all_theta, self.all_thetadot, c=self.all_value_func, cmap='viridis')
    #     plt.colorbar(label='Value Function')
    #     plt.xlim(-np.pi, np.pi)
    #     plt.ylim(-self.env.max_thetadot, self.env.max_thetadot)
    #     plt.xlabel(r'$\theta$')
    #     plt.ylabel(r'$\dot{\theta}$')
    #     plt.title('Value Function')
    #     plt.savefig(f'./figures/{self.save_dir}/value_func_{self.num_episodes}_{self.target_update}.png', dpi=300, bbox_inches='tight')
    #     plt.close()
    #     return None
