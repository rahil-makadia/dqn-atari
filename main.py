!nvidia-smi
import time
import torch
import gym
import numpy as np
import matplotlib.pyplot as plt

from DQN import neural_net, DQN, lr, MEMORY_SIZE
from atari_wrappers import modify_env
from replay_memory import ReplayMemory

if __name__ == '__main__':
    # create environment
    DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
    env = gym.make(DEFAULT_ENV_NAME)
    env = modify_env(env)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create networks
    # if the file exists, load the model
    if os.path.exists("dqn_pong_model"):
        policy_net = torch.load("dqn_pong_model")
    else:
        policy_net = neural_net(n_actions=4).to(device)
    target_net = neural_net(n_actions=4).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # setup optimizer
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)

    # initialize replay memory
    memory = ReplayMemory(MEMORY_SIZE)

    # train model
    model = DQN(policy_net, target_net, optimizer, memory, device, n_actions=4)
    model.train(env, 3000)
    torch.save(model.policy_net, "dqn_pong_model")

    # test model
    policy_net = torch.load("dqn_pong_model")
    model.test(env, 1, policy_net, render=False)

    # plot results
    model.plot()
    model.plot(logarithmic=True)

    # save text file of log
    n_steps = len(model.log['steps_done'])
    array = np.zeros((n_steps, 5))
    array[:, 0] = model.log['steps_done']
    array[:, 1] = model.log['episode']
    array[:, 2] = model.log['reward']
    array[:, 3] = model.log['total_reward']
    array[:, 4] = model.log['loss']
    np.savetxt('dqn_pong_model.csv', array, delimiter=',', header='steps_done,episode,reward,total_reward,loss')