!nvidia-smi
import time
import torch
import gym
import matplotlib.pyplot as plt

from DQN import neural_net, DQN, lr, MEMORY_SIZE
from atari_wrappers import modify_env
from replay_memory import ReplayMemory

def env_test_render(env):
    # run random policy and render
    env.reset()
    for _ in range(1000):
        fig = plt.figure()
        arr = env.render(mode='rgb_array')
        plt.imshow(arr)
        plt.show()
        env.step(env.action_space.sample())
    env.close()
    return

if __name__ == '__main__':
    # create environment
    DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
    env = gym.make(DEFAULT_ENV_NAME)
    env = modify_env(env)
    # env_test_render(env)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create networks
    policy_net = neural_net(n_actions=4).to(device)
    target_net = neural_net(n_actions=4).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # setup optimizer
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)

    # initialize replay memory
    memory = ReplayMemory(MEMORY_SIZE)
    
    # train model
    model = DQN(policy_net, target_net, optimizer, memory, device, n_actions=4)
    model.train(env, 20)
    # model.plot()
    torch.save(model.policy_net, "dqn_pong_model")

    # test model
    policy_net = torch.load("dqn_pong_model")
    model.test(env, 1, policy_net, render=False)
