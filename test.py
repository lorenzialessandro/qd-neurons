import numpy as np
import torch

import gymnasium as gym

from network import NCHL, Neuron

data = np.load('results/qd_mountaincar_20250403-103431/best_network_params.npy', allow_pickle=True)
nodes = [2, 4, 3]
pop = []

for i, node in enumerate(data):
    neuron = Neuron(node["neuron_id"])
    neuron.set_params(node["params"])
    pop.append(neuron)
    
net = NCHL(nodes, population=pop)

env = gym.make("MountainCar-v0", render_mode="human")
state, info = env.reset()
done = False
while not done:
    input = torch.tensor(state).double()
    output = net.forward(input)
    action = np.argmax(output.tolist())
    state, reward, done, truncated, _ = env.step(action)
    env.render()
    
    