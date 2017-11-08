import random

import numpy as np

import torch
from torch.autograd import Variable


def get_agents_pool(args):

    agents_pool = torch.FloatTensor(args.num_agents, args.hidden_layer_size)

    agents_pool.uniform_(-args.num_agents, args.num_agents)

    return agents_pool


def draw_agents(agents_pool, num_agents):

    ids = random.sample(range(0, len(agents_pool)), num_agents)

    agents = []

    for id in ids:
        agent = agents_pool[id]
        agents.append((id, agent))

    return agents


def get_rewards(policy, sample, args):
    agents_initial_hidden_states = Variable(torch.FloatTensor(len(sample), args.hidden_layer_size))

    for idx, agent in enumerate(sample):
        # [1] to skip the agent id
        hidden_state = agent[1]
        agents_initial_hidden_states[idx].data.copy_(hidden_state)

    agents_actions, sum_log_probs, baseline = policy(agents_initial_hidden_states)

    reward = len(np.unique(agents_actions.data.numpy()))

    return reward, sum_log_probs, baseline
