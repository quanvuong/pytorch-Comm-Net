import argparse
import random
import sys

import torch
from torch.optim import Adam
from torch.autograd import Variable

from environment import get_agents_pool, draw_agents, get_rewards
from policy import net_constructor
from wrappers import ZeroTensorVar

parser = argparse.ArgumentParser()

parser.add_argument('--num_levers', type=int, default=5)

parser.add_argument('--num_agents', type=int, default=500)

parser.add_argument('--num_agents_drawn', type=int, default=5)

parser.add_argument('--num_comm_step', type=int, default=2)

parser.add_argument('--hidden_layer_size', type=int, default=128)

parser.add_argument('--num_trials', type=int, default=500)

parser.add_argument('--num_training_batch', type=int, default=50000)

parser.add_argument('--batch_size', type=int, default=64)

parser.add_argument('--use_cuda', type=int, default=int(torch.cuda.is_available()))

parser.add_argument('--baseline_discount', type=float, default=0.03)


def main(args):

    agents_pool = get_agents_pool(args)
    policy = net_constructor(args)
    optim = Adam(policy.parameters())

    for batch_i in range(args.num_training_batch):

        # Get args.batch_size samples and get their respective rewards
        samples = []

        for batch_size_i in range(args.batch_size):
            agents_with_idxes = draw_agents(agents_pool, args.num_agents_drawn)

            samples.append(agents_with_idxes)

        rewards = Variable(torch.FloatTensor(args.num_agents_drawn, 1))
        sums_log_probs = Variable(torch.FloatTensor(args.num_agents_drawn, 1))
        baselines = Variable(torch.FloatTensor(args.num_agents_drawn, 1))

        for idx, sample in enumerate(samples):
            r, p, b = get_rewards(policy, sample, args)
            sums_log_probs[idx] = p
            rewards[idx] = r
            baselines[idx] = b

        # Update the policies
        optim.zero_grad()
        update_target = sums_log_probs * (baselines - rewards) \
                        + args.baseline_discount * (rewards - baselines) ** 2 / args.num_agents_drawn
        update_target.backward()
        optim.step()




if __name__ == '__main__':

    random.seed(1)
    torch.manual_seed(1)

    args = parser.parse_args()
    main(args)
