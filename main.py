import argparse
import random
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--num_levers', type=int, default=5)

parser.add_argument('--num_agents', type=int, default=500)

parser.add_argument('--num_agents_drawn', type=int, default=5)

parser.add_argument('--num_comm_step', type=int, default=2)

parser.add_argument('--hidden_state_size', type=int, default=128)

parser.add_argument('--num_trials', type=int, default=500)

parser.add_argument('--num_training_batch', type=int, default=50000)

parser.add_argument('--batch_size', type=int, default=64)


def main(args):
    pass

if __name__ == '__main__':

    random.seed(1)
    torch.manual_seed(1)

    args = parser.parse_args()
    main(args)
