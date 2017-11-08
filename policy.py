import torch
import torch.nn as nn
from torch.autograd import Variable

from wrappers import ByteTensorVar, ZeroTensorVar


def net_constructor(args):

    # input size = h size + c size + h size (again because of skip connection)
    input_size = args.hidden_layer_size * 3
    output_size = args.num_levers

    class Policy(nn.Module):

        def __init__(self):
            super().__init__()

            self.f_linear1 = nn.Linear(input_size, args.hidden_layer_size)
            self.relu = nn.ReLU()
            self.f_linear2 = nn.Linear(args.hidden_layer_size, args.hidden_layer_size)

            self.baseline_head = nn.Linear(args.hidden_layer_size, 1)

            self.decoder = nn.Linear(args.hidden_layer_size, output_size)
            self.softmax = nn.Softmax()

        def get_comm_vectors(self, hidden_states):

            num_agents = len(hidden_states)

            vectors = Variable(torch.FloatTensor(num_agents, args.hidden_layer_size))

            for agent_i in range(num_agents):
                mask = torch.ones(num_agents, args.hidden_layer_size)
                mask[agent_i].copy_(torch.zeros(args.hidden_layer_size))
                mask = mask.byte()
                mask = ByteTensorVar(mask)
                masked_out = hidden_states[mask].view(-1, args.hidden_layer_size)
                comm_v = torch.sum(masked_out, 0) / (num_agents - 1)
                vectors[agent_i] = comm_v

            return vectors

        def get_hidden_states(self, prev_hidden_states, comm_vectors, initial_hidden_states):

            num_agents = len(initial_hidden_states)

            net_input = Variable(torch.FloatTensor(num_agents, input_size))

            for agent_i in range(num_agents):
                one_agent_input = torch.cat((prev_hidden_states[agent_i],
                                            comm_vectors[agent_i], initial_hidden_states[agent_i]))
                net_input[agent_i] = one_agent_input

            output = self.f_linear2(self.relu(self.f_linear1(net_input)))

            return output

        def forward(self, initial_hidden_states):

            hidden_states = initial_hidden_states.clone()

            for comm_step in range(args.num_comm_step):
                comm_vectors = self.get_comm_vectors(hidden_states)
                hidden_states = self.get_hidden_states(hidden_states, comm_vectors, initial_hidden_states)

            decoded = self.decoder(hidden_states)
            baseline = self.baseline_head(hidden_states)

            # print(decoded)
            # assert False

            softmaxed = self.softmax(decoded)

            # print(softmaxed)
            # assert False
            acts = torch.multinomial(softmaxed, num_samples=1)[:, ]

            sum_log_probs = ZeroTensorVar(1)

            for idx, act in enumerate(acts.data.view(args.num_agents_drawn)):
                sum_log_probs = sum_log_probs + softmaxed[idx][act]

            print(baseline)

            return acts, sum_log_probs, baseline

    net = Policy()
    return net
