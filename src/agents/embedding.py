import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class NNSetEmbedding(nn.Module):
    def __init__(self, input_size):
        super(NNSetEmbedding, self).__init__()

        self.leader_state_size = 5                  # previously it was 2
        self.one_pedestrian_state_size = 5          # 2 coordinates + 3 binary pointers (catch, exit_zone, saved)
        self.hidden_size = (input_size // 5) * 2
        self.output_size = 2                        # input_size - self.leader_state_size

        self.embedding = nn.Sequential(
            nn.Linear(2, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, inputs):

        batch_flag = 0

        if len(inputs.shape) == 1:  # for one state without batch
            inputs = inputs.unsqueeze(0)
            batch_flag = 1

        batch_size = inputs.size(0)
        inputs = inputs.float().unsqueeze(1)

        input_leader = inputs[:, 0, :self.leader_state_size]

        input_pedestrians = inputs[:, 0, self.leader_state_size:].reshape([batch_size, -1, self.one_pedestrian_state_size])

        # print('all\n', input_pedestrians)

        x_pedestrians = input_pedestrians[:, :, 0]
        y_pedestrians = input_pedestrians[:, :, 1]
        catch_status = input_pedestrians[:, :, 3]

        x_catched = torch.mul(x_pedestrians, catch_status)
        y_catched = torch.mul(y_pedestrians, catch_status)

        splited_pedestrians = torch.cat((x_catched.unsqueeze(2), y_catched.unsqueeze(2)), 2)

        output_pedestrians = self.embedding(splited_pedestrians)
        # output_pedestrians = self.embedding(input_pedestrians)

        # print('out_net\n', output_pedestrians)

        output_pedestrians = output_pedestrians.sum(axis=1)

        # print('out_sum\n', output_pedestrians)
        
        merged_output = torch.cat((input_leader, output_pedestrians), 1)
        embedded = merged_output

        if batch_flag == 1:
            embedded = torch.squeeze(embedded)

        return embedded