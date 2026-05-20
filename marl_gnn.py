import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN_Q_Network(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN_Q_Network, self).__init__()

        self.fc_self = nn.Linear(input_dim, hidden_dim)
        self.fc_neighbor = nn.Linear(input_dim, hidden_dim)

        self.fc_policy = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, state, adj):

        h_self = F.relu(self.fc_self(state))

        neighbor_state = torch.matmul(adj, state) 
        h_neighbor = F.relu(self.fc_neighbor(neighbor_state))
        
        h_combined = torch.cat([h_self, h_neighbor], dim=1)
        

        q_values = self.fc_policy(h_combined)
        return q_values