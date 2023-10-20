
import numpy as np
from .auto_encoder import AutoEncoder, VariationalAutoEncoder
from .molecule_gnn_model import GNN, GNN_graphpred
from .schnet import SchNet

import torch
import torch.nn as nn
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import (global_add_pool, global_max_pool, global_mean_pool)


class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        h = torch.matmul(summary, self.weight)
        return torch.sum(x*h, dim=1)



class AuxiliaryNet(nn.Module):
    def __init__(self, args, psi, molecule_model):
        super().__init__()
        self.molecule_model = molecule_model
        self.emb_dim = args.emb_dim
        self.psi = psi

        # Different kind of graph pooling
        if args.graph_pooling == "sum":
            self.pool = global_add_pool
        elif args.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif args.graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")

        # generate label head
        self.aux_classifier = nn.Sequential(
            nn.Linear(self.emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, int(np.sum(self.psi)))
        )

    def from_pretrained(self, model_file, device):
        self.molecule_model.load_state_dict(torch.load(model_file, map_location=device))
        return

    def mask_softmax(self, x, mask, dim=1):
        z = x.max(dim=dim)[0]
        x = x - z.reshape(-1, 1)
        logits = torch.exp(x) * mask / (torch.sum(torch.exp(x) * mask, dim=dim, keepdim=True) + 1e-7)
        return logits

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, \
                                              data.edge_attr, data.batch
        y = ((data.y.reshape(-1,len(self.psi))+1)//2).long()
        node_representation = self.molecule_model(x, edge_index, edge_attr)
        graph_representation = self.pool(node_representation, batch)
        output = self.aux_classifier(graph_representation)

        # build a binary mask by psi, we add epsilon=1e-8 to avoid nans
        index = torch.zeros([len(self.psi), np.sum(self.psi)], device=output.device) + 1e-8
        for i in range(len(self.psi)):
            index[i, int(np.sum(self.psi[:i])):np.sum(self.psi[:i + 1])] = 1
        mask = index[y]

        label_pred = self.mask_softmax(output, mask, dim=1)
        return label_pred 