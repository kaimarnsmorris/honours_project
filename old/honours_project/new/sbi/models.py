import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32, n_layers=2):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        return self.nn(x)


class DeepSets(nn.Module):
    def __init__(self, input_dim, output_dim, summary_dim=32, phi_width=32,
                 rho_width=32, n_layers=2):
        super().__init__()
        self.phi = MLP(input_dim, summary_dim, phi_width, n_layers)
        self.rho = MLP(summary_dim, output_dim, rho_width, n_layers)

    def alter_inputs(self, x):
        return x

    def alter_outputs(self, x):
        return x

    def forward(self, x):
        x = self.alter_inputs(x)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = x.float()
        phi_x = self.phi(x)
        sum_phi_x = torch.sum(phi_x, dim=1)
        return self.alter_outputs(self.rho(sum_phi_x))


class GRUEstimator(nn.Module):
    def __init__(self, output_dim=2, hidden_dim=64, num_layers=2,
                 mlp_hidden=128, mlp_depth=3):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_dim,
                          num_layers=num_layers, batch_first=True)
        layers = []
        layers.append(nn.Linear(hidden_dim, mlp_hidden))
        layers.append(nn.ReLU())
        for _ in range(mlp_depth - 1):
            layers.append(nn.Linear(mlp_hidden, mlp_hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(mlp_hidden, output_dim))
        self.mlp = nn.Sequential(*layers)

    def alter_outputs(self, x):
        return x

    def forward(self, x):
        x = x.unsqueeze(-1)
        gru_out, _ = self.gru(x)
        pooled = gru_out.mean(dim=1)
        return self.alter_outputs(self.mlp(pooled))
