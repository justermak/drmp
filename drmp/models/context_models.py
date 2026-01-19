import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, n_layers=2, act='relu'):
        super().__init__()
        layers = []
        in_d = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_d, hidden_dim))
            if act == 'relu': layers.append(nn.ReLU())
            elif act == 'mish': layers.append(nn.Mish())
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ContextModelStartGoal(nn.Module):
    def __init__(self, state_dim, out_dim=64, hidden_dim=64, n_layers=2):
        super().__init__()
        # Input is Concatenated (Start, Goal) -> 2 * state_dim
        self.input_dim = state_dim * 2
        self.net = MLP(self.input_dim, out_dim, hidden_dim=hidden_dim, n_layers=n_layers)

    def forward(self, start_state, goal_state):
        # Concatenate start and goal
        # Expected shapes: (Batch, Dim)
        x = torch.cat([start_state, goal_state], dim=-1)
        return self.net(x)
