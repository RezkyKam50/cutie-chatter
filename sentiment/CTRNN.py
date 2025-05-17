import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from torchdiffeq import odeint


class Ephireus(nn.Module):
    def __init__(self, hidden_dim, skip_type=None, gated=None):
        super().__init__()
        self.skip_type = skip_type
        self.gated = gated
        self.linear = nn.Sequential(
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh()
        )
        if skip_type == 'concat':
            self.proj = nn.Linear(2*hidden_dim, hidden_dim)

        if gated:
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid()
            )
        self.activation = nn.Tanh()

    def forward(self, t, h):
        out = self.activation(self.linear(h))
        if self.skip_type == 'add':
            out = h + out
        elif self.skip_type == 'concat':
            out = torch.cat([h, out], dim=-1)
            out = self.proj(out)
        if self.gated:
            gate = self.gate(h)
            out = gate * out + (1 - gate) * h
        return out


class Sephora(nn.Module):
    def __init__(self, recurrent_path, tau):
        super().__init__()
        self.recurrence = recurrent_path
        self.tau = tau  

    def forward(self, t, h):
        tau = self.tau.unsqueeze(0)  
        return (-h + self.recurrence(t, h)) / tau   


class Perseus(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(hidden_dim, elementwise_affine=True),
        )
        self.recurrent_block = Ephireus(hidden_dim, skip_type='concat', gated=True)
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim, elementwise_affine=True),
            nn.Linear(hidden_dim, output_dim)
        )
        self.tau = nn.Parameter(torch.ones(hidden_dim))   
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                torch.nn.init.orthogonal_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)  
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                torch.nn.init.orthogonal_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)   


    def forward(self, x, timestamps, return_hidden_states=True):
        """
        Args:
            x: [batch_size, input_dim]
            timestamps: [batch_size, 1] ex. 64 batches = 64 row : project smooth derivatives to t+1
            Note : No sequence, the dataset is only for global pattern recognition
        """
        h0 = self.encoder(x)  
        taus = F.softplus(self.tau) + 1e-5
        ode_func = Sephora(recurrent_path=self.recurrent_block, tau=taus)

        epsilon = 1e-5   
        timestamps = timestamps + torch.cumsum(torch.full_like(timestamps, epsilon), dim=0)

        sol = odeint(
            ode_func, 
            h0, 
            timestamps.squeeze(1), 
            method='dopri5'
        ) 

        h_t = sol[-1]  
        output = self.decoder(h_t)

        return (output, h_t) if return_hidden_states else output
