import torch
import torch.nn as nn
import torch.optim as optim
import torch.random as random
import torchdiffeq
import random
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class LiquidODEFunc(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.log_tau = nn.Parameter(torch.randn(hidden_size))
        self.input_map = nn.Linear(input_size, hidden_size)
        self.recurrent_map = nn.Linear(hidden_size, hidden_size)
        nn.init.normal_(self.recurrent_map.weight, mean=0.0, std=0.1)
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, t, h):
        tau = torch.exp(self.log_tau).clamp(min=0.1, max=10)
        input_term = self.input_map(self._current_input(t))
        dhdt = (-h + torch.sigmoid(self.recurrent_map(h)) + input_term) / tau
        return self.norm(dhdt)
    
    def set_input(self, input_func):
        self._current_input = input_func

class EmotionLNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.odefunc = LiquidODEFunc(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x, t_span=None):
        batch_size, seq_len, num_features = x.size()
        if t_span is None:
            t_span = torch.linspace(0.0, seq_len, seq_len + 1, device=x.device)  # Explicit time points
        else:
            # Ensure t_span has len(seq_len) + 1 points to match sequence steps
            t_span = torch.linspace(t_span[0], t_span[-1], seq_len + 1, device=x.device)
        
        def input_func(t):
            t_idx = torch.clamp(t.long(), 0, seq_len - 1)
            return x[:, t_idx, :].squeeze(1)
    
        self.odefunc.set_input(input_func)
        h0 = torch.zeros(batch_size, self.hidden_size, device=x.device)
        # Use fixed step size to avoid adaptive step issues
        solution = torchdiffeq.odeint(
            self.odefunc,
            h0,
            t_span,
            method='rk4',
            atol=1e-4,
            rtol=1e-4,
            options={'step_size': 1.0}  # Fixed step size to match seq_len
        )
        h_final = solution[-1]
        return self.fc(h_final), h_final




class EmotionDataGenerator:
    """Generates synthetic emotion data with temporal continuity"""
    @staticmethod
    def create_base_emotion(noise_range=0.1):
        """Create base emotion vector with smooth transitions"""
        emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        base = {e: torch.rand(1).item() * 0.9 + 0.1 for e in emotions} 
        return [base[e] + random.uniform(-noise_range, noise_range) for e in emotions]

    @classmethod
    def generate_sequences(cls, num_samples, seq_length=55, num_features=6):
        """
        Generate sequential emotion data with temporal patterns
        Returns: (num_samples, seq_length, num_features) tensor
        """
        # Preallocate numpy array for efficiency
        sequences = np.zeros((num_samples, seq_length, num_features), dtype=np.float32)
        
        for i in range(num_samples):
            # Generate initial base emotion with noise
            base = np.array(cls.create_base_emotion(), dtype=np.float32)
            sequences[i, 0] = np.clip(base, 0.1, 1)
            
            # Generate subsequent steps with smooth transitions
            for j in range(1, seq_length):
                noise = np.random.uniform(-0.2, 0.2, size=num_features)
                new_step = sequences[i, j-1] + noise
                sequences[i, j] = np.clip(new_step, 0, 1)
        
        return torch.from_numpy(sequences)

class EmotionTrainer:
    def __init__(self, config=None):
        '''

            seq_length -> num sequence of input vector data, 
                          where predicted vector[-1] is taken as last sequence
        
        '''
        self.config = config or {
            'seq_length': 5,
            'num_features': 6,
            'hidden_size': 64,
            'lr': 0.001,
            'weight_decay': 1e-5
        }
        
    def prepare_data(self, num_samples=1000):
        full_data = EmotionDataGenerator.generate_sequences(
            num_samples, self.config['seq_length'])
        X = full_data[:, :-1]
        y = full_data[:, -1]
        split = int(0.8 * len(X))
        return (X[:split], y[:split]), (X[split:], y[split:])

    def train(self):
        (train_X, train_y), (val_X, val_y) = self.prepare_data()
        
        model = EmotionLNN(
            input_size=self.config['num_features'],
            hidden_size=self.config['hidden_size'],
            output_size=self.config['num_features']
        )
        
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), 
                              lr=self.config['lr'],
                              weight_decay=self.config['weight_decay'])

        train_loader = DataLoader(TensorDataset(train_X, train_y),
                                batch_size=16, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_X, val_y),
                              batch_size=16)

        print("Training Liquid Neural Network with ODE solver...")

        num_epoch = 26

        for epoch in range(num_epoch):
            model.train()
            train_loss = 0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs, _ = model(inputs)
                    val_loss += criterion(outputs, targets).item()
            
            print(f"Epoch {epoch+1}/{num_epoch} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

        return model
    


if __name__ == "__main__":
    """Run main program"""

    '''
    
        Generate synthetic past vector data
    
    '''
    trainer = EmotionTrainer()
    model = trainer.train()
    
    '''
    
        Generate synthetic input sequence
    
    '''
    test_sequence = EmotionDataGenerator.generate_sequences(1, trainer.config['seq_length'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Auto-detect device
    model.to(device)  # Move model to device
    
    with torch.no_grad():
        model.eval()
        test_sequence = test_sequence.to(device)  # Move input to same device as model
        prediction, _ = model(test_sequence[:, :-1])  # Forward pass
        
        print("\nSample prediction:")
        print("Input sequence:", test_sequence[0, :-1].cpu().numpy())  # Move back to CPU for printing
        print("Predicted next step:", prediction[0].cpu().numpy()) 