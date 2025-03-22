import torch
import torch.nn as nn
import torch.optim as optim
import torch.random as random
import torchdiffeq
import random
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


'''

This code is under development for Modeling Affective States where emotional response factor is determined by Tau parameter

Model : Liquid Time Constant by Ramin Hasani, MIT CSAIL

'''

class ODE(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.log_tau = nn.Parameter(torch.randn(hidden_size))  
        self.input_map = nn.Linear(input_size, hidden_size)
        self.recurrent_map = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, t, h):
        tau = torch.nn.functional.softplus(self.log_tau) + 0.1  
        input_term = self.input_map(self._current_input(t))
        recurrent_term = torch.sigmoid(self.gate(h)) * torch.tanh(self.recurrent_map(h))
        dhdt = (-h + recurrent_term + input_term) / tau

        return self.norm(dhdt)

    def set_input(self, input_func):
        self._current_input = input_func

class EmoMODEL(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.odefunc = ODE(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x, t_span=None):
        batch_size, seq_len, num_features = x.size()
        if t_span is None:
            t_span = torch.linspace(0.0, seq_len, seq_len + 1, device=x.device)  
        else:
            t_span = torch.linspace(t_span[0], t_span[-1], seq_len + 1, device=x.device)

        def input_func(t):
            t_idx = torch.clamp(t.long(), 0, seq_len - 1)
            return x[:, t_idx, :].squeeze(1)

        self.odefunc.set_input(input_func)
        
        h0 = torch.zeros(batch_size, self.hidden_size, device=x.device)
        solution = torchdiffeq.odeint(
            self.odefunc, 
            h0, 
            t_span, 
            method='rk4', 
            atol=1e-4, 
            rtol=1e-4, 
            options={'step_size': 1.0}
        )

        h_final = solution[-1]
        return self.fc(h_final), h_final
    
class EmotionDataGenerator:
    @staticmethod
    def create_base_emotion(noise_range=0.01):
        emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        base = {e: random.uniform(0.1, 0.99) for e in emotions}   
        return [np.clip(base[e] + random.uniform(-noise_range, noise_range), 0.1, 0.99) for e in emotions]

    @classmethod
    def generate_sequences(cls, num_samples, seq_length, num_features=6):
            sequences = np.zeros((num_samples, seq_length, num_features), dtype=np.float32)
            
            for i in range(num_samples):
                base = np.array(cls.create_base_emotion(), dtype=np.float32)
                sequences[i, 0] = np.clip(base, 0.1, 0.99)
                
                for j in range(1, seq_length):
                    noise = np.random.uniform(-0.2, 0.2, size=num_features)
                    new_step = sequences[i, j-1] + noise
                    sequences[i, j] = np.clip(new_step, 0.1, 0.99)   
                    print(sequences)
            
            return torch.from_numpy(sequences)

class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class EmotionTrainer:
    def __init__(self, config=None):
        self.config = config or {
            'seq_length': 3,
            'num_features': 6,
            'hidden_size': 128,
            'lr': 0.00001,
            'weight_decay': 1e-7,
            'patience': 5,   
            'min_delta': 1e-4   
        }
        
    def prepare_data(self, num_samples=10000):
        full_data = EmotionDataGenerator.generate_sequences(
            num_samples, self.config['seq_length'])
        X = full_data[:, :-1]
        y = full_data[:, -1]
        split = int(0.8 * len(X))
        return (X[:split], y[:split]), (X[split:], y[split:])

    def train(self):
        (train_X, train_y), (val_X, val_y) = self.prepare_data()
        
        model = EmoMODEL(
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

        early_stopping = EarlyStopping(
            patience=self.config['patience'],
            min_delta=self.config['min_delta']
        )
        
        num_epochs = 1000
        best_model_state = None
        best_val_loss = float('inf')

        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
 
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
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f}")
            

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

 
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()

 
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
 
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Loaded best model with validation loss: {best_val_loss:.4f}")

        self.PLOT_TLVL(train_losses, val_losses, avg_t=avg_train_loss, avg_v=avg_val_loss)

        return model
    
    def PLOT_TLVL(self, train_losses, val_losses, avg_t, avg_v):
        sns.set_style("darkgrid")   
        plt.figure(figsize=(10, 5))

        sns.lineplot(x=range(len(train_losses)), y=train_losses, label='Training', color='#07fafa', linewidth=1)
        sns.lineplot(x=range(len(val_losses)), y=val_losses, label='Validation', color='#07fa91', linewidth=1)

        plt.xlabel('Epochs', color='#141414')
        plt.ylabel('Loss', color='#141414')
        plt.title(f'T {avg_t},V {avg_v}', color='#141414')
        
        legend = plt.legend()
        for text in legend.get_texts():
            text.set_color("#141414")

        plt.show()


if __name__ == "__main__":

    trainer = EmotionTrainer()
    model = trainer.train()
    

    test_sequence = EmotionDataGenerator.generate_sequences(1, trainer.config['seq_length'])
    print(f"\nTest Sequence:\n{test_sequence}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    model.to(device)  
    

    with torch.no_grad():
        model.eval()
        test_sequence = test_sequence.to(device)  
        prediction, _ = model(test_sequence[:, :-1])  
        
        print("\nSample prediction:")
        print("Test Input Sequence:", test_sequence[0, :-1].cpu().numpy()) 
        print("Predicted test_sequence[-1]:", prediction[0].cpu().numpy())
 
