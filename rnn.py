"""

"""

# %% Library
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# %% Create Dataset and Visualize Dataset
def generate_data(seq_length = 50, num_samples = 1000):
    X = np.linspace(0, 100, num_samples) # Generate num_samples data between 0 and 100
    y = np.sin(X)
    sequence = []
    targets = []
    
    for i in range(len(X)- seq_length):
        sequence.append(y[i:i+seq_length])
        targets.append(y[i+seq_length])
        
    plt.figure(figsize = (8, 4))
    plt.plot(X, y, label = "sin(X)", color = "b", linewidth = 2)
    plt.title("Sinusoidal Waveform")
    plt.xlabel("Time(Radian)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return np.array(sequence), np.array(targets)

sequence, targets = generate_data()

# %% Build RNN Model
class RNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, num_layers = 1):
       """ 
       RNN -> Linear (Output)
       """
       super(RNN,self).__init__()
       self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True)
       self.fc = nn.Linear(hidden_size, output_size)
    
    def forward (self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

model = RNN(1, 16, 1, 1)

# %% Train
# Hyperparameters
seq_length = 50 # Size of the input array
input_size = 1 
hidden_size = 16 # Number of hidden layer's nodes in RNN
output_size = 1
num_layers = 1 # Number of layers in RNN
epochs = 20
batch_size = 32
learning_rate = 0.001

X, y = generate_data(seq_length)
X = torch.tensor(X, dtype = torch.float32).unsqueeze(-1) # Convert Pytorch Tensor and add dimension
y = torch.tensor(y, dtype = torch.float32).unsqueeze(-1)

dataset = torch.utils.data.TensorDataset(X, y) # Create Pytorch Dataset
dataLoader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)

model = RNN(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range (epochs):
    for batch_x, batch_y in dataLoader:
        optimizer.zero_grad()
        pred_y = model(batch_x)
        loss = criterion(pred_y, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch: {epoch+1}/{epochs}, Loss {loss.item():.4f}")
# %% Test
# Create data for test
X_test = np.linspace(100, 110, seq_length).reshape(1, -1)
y_test = np.sin(X_test) # Real result for X_test

# From numpy to tensor
X_test2 = np.linspace(120, 130, seq_length).reshape(1, -1)
y_test2 = np.sin(X_test) # Real result for X_test2
X_test = torch.tensor(y_test, dtype = torch.float32).unsqueeze(-1)
X_test2 = torch.tensor(y_test2, dtype = torch.float32).unsqueeze(-1)

# Make predictions with using model
model.eval()
prediction1 = model(X_test).detach().numpy()
prediction2 = model(X_test2).detach().numpy()

# View the results
plt.figure()
plt.plot(np.linspace(0, 100, len(y)), y, marker = "o", label = "Training Dataset")
plt.plot(X_test.numpy().flatten(), marker = "o", label = "Test1")
plt.plot(X_test2.numpy().flatten(), marker = "o", label = "Test2")

plt.plot(np.arange(seq_length, seq_length+1), prediction1.flatten(), "ro", label = "Prediction 1")
plt.plot(np.arange(seq_length, seq_length+1), prediction2.flatten(), "ro", label = "Prediction 2")

plt.legend()
plt.show()