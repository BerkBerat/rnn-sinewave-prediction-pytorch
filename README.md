# 🔁 Sinusoidal Waveform Prediction with PyTorch RNN

This repository provides a PyTorch implementation of a **Recurrent Neural Network (RNN)** designed for **time series prediction**. The project demonstrates how to create a synthetic sinusoidal dataset, build a simple RNN model, train it to learn the patterns in the sequence, and then use it to predict future values.

---

## 📚 Project Overview

The goal of this project is to predict the next value in a **continuous sinusoidal waveform** based on a sequence of preceding values. This is a fundamental **sequence-to-one prediction problem**, showcasing the capabilities of RNNs in modeling **temporal patterns**.

By training on a sine wave, the model learns to capture the underlying function and extrapolate its behavior into the future.

---

## ✨ Features

- **Synthetic Dataset Generation**  
  Creates a sinusoidal waveform for training and testing.

- **Data Visualization**  
  Includes utilities to visualize the generated dataset and prediction outputs.

- **Custom RNN Architecture**  
  Implements a simple yet effective RNN with a single recurrent layer and a fully connected output layer.

- **Training Loop**  
  Trains the model using the Adam optimizer and Mean Squared Error (MSE) loss.

- **Prediction and Evaluation**  
  Uses the trained model to predict future values based on test sequences.

- **Result Visualization**  
  Plots original sine wave, test inputs, and predicted outputs.

---

## 🛠️ Requirements

Before running the code, make sure the following dependencies are installed:

- Python 3.x
- PyTorch
- NumPy
- Matplotlib

You can install them using pip:

```bash
pip install torch numpy matplotlib


Model Architecture
This model is built for sequence-to-one prediction using PyTorch’s nn.RNN module.

🔹 Recurrent Layer
python
Kopyala
Düzenle
nn.RNN(
    input_size=1,
    hidden_size=16,
    num_layers=1,
    batch_first=True
)
input_size=1: Each input is a single value (sine amplitude)

hidden_size=16: Size of hidden state vector

num_layers=1: A single RNN layer

batch_first=True: Input shape is (batch, sequence, feature)

🔹 Output Layer
python
Kopyala
Düzenle
nn.Linear(hidden_size, 1)
Transforms the final hidden state into a single predicted value

🔹 Forward Pass
python
Kopyala
Düzenle
def forward(self, x):
    out, _ = self.rnn(x)
    out = self.fc(out[:, -1, :])  # Use the output at the last time step
    return out
📂 Code Structure
rnn.py: Contains the full implementation

generate_data(seq_length, num_samples): Creates and visualizes sine sequences

RNN: Custom RNN model class

Hyperparameters: Set at the top (e.g., seq_length, hidden_size, epochs)

train_model(): Trains the model with MSELoss and Adam

test_model(): Generates test sequences and makes predictions

plot_results(): Visualizes the comparison of real vs predicted sine waves

🔧 Hyperparameters
You can adjust the following parameters in rnn.py:

python
Kopyala
Düzenle
seq_length = 20        # Input sequence length
hidden_size = 16       # Number of hidden units in the RNN
epochs = 100           # Number of training epochs
learning_rate = 0.01   # Learning rate for Adam optimizer
