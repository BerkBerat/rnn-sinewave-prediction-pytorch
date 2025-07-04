v# 🔁 Sinusoidal Waveform Prediction with PyTorch RNN

This repository provides a PyTorch implementation of a **Recurrent Neural Network (RNN)** designed for **time series prediction**. The project demonstrates how to create a synthetic sinusoidal dataset, build a simple RNN model, train it to learn the patterns in the sequence, and then use it to predict future values.

---

## 📚 Project Overview

The goal of this project is to predict the next value in a **continuous sinusoidal waveform** based on a sequence of preceding values. This is a fundamental **sequence-to-one prediction problem**, showcasing the capabilities of RNNs in modeling **temporal patterns**.

By training on a sine wave, the model learns to capture the underlying function and extrapolate its behavior into the future.

---

## ✨ Features

* **Synthetic Dataset Generation**: Creates a sinusoidal waveform for training and testing.
* **Data Visualization**: Includes utilities to visualize the generated dataset and prediction outputs.
* **Custom RNN Architecture**: Implements a simple yet effective RNN with a single recurrent layer and a fully connected output layer.
* **Training Loop**: Trains the model using the Adam optimizer and Mean Squared Error (MSE) loss.
* **Prediction and Evaluation**: Uses the trained model to predict future values based on test sequences.
* **Result Visualization**: Plots original sine wave, test inputs, and predicted outputs.

---

## 🛠️ Requirements

Before running the code, make sure the following dependencies are installed:

* Python 3.x
* PyTorch
* NumPy
* Matplotlib

## 🧠 Model Architecture

This model is built for sequence-to-one prediction using PyTorch’s nn.RNN module.

🔹 Recurrent Layer (`nn.RNN`)

This is the core recurrent layer of the model, processing the input sequences:

```python
nn.RNN(
    input_size=1,
    hidden_size=16,
    num_layers=1,
    batch_first=True
)
```
## 📂 Code Structure

The `rnn.py` file contains the full implementation, organized into several key components:

* **`generate_data(seq_length, num_samples)`**: This function is responsible for creating the synthetic sinusoidal waveform dataset and visualizing the generated sine sequences.
* **`RNN` Class**: This is the custom PyTorch class that defines the architecture of the Recurrent Neural Network model.
* **Hyperparameters**: Key training and model parameters such as `seq_length`, `hidden_size`, and `epochs` are explicitly set for easy modification and experimentation.
* **Training Loop**: This section manages the entire training process of the RNN model, utilizing `MSELoss` as the criterion and `Adam` as the optimizer.
* **Test Section**: This part of the script handles the generation of new test sequences and uses the trained model to make predictions on these unseen data points.
* **Visualization**: After training and testing, this component is responsible for plotting the comparison between the real (actual) and predicted sine wave segments, allowing for visual evaluation of the model's performance.


## 🔧 Hyperparameters
You can adjust the following parameters in rnn.py:

| Parameter     | Value | Description                             |
| :------------ | :---- | :-------------------------------------- |
| `seq_length`  | 50    | Input sequence length                   |
| `hidden_size` | 16    | Number of hidden units in the RNN       |
| `epochs`      | 20    | Number of training epochs               |
| `learning_rate` | 0.001 | Learning rate for Adam optimizer        |
| `batch_size`  | 32    | Number of samples per training batch    |

