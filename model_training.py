import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Step 1: Fetch Historical Price Data
def fetch_data(ticker, start_date):
    data = yf.download(ticker, start=start_date)
    return data['Close'].values.reshape(-1, 1)

# Step 2: Preprocess Data
def preprocess_data(data, window_size):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    X = []
    y = []
    for i in range(window_size, len(data_scaled)):
        X.append(data_scaled[i-window_size:i, 0])
        y.append(data_scaled[i, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y, scaler

# Step 3: Build the Neural Network Model
class PricePredictor(nn.Module):
    def __init__(self, input_size):
        super(PricePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Step 4: Train the Model
def train_model(model, dataloader, criterion, optimizer, epochs=50):
    model.train()
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

# Step 5: Predict the Next Price
def predict_next_price(model, data, window_size, scaler):
    model.eval()
    last_window = data[-window_size:]
    last_window_scaled = scaler.transform(last_window)
    last_window_scaled = torch.tensor(last_window_scaled, dtype=torch.float32).view(1, -1)
    
    with torch.no_grad():
        predicted_price_scaled = model(last_window_scaled)
        print(predicted_price_scaled)
    
    predicted_price = scaler.inverse_transform(predicted_price_scaled.numpy().reshape(-1, 1))
    return predicted_price[0, 0]

# Step 6: Convert Model to ONNX
def serialize_to_onnx(model, input_size, onnx_file_path):
    model.eval()
    dummy_input = torch.randn(1, input_size)
    torch.onnx.export(model, dummy_input, onnx_file_path,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    print(f"✅ Model has been converted to ONNX and saved at {onnx_file_path} ")

# Main execution
if __name__ == "__main__":
    # Parameters
    ticker = "ETH-USD"
    start_date = "2020-01-01"
    window_size = 60
    batch_size = 32
    epochs = 50
    onnx_file_path = "torch_eth_price_predictor.onnx"
    
    # Fetch and preprocess data
    print("Fetching and preprocessing data...")
    data = fetch_data(ticker, start_date)
    X, y, scaler = preprocess_data(data, window_size)
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    # Create DataLoader
    print("Creating DataLoader...")
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Build and train model
    print("Building and training model...")
    model = PricePredictor(X_tensor.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_model(model, dataloader, criterion, optimizer, epochs=epochs)
    
    # Predict the next price
    print("Predicting the next price...")
    next_price = predict_next_price(model, data, window_size, scaler)
    print(f"Predicted next price of ETH/USDC: {next_price}")
    
    # Convert model to ONNX
    print("Converting model to ONNX format...")
    serialize_to_onnx(model, X_tensor.shape[1], onnx_file_path)

