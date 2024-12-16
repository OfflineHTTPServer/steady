import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
from model import CorePredictionModel
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import RobustScaler

def train_model(csv_file, input_size=5, output_size=5, num_epochs=100, takes=30, optimizer_cls=optim.Adam, loss_fn=nn.HuberLoss(), model=None,  validation_split=0.2):
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load the data from the CSV file
    data = pd.read_csv(csv_file)

    # Extract the relevant columns and reverse the order
    data = data[['1. open', '2. high', '3. low', '4. close', '5. volume']].iloc[::-1].values

    # Scale the data using RobustScaler
    scaler = RobustScaler()
    data = scaler.fit_transform(data)

    # Prepare the input and output sequences
    X, y = [], []
    for i in range(len(data) - takes - output_size + 1):
        X.append(data[i:i + takes])
        y.append(data[i + takes:i + takes + output_size, 3])  # Predicting the 'close' prices

    X = np.array(X)
    y = np.array(y)

    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)

    # Split the data into training and validation sets
    dataset = TensorDataset(X, y)
    train_size = int((1 - validation_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize the model if not provided
    if model is None:
        model = CorePredictionModel(input_size, output_size).to(device)
    else:
        model.to(device)
        model.train()

    # Initialize the optimizer and loss function
    optimizer = optimizer_cls(model.parameters())
    loss_function = loss_fn

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = loss_function(outputs, batch_y)
                val_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}')

    # Save the trained model with the specified filename format
    def save():
        model_save_path = f'model-t{takes}-o{output_size}.pth'
        torch.save(model.state_dict(), model_save_path)
        print(f'Model saved to {model_save_path}')

    return save

if __name__ == '__main__':
    train_model(
        "./data/NEE.csv",
        num_epochs=150,
        output_size=5,
        takes=30
    )()
