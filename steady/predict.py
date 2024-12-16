import torch
import numpy as np
from model import CorePredictionModel
from sklearn.preprocessing import RobustScaler
import re

def predict(model_path, dataframes):
    # Extract 'takes' and 'output_size' from the model filename
    match = re.search(r'model-t(\d+)-o(\d+)', model_path)
    if not match:
        raise ValueError("Model filename does not match the expected format 'model-t{takes}-o{output_size}.pth'")

    takes = int(match.group(1))
    output_size = int(match.group(2))
    input_size = 5  # As specified, the input size is always 5

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Initialize the model
    model = CorePredictionModel(input_size, output_size).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # Prepare the scaler
    scaler = RobustScaler()

    predictions = []
    for df in dataframes:
        # Extract the relevant columns and reverse the order
        data = df[['1. open', '2. high', '3. low', '4. close', '5. volume']].values

        # Scale the data using RobustScaler
        data = scaler.fit_transform(data)

        # Prepare the input sequence
        input_data = data[-takes:]
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            predicted_data = model(input_tensor).cpu().numpy().flatten()

        # Inverse transform the predictions
        dummy_array = np.zeros((predicted_data.shape[0], data.shape[1]))
        dummy_array[:, 3] = predicted_data
        predicted_data = scaler.inverse_transform(dummy_array)[:, 3]

        predictions.append(predicted_data.tolist())

    return predictions
