import torch
import pandas as pd
from model import CorePredictionModel
from sklearn.preprocessing import RobustScaler
import plotly.graph_objects as go
import numpy as np
import re

def evaluate_model(csv_file, model_path):
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

    # Load the data from the CSV file
    data = pd.read_csv(csv_file)

    # Extract the relevant columns and reverse the order
    data = data[['1. open', '2. high', '3. low', '4. close', '5. volume']].iloc[::-1].values

    # Scale the data using RobustScaler
    scaler = RobustScaler()
    data = scaler.fit_transform(data)

    # Initialize the model
    model = CorePredictionModel(input_size, output_size).to(device)

    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device(device)))

    model.eval()

    # Prepare the input sequences and make predictions
    predictions = []
    actuals = []
    for i in range(0, len(data) - takes - output_size + 1, output_size):
        input_data = data[i:i + takes]
        actual_data = data[i + takes:i + takes + output_size, 3]  # Actual 'close' prices

        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            predicted_data = model(input_tensor).cpu().numpy().flatten()

        predictions.extend(predicted_data)
        actuals.extend(actual_data)

   # Inverse transform the predictions and actuals
    predictions = np.array(predictions).reshape(-1, 1)
    actuals = np.array(actuals).reshape(-1, 1)

    # We need to create a dummy array to inverse transform
    dummy_array = np.zeros((predictions.shape[0], data.shape[1]))
    dummy_array[:, 3] = predictions.flatten()
    predictions = scaler.inverse_transform(dummy_array)[:, 3]

    dummy_array[:, 3] = actuals.flatten()
    actuals = scaler.inverse_transform(dummy_array)[:, 3]

    # Create a DataFrame for plotting
    df = pd.DataFrame({
        'Actual': actuals,
        'Predicted': predictions
    })

    # Plot the results using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Actual'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Predicted'], mode='lines', name='Predicted'))

    fig.update_layout(
        title='Model Predictions vs Actual Close Prices',
        xaxis_title='Time',
        yaxis_title='Close Price',
        legend_title='Legend'
    )

    fig.show()

if __name__ == '__main__':
    evaluate_model('./data/JPM.csv', './model-t30-o5.pth')
