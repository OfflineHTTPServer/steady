import torch
import torch.nn as nn
import torch.nn.functional as F

class CorePredictionModel(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_size=64,
            num_layers=2,
            hidden_layers=[128, 64],
            dropout_rate=0.2
    ):
        super(CorePredictionModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_rate
        )

        self.attention = nn.Linear(hidden_size, 1)

        self.feedforward_layers = nn.ModuleList()
        in_features = hidden_size

        for hidden_size in hidden_layers:
            self.feedforward_layers.append(nn.Linear(in_features, hidden_size))
            self.feedforward_layers.append(nn.BatchNorm1d(hidden_size))
            self.feedforward_layers.append(nn.ReLU())
            self.feedforward_layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_size

        self.output_layer = nn.Linear(in_features, output_size)
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.feedforward_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h_0, c_0))

        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        x = torch.sum(attn_weights * lstm_out, dim=1)
        # x = lstm_out[:, -1, :]

        for layer in self.feedforward_layers:
            x = layer(x)

        x = self.output_layer(x)
        return x

# if __name__ == '__main__':
#     print("works!")
#     short_term = CorePredictionModel(3, 2)
#     x = short_term(
#         torch.tensor(
#             [
#                 [0.2, 0.2, 0.2],
#                 [0.2, 0.2, 0.2]
#             ]
#         ).unsqueeze(1)
#     )

#     print(x)

#     x = short_term(
#         torch.tensor(
#             [
#                 [0.2, 0.2, 0.2],
#                 [0.2, 0.2, 0.2]
#             ]
#         ).unsqueeze(1)
#     )

#     print(x)
