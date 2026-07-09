import torch
import torch.nn as nn


class MultimodalModel(nn.Module):
    def __init__(self, time_input_size, env_input_size):
        super().__init__()

        # TIME SERIES (CNN)
        self.conv1 = nn.Conv1d(time_input_size, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # ENV
        self.env_net = nn.Sequential(
            nn.Linear(env_input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # FINAL
        self.fc = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x_time, x_env):
        # CNN
        x_time = x_time.permute(0, 2, 1)
        x_time = self.relu(self.bn1(self.conv1(x_time)))
        x_time = self.relu(self.bn2(self.conv2(x_time)))
        x_time = x_time.permute(0, 2, 1)

        # LSTM
        x_time, _ = self.lstm(x_time)
        x_time = x_time[:, -1, :]

        x_time = self.dropout(x_time)

        # ENV
        x_env = self.env_net(x_env)

        # FUSION
        x = torch.cat([x_time, x_env], dim=1)

        return self.fc(x)