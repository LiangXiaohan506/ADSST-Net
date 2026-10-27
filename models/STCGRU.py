import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in

        self.cnn_layer_1_large = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1,
                      kernel_size=(1, 21), stride=(1, 3), bias=False),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(in_channels=1, out_channels=1,
                      kernel_size=(1, 11), stride=(1, 3), bias=False),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )
        self.cnn_layer_1_small = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1,
                      kernel_size=(1, 7), stride=(1, 3), bias=False),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(in_channels=1, out_channels=1,
                      kernel_size=(1, 3), stride=(1, 3), bias=False),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )
        self.cnn_layer_2 = nn.Sequential(
            nn.Conv1d(in_channels=self.enc_in, out_channels=self.enc_in,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(num_features=self.enc_in),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.enc_in, out_channels=self.enc_in * 2,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(num_features=self.enc_in * 2),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.enc_in * 2, out_channels=self.enc_in * 4,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(num_features=self.enc_in * 4),
            nn.ReLU(),
        )
        # Use adaptive pooling to fix the time dimension to 60, so fc_layer_1 can work with any input size
        large_out = ((((self.seq_len - 21) // 3 + 1 - 2) // 2 + 1 - 11) // 3 + 1 - 2) // 2 + 1
        small_out = ((((self.seq_len - 7) // 3 + 1 - 2) // 2 + 1 - 3) // 3 + 1 - 2) // 2 + 1
        self.fc_layer_1 = nn.Linear(large_out+small_out, large_out+small_out)
        self.gru = nn.GRU(input_size=self.enc_in * 4, hidden_size=32,
                          num_layers=1, batch_first=True, bidirectional=True)
        self.flatten = nn.Flatten()
        
        # Decoder
        if self.task_name == 'supervised':
            self.fc_layer_2 = nn.Linear(32 * 2, configs.num_class)
            self.sigmoid = nn.Sigmoid()

    def supervised(self, x_enc, x_mark_enc):  # (batch_size, timestamps, enc_in)
        x = x_enc.permute(0, 2, 1)  # (batch_size, enc_in, seq_len)
        out1 = self.cnn_layer_1_large(torch.unsqueeze(x, dim=1))  # (batch_size, 1, enc_in, time_dim_1)
        out2 = self.cnn_layer_1_small(torch.unsqueeze(x, dim=1))  # (batch_size, 1, enc_in, time_dim_2)
        out = torch.cat((out1, out2), dim=3)  # (batch_size, 1, enc_in, time_dim_combined)
        out = self.cnn_layer_2(torch.squeeze(out, dim=1))  # (batch_size, enc_in*4, time_dim)

        out = self.fc_layer_1(out)  # (batch_size, enc_in*4, time_dim)
        out = out.permute(0, 2, 1)  # (batch_size, time_dim, enc_in*4)
        h0 = torch.zeros(1 * 2, x.size(0), 32).to(x.device)
        # Forward propagate GRU
        _, hn = self.gru(out, h0)  # hn: (2, batch_size, 32) for bidirectional GRU

        # Decode the hidden state of the last time step
        out = self.flatten(hn.permute(1, 0, 2))  # (batch_size, 64)
        out = self.fc_layer_2(out)  # (batch_size, num_class)
        out = self.sigmoid(out)
        return out  # [B, num_class]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'supervised':
            dec_out = self.supervised(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        else:
            raise ValueError("Task name not recognized or not implemented within the STCGRU Model")
