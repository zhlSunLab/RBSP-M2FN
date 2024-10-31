import torch
from utils import DefinedConfig
import torch.nn as nn
import torch.nn.functional as F

defconstant = DefinedConfig()


class BiLSTM_Attention(nn.Module):
    def __init__(self):
        super(BiLSTM_Attention, self).__init__()
        input_dim = 10
        n_hidden = 20
        output_dim = 128
        self.lstm = nn.LSTM(input_dim, n_hidden, bidirectional=True)
        self.out = nn.Linear(n_hidden * 2, output_dim)

    # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix

    def attention_net(self, lstm_output, final_state):
        batch_size = len(lstm_output)
        hidden = final_state.view(batch_size, -1,
                                  1)  # hidden : [batch_size, n_hidden * num_directions(=2), n_layer(=1)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)  # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # context : [batch_size, n_hidden * num_directions(=2)]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights

    def forward(self, X):
        '''
        X: [batch_size, seq_len]
        '''
        # X : [batch_size, seq_len, input_dim]
        input = X.transpose(0, 1)  # input : [seq_len, batch_size, embedding_dim]
        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        output, (final_hidden_state, final_cell_state) = self.lstm(input)
        output = output.transpose(0, 1)  # output : [batch_size, seq_len, n_hidden]
        attn_output, attention_weights = self.attention_net(output, final_hidden_state)
        out = self.out(attn_output)
        return out  # model : [batch_size, num_classes], attention : [batch_size, n_step]

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(17, 10), padding=(8, 0))
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=(64, 1), stride=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class InceptionA(nn.Module):
    def __init__(self):
        super(InceptionA, self).__init__()
        self.branch1x1 = torch.nn.Conv1d(in_channels=1, out_channels=64, kernel_size=1)
        self.bn1 = torch.nn.BatchNorm1d(64)

        self.branch5x5_1 =torch.nn.Conv1d(in_channels=1, out_channels=48, kernel_size=1)
        self.bn2_1 = torch.nn.BatchNorm1d(48)
        self.branch5x5_2 = torch.nn.Conv1d(in_channels=48, out_channels=64, kernel_size=5, padding=2)
        self.bn2_2 = torch.nn.BatchNorm1d(64)

        self.branch3x3_1 = torch.nn.Conv1d(in_channels=1, out_channels=64, kernel_size=1)
        self.bn3_1 = torch.nn.BatchNorm1d(64)
        self.branch3x3_2 = torch.nn.Conv1d(in_channels=64, out_channels=96, kernel_size=3, padding=1)
        self.bn3_2 = torch.nn.BatchNorm1d(96)
        self.branch3x3_3 = torch.nn.Conv1d(in_channels=96, out_channels=96, kernel_size=3, padding=1)
        self.bn3_3 = torch.nn.BatchNorm1d(96)

        self.branch_pool = torch.nn.Conv1d(in_channels=1, out_channels=64, kernel_size=1)
        self.bnb = torch.nn.BatchNorm1d(64)

    def forward(self, x):
        x_1x1 = self.branch1x1(x)
        x_1x1 = self.bn1(x_1x1)

        x_5x5 = self.branch5x5_1(x)
        x_5x5 = self.bn2_1(x_5x5)
        x_5x5 = self.branch5x5_2(x_5x5)
        x_5x5 = self.bn2_2(x_5x5)

        x_3x3 = self.branch3x3_1(x)
        x_3x3 = self.bn3_1(x_3x3)
        x_3x3 = self.branch3x3_2(x_3x3)
        x_3x3 = self.bn3_2(x_3x3)
        x_3x3 = self.branch3x3_3(x_3x3)
        x_3x3 = self.bn3_3(x_3x3)

        x_branch = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x_branch = self.branch_pool(x_branch)
        x_branch = self.bnb(x_branch)

        output = torch.cat([x_1x1, x_5x5, x_3x3, x_branch], 1)
        return output

class DON(torch.nn.Module):
    def __init__(self):
        super(RLBind, self).__init__()

        self.BA = BiLSTM_Attention()
        self.CNN = CNN()
        self.inception = InceptionA()
        self.fusion = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=288, out_channels=3, kernel_size=5, stride=2),
            torch.nn.BatchNorm1d(3)
        )

        self.DNN = torch.nn.Sequential(
            torch.nn.Linear(351, 196),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(196, 96),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3)
        )

        self.OutLayer = torch.nn.Sequential(
            torch.nn.Linear(96, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, g_feature, l_feature):
        g_feature_CNN = self.CNN(g_feature)
        g_feature_CNN_shape = g_feature_CNN.data.shape
        g_feature_CNN = g_feature_CNN.view(g_feature_CNN_shape[0], g_feature_CNN_shape[1]*g_feature_CNN_shape[2]*g_feature_CNN_shape[3])

        g_shapes = g_feature.data.shape
        g_feature = g_feature.view(g_shapes[0], g_shapes[2], g_shapes[3])
        g_feature_BA = self.BA(g_feature)

        g_feature = g_feature_BA + g_feature_CNN

        l_shapes = l_feature.data.shape
        l_feature = l_feature.view(l_shapes[0], l_shapes[1] * l_shapes[2] * l_shapes[3])

        feature = torch.cat((g_feature, l_feature), 1)
        feature_shape = feature.data.shape
        feature = feature.view(feature_shape[0], 1, feature_shape[1])
        feature = self.inception(feature)
        feature = self.fusion(feature)
        feature = feature.view(feature_shape[0], -1)

        feature = self.DNN(feature)
        feature = self.OutLayer(feature)

        return feature
