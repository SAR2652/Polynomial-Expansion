import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 2):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers = num_layers)

    def forward(self, inputs):
        # print(inputs.shape)
        input_embeds = self.embedding(inputs)
        # print(input_embeds.shape)
        out1, (h1, _) = self.lstm(input_embeds)
        return out1, h1

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device = device)

class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=30, num_layers = 1):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers = num_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        # print('Input Shape to Decoder = ', input.shape)
        embedded = self.embedding(input).view(1, 1, -1)
        # print('Embedded Shape before Dropout = ', embedded.shape)
        embedded = self.dropout(embedded)
        # print('Embedded Shape after Dropout = ', embedded.shape)
        # print('Embedded[0] Shape = ', embedded[0].shape)
        # print('Hidden[0] Shape = ', hidden[0].shape)
        # attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], torch.squeeze(hidden[0], axis = 1)), 1)), dim=1)

        # print('Attention Weights Shape after Concatenation & Softmax = ', attn_weights.shape)
        # attn_weights = attn_weights.to(torch.long)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        # print('Attention Shape after BMM = ', attn_applied.shape)

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        # print('Output Shape after Concatenation = ', output.shape)
        output = self.attn_combine(output).unsqueeze(0)
        # print('Output Shape after Attention Combination = ', output.shape)

        output = F.relu(output)
        output = output.to(torch.float32)
        # print('Output Shape before LSTM = ', output.shape)
        output, hidden = self.lstm(output)
        # print('Output Shape after LSTM = ', output.shape)

        output = F.log_softmax(self.out(output[0]), dim=1)
        # print('Output Shape after Log Softmax = ', output.shape)
        return output, hidden, attn_weights

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device = device)

# class EncoderDecoderModel(nn.Module):
#     def __init__(self):

