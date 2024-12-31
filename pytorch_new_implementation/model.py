import torch
# import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 p: float = 0.1):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                            bidirectional=True)

    def forward(self, inputs):
        embeddings = self.dropout(self.embedding(inputs))
        outputs, (hidden, cell) = self.lstm(embeddings)
        return outputs, hidden, cell


class BahdanauAttention(nn.Module):
    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int,
                 attention_size: int):
        super(BahdanauAttention, self).__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.attention_size = attention_size

        # Layers for calculating attention scores
        self.W = nn.Linear(encoder_hidden_size * 2, attention_size, bias=False)
        self.U = nn.Linear(decoder_hidden_size * 2, attention_size, bias=False)
        self.v = nn.Linear(attention_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        """
        Compute the Bahdanau attention scores and context vector.

        Args:
            decoder_hidden (Tensor): Current hidden state of the decoder
            (batch_size, decoder_hidden_size).
            encoder_outputs (Tensor): Outputs from the encoder
            (batch_size, seq_len, encoder_hidden_size).

        Returns:
            context_vector (Tensor): Weighted sum of encoder outputs
            (batch_size, encoder_hidden_size).
            attention_weights (Tensor): Attention weights
            (batch_size, seq_len).
        """
        # Expand decoder_hidden to match encoder_outputs dimensions
        # decoder_hidden = decoder_hidden.unsqueeze(1)
        # (batch_size, 1, decoder_hidden_size)

        _, seq_len, _ = encoder_outputs.shape
        decoder_hidden = decoder_hidden.transpose(0, 1)
        # (batch_size, 2, encoder_hidden_dim)
        decoder_hidden = decoder_hidden.reshape(decoder_hidden.size(0), -1)
        decoder_hidden = decoder_hidden.unsqueeze(1)
        decoder_hidden = decoder_hidden.repeat(1, seq_len, 1)

        temp1 = self.W(encoder_outputs)
        # print(temp1.shape)
        temp2 = self.U(decoder_hidden)
        # print(temp2.shape)
        temp3 = torch.tanh(temp1 + temp2)

        # Calculate alignment scores
        score = self.v(temp3)
        # (batch_size, seq_len, 1)
        score = score.squeeze(-1)  # (batch_size, seq_len)

        # Compute attention weights
        attention_weights = F.softmax(score, dim=-1)  # (batch_size, seq_len)

        # Compute context vector as the weighted sum of encoder outputs
        context_vector = torch.bmm(attention_weights.unsqueeze(1),
                                   encoder_outputs)
        # (batch_size, 1, encoder_hidden_size)
        return context_vector, attention_weights


class DecoderWithBahdanauAttention(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int, embedding_dim: int,
                 encoder_hidden_dim: int, p: float = 0.1):
        super(DecoderWithBahdanauAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embedding_dim
        self.encoder_hidden_dim = encoder_hidden_dim

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Dropout Layer
        self.dropout = nn.Dropout(p)

        # Attention mechanism
        self.attention = BahdanauAttention(encoder_hidden_dim, hidden_dim,
                                           hidden_dim)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=embedding_dim + encoder_hidden_dim * 2,
                            hidden_size=hidden_dim * 2, batch_first=True)

        # Fully connected layer for output
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, input_token, hidden_state, cell_state, encoder_outputs):
        """
        Perform a decoding step.

        Args:
            input_token (Tensor): Input token indices (batch_size,).
            hidden_state (Tensor): Hidden state of the decoder
            (batch_size, hidden_dim).
            cell_state (Tensor): Cell state of the decoder
            (batch_size, hidden_dim).
            encoder_outputs (Tensor): Outputs from the encoder
            (batch_size, seq_len, encoder_hidden_dim).

        Returns:
            output (Tensor): Decoder output (batch_size, vocab_size).
            hidden_state (Tensor): Updated hidden state
            (batch_size, hidden_dim).
            cell_state (Tensor): Updated cell state
            (batch_size, hidden_dim).
            attention_weights (Tensor): Attention weights
            (batch_size, seq_len).
        """
        # Get embedding of input token
        input_embedding = self.dropout(
            self.embedding(input_token).unsqueeze(1)
            )

        # Compute attention context vector
        context_vector, attention_weights = self.attention(
            hidden_state, encoder_outputs
        )
        # print(context_vector.shape)
        # (batch_size, 1, encoder_hidden_dim)
        # print(input_embedding.shape)

        # Concatenate input embedding and context vector
        lstm_input = torch.cat((input_embedding, context_vector), dim=-1)
        # (batch_size, 1, embedding_dim + encoder_hidden_dim)

        # print(lstm_input.shape)
        # print(self.embed_dim + self.encoder_hidden_dim * 2)
        # print(hidden_state.shape)
        # print(cell_state.shape)

        hidden_state = hidden_state.permute(1, 0, 2)
        # (batch_size, num_directions, hidden_dim)
        hidden_state = hidden_state.reshape(hidden_state.size(0), -1) \
            .unsqueeze(0)

        cell_state = cell_state.permute(1, 0, 2)
        # (batch_size, num_directions, hidden_dim)
        cell_state = cell_state.reshape(cell_state.size(0), -1) \
            .unsqueeze(0)

        # Pass through LSTM
        lstm_output, (hidden_state, cell_state) = self.lstm(
            lstm_input, (hidden_state, cell_state)
            )

        # print(lstm_output.shape)
        # Compute output
        output = self.fc(lstm_output.squeeze(1))  # (batch_size, vocab_size)
        # print(output.shape)
        return output, hidden_state, cell_state, attention_weights


class Seq2SeqModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int,
                 encoder_hidden_dim: int, decoder_hidden_dim: int,
                 sos_token_id: int, target_len: int, device: torch.device):
        super(Seq2SeqModel, self).__init__()

        self.vocab_size = vocab_size
        self.encoder = Encoder(vocab_size, embed_dim, encoder_hidden_dim)
        self.decoder = DecoderWithBahdanauAttention(
            decoder_hidden_dim, vocab_size, embed_dim, encoder_hidden_dim
            )
        self.sos_token_id = sos_token_id
        self.target_len = target_len
        self.device = device

    def forward(self, inputs, teacher_force_ratio: float = 0.5, targets=None,
                eval: bool = False):

        encoder_outputs, decoder_hidden_state, decoder_cell_state = \
            self.encoder(inputs)

        batch_size, _, _ = encoder_outputs.shape
        # use_teacher_forcing = random.random() < teacher_force_ratio
        # use_teacher_forcing = torch.tensor([use_teacher_forcing] *
        #                                    batch_size).to(self.device)

        decoder_input = torch.tensor([self.sos_token_id] * batch_size,
                                     dtype=torch.int32).to(self.device)

        outputs = torch.zeros(batch_size, self.target_len,
                              self.vocab_size).to(self.device)
        # best_guesses = np.array

        if eval:
            best_guesses = np.zeros((batch_size, self.target_len))

        for t in range(1, self.target_len):

            logits, decoder_hidden_state, decoder_cell_state, _ = \
                self.decoder(decoder_input, decoder_hidden_state,
                             decoder_cell_state, encoder_outputs)

            best_guess = logits.argmax(1)
            outputs[:, t, :] = logits

            if not eval:
                # decoder_input = torch.where(
                #     # Align dimensions for broadcasting
                #     use_teacher_forcing.unsqueeze(1),
                #     # Use target token (teacher forcing)
                #     targets[:, t].unsqueeze(1),
                #     best_guess.unsqueeze(1)     # Use model's predicted token
                # ).squeeze(1)
                decoder_input = targets[:, t]
            else:
                decoder_input = best_guess
                best_guess_np = best_guess.detach().cpu().numpy()
                best_guesses[:, t] = best_guess_np
                # print(best_guesses)
                # best_guesses.append(best_guess.item())

        # print(outputs.shape)
        if not eval:
            return outputs
        else:
            return outputs, best_guesses
