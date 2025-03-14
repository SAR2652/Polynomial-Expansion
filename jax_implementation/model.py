import random
import jax.numpy as jnp
import flax.linen as nn


class EncoderFLAX(nn.Module):
    vocab_size: int
    embed_dim: int
    hidden_dim: int
    bidirectional: bool = False

    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.embed_dim)
        self.forward_lstm = nn.LSTMCell(self.hidden_dim)
        if self.bidirectional:
            self.backward_lstm = nn.LSTMCell(self.hidden_dim)

    def __call__(self, inputs):
        """ Forward pass of encoder """
        embeddings = self.embedding(inputs)
        # print(embeddings.shape)
        batch_size, seq_len, embed_dim = embeddings.shape

        fwd_hidden = jnp.zeros((batch_size, embed_dim))
        fwd_cell = jnp.zeros((batch_size, embed_dim))
        bkwd_hidden = jnp.copy(fwd_hidden)
        bkwd_cell = jnp.copy(fwd_cell)

        # print('hidden and cell states organized')

        outputs = []
        # Iterate over sequence
        for t in range(seq_len):
            (fwd_hidden, fwd_cell), _ = self.forward_lstm(
                (fwd_hidden, fwd_cell), embeddings[:, t, :])
            outputs.append(fwd_hidden)

        outputs = jnp.stack(outputs, axis=1)
        # Convert list to array

        # print('outputs organized')

        if self.bidirectional:
            backward_outputs = []
            # Iterate over sequence
            for t in range(seq_len - 1, -1, -1):
                (bkwd_hidden, bkwd_cell), out = self.backward_lstm(
                    (bkwd_hidden, bkwd_cell), embeddings[:, t, :])
                backward_outputs.append(bkwd_hidden)

            backward_outputs = jnp.stack(backward_outputs, axis=1)
            outputs = jnp.concatenate([outputs, backward_outputs], axis=-1)

            hidden = jnp.concatenate([fwd_hidden, bkwd_hidden], axis=-1)
            cell = jnp.concatenate([fwd_cell, bkwd_cell], axis=-1)

        else:
            hidden = fwd_hidden
            cell = fwd_cell

        # print(f'Encoder Outputs Shape = {outputs.shape}')
        # print(f'Encoder Hidden Shape = {hidden.shape}')
        # print(f'Encoder Cell Shape = {cell.shape}')

        return outputs, hidden, cell


class MultiHeadAttentionFLAX(nn.Module):
    embed_dim: int
    num_heads: int

    def setup(self):
        assert self.embed_dim % self.num_heads == 0, \
            "Embedding dim must be divisible by num_heads"
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        # Linear transformations for Q, K, V
        self.query_proj = nn.Dense(self.embed_dim)
        self.key_proj = nn.Dense(self.embed_dim)
        self.value_proj = nn.Dense(self.embed_dim)

        # Output projection
        self.out_proj = nn.Dense(self.embed_dim)

    def __call__(self, query, key=None, value=None):
        batch_size, query_seq_len, _ = query.shape

        # Default to self-attention
        if key is None:
            key = query
        if value is None:
            value = query

        _, key_seq_len, _ = key.shape
        _, value_seq_len, _ = value.shape

        # Project queries, keys, and values
        Q = self.query_proj(query)  # (batch_size, query_seq_len, embed_dim)
        K = self.key_proj(key)      # (batch_size, key_seq_len, embed_dim)
        V = self.value_proj(value)  # (batch_size, value_seq_len, embed_dim)

        # Reshape to (batch_size, seq_len, num_heads, head_dim)
        Q = Q.reshape(batch_size, query_seq_len, self.num_heads, self.head_dim)
        K = K.reshape(batch_size, key_seq_len, self.num_heads, self.head_dim)
        V = V.reshape(batch_size, value_seq_len, self.num_heads, self.head_dim)

        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        Q = jnp.transpose(Q, (0, 2, 1, 3))
        K = jnp.transpose(K, (0, 2, 1, 3))
        V = jnp.transpose(V, (0, 2, 1, 3))

        # Compute scaled dot-product attention
        attention_scores = jnp.einsum("bhqd, bhkd -> bhqk", Q, K) * self.scale
        # (batch_size, num_heads, query_seq_len, key_seq_len)
        attention_weights = nn.softmax(attention_scores, axis=-1)

        # Compute attention output
        attention_output = jnp.einsum("bhqk, bhvd -> bhqd", attention_weights,
                                      V)

        # Restore shape: (batch_size, query_seq_len, num_heads, head_dim)
        attention_output = jnp.transpose(attention_output, (0, 2, 1, 3))

        # Reshape back to (batch_size, query_seq_len, embed_dim)
        attention_output = attention_output.reshape(batch_size, query_seq_len,
                                                    self.embed_dim)

        return self.out_proj(attention_output)


class DecoderSACAFLAX(nn.Module):
    embed_dim: int
    num_heads: int
    vocab_size: int

    def setup(self):
        # Token embedding layer
        self.embedding = nn.Embed(self.vocab_size, self.embed_dim)

        # Self-Attention (decoder attending to its own past tokens)
        self.self_attention = MultiHeadAttentionFLAX(self.embed_dim,
                                                     self.num_heads)

        # Cross-Attention (queries from decoder, keys/values from encoder)
        self.cross_attention = MultiHeadAttentionFLAX(self.embed_dim,
                                                      self.num_heads)

        # LSTM processes combined attention representations
        self.lstm = nn.LSTMCell(self.embed_dim)

        # Final projection to vocabulary size
        self.fc_out = nn.Dense(self.vocab_size)

    def __call__(self, target_token, encoder_outputs, hidden_state,
                 cell_state):
        """
        Args:
            target_token: (B, 1) Last generated token.
            encoder_outputs: (B, S, H) Encoder outputs (keys & values for
            cross-attention).
            hidden_state: (B, H) Previous LSTM hidden state.
            cell_state: (B, H) Previous LSTM cell state.

        Returns:
            next_token_logits: (B, 1, V) Predicted token logits for next step.
            new_hidden: (B, H) Updated hidden state.
            new_cell: (B, H) Updated cell state.
        """

        # 1. Embed the target token
        embedded = self.embedding(target_token)  # (B, 1, E)

        # print(f'Embedded = {embedded.shape}')

        # 2. Apply Self-Attention (causal)
        self_attn_output = self.self_attention(embedded, embedded, embedded)
        # (B, 1, E)

        # print(f'Self Attn = {self_attn_output.shape}')

        # 3. Apply Cross-Attention (queries from Self-Attention Output,
        # keys/values from encoder)
        cross_attn_output = self.cross_attention(self_attn_output,
                                                 encoder_outputs,
                                                 encoder_outputs)  # (B, 1, E)

        # print(f'Cross Attn = {cross_attn_output.shape}')

        # 4. Concatenate embeddings and attention output
        lstm_input = jnp.concatenate([embedded, cross_attn_output],
                                     axis=-1)  # (B, 1, 2E)

        # lstm_ip_hidden = jnp.zeros((batch_size, self.embed_dim * 2))
        # lstm_ip_cell = jnp.zeros((batch_size, self.embed_dim * 2))
        # print(f'LSTM Input = {lstm_input.shape}')
        # print(f'Hidden Shape = {hidden_state.shape}')
        # print(f'Cell Shape = {cell_state.shape}')

        # 5. LSTM step (JAX LSTMCell operates per timestep)
        (hidden_state, cell_state), _ = self.lstm(
            (hidden_state, cell_state), lstm_input[:, 0, :])

        # print(f'Out Hidden Shape = {hidden_state.shape}')
        # print(f'Out Shape = {out.shape}')

        # 6. Predict next token
        next_token_logits = self.fc_out(hidden_state)  # (B, 1, V)

        return next_token_logits, hidden_state, cell_state


class CrossAttentionModelFLAX(nn.Module):
    enc_embed_dim: int
    hidden_dim: int
    vocab_size: int
    num_heads: int
    sos_token_id: int
    bidirectional: bool = False
    teacher_force_ratio: float = 0.5

    def setup(self):
        self.encoder = EncoderFLAX(self.vocab_size, self.enc_embed_dim,
                                   self.hidden_dim, self.bidirectional)

        hidden_dim = self.hidden_dim * 2 if self.bidirectional else \
            self.hidden_dim
        self.decoder = DecoderSACAFLAX(hidden_dim, self.num_heads,
                                       self.vocab_size)

    def __call__(self, inputs, targets=None):
        """
        Args:
            inputs: (B, S) Input token indices.
            targets: (B, T) Target token indices (optional, for teacher
            forcing).
            rng: PRNG key for random number generation.

        Returns:
            outputs: (B, T, V) Logits for each token position.
        """

        # Encode input sequence
        encoder_outputs, decoder_hidden_state, decoder_cell_state = \
            self.encoder(inputs)

        batch_size, target_len, _ = encoder_outputs.shape
        outputs = jnp.zeros((batch_size, target_len, self.vocab_size))

        # Initial decoder input: <SOS> token
        decoder_input = jnp.full((batch_size, 1), self.sos_token_id)

        for t in range(target_len):

            logits, decoder_hidden_state, decoder_cell_state = self.decoder(
                decoder_input, encoder_outputs, decoder_hidden_state,
                decoder_cell_state
            )

            # print(f'Logits Shape = {logits.shape}')

            outputs = outputs.at[:, t, :].set(logits)

            # Decide whether to use teacher forcing
            use_teacher_forcing = random.random() < \
                self.teacher_force_ratio

            if targets is not None and use_teacher_forcing:
                decoder_input = targets[:, t:t+1]  # Use ground truth
            else:
                print(logits.shape)
                decoder_input = jnp.argmax(logits, axis=-1)
                print(decoder_input.shape)
                # Use predicted token

        return outputs
