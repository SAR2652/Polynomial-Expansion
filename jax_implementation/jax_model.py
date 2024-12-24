import jax      # type: ignore
import jax.numpy as jnp     # type: ignore
from jax import random     # type: ignore
from jax.lax import reshape, batch_matmul     # type: ignore


class LSTMCell:
    def __init__(self, input_dim: int, hidden_dim: int, prng_key):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Initial Hidden State and Context Vector
        self.h_prev = jnp.zeros((hidden_dim,))
        self.c_prev = jnp.zeros((hidden_dim,))

        k1, k2, k3, k4, k5, k6, k7, k8 = random.split(prng_key, 8)

        """Forget Gate"""

        """Part 1: Weights and Bias to remove info from context that is no
        longer needed"""

        # For hidden state
        self.U_f = random.normal(k1, (hidden_dim, hidden_dim))
        self.b1_f = random.normal(k1, (hidden_dim,))

        # For current input
        self.W_f = random.normal(k2, (input_dim, hidden_dim))
        self.b2_f = random.normal(k2, (hidden_dim,))

        """Part 2: Weights and Bias to extract info from previous hidden state
        and current inputs"""

        # For hidden state
        self.U_g = random.normal(k3, (hidden_dim, hidden_dim))
        self.b1_g = random.normal(k3, (hidden_dim,))

        # For current input
        self.W_g = random.normal(k4, (input_dim, hidden_dim))
        self.b2_g = random.normal(k4, (hidden_dim,))

        """Add Gate: Weights and Bias to select information to add to current
        context"""

        # For hidden state
        self.U_i = random.normal(k5, (hidden_dim, hidden_dim))
        self.b1_i = random.normal(k5, (hidden_dim,))

        # For current input
        self.W_i = random.normal(k6, (input_dim, hidden_dim))
        self.b2_i = random.normal(k6, (hidden_dim,))

        """Output Gate: Weights and Bias to decide information needed for
        current state"""

        # For hidden state
        self.U_o = random.normal(k7, (hidden_dim, hidden_dim))
        self.b1_o = random.normal(k7, (hidden_dim,))

        # For current input
        self.W_o = random.normal(k8, (input_dim, hidden_dim))
        self.b2_o = random.normal(k8, (hidden_dim,))

    def __call__(self, x_t, h_prev, c_prev):

        temp1 = jnp.dot(h_prev, self.U_f) + self.b1_f
        temp2 = jnp.dot(x_t, self.W_f) + self.b2_f
        temp_sum1 = temp1 + temp2
        f_t = jax.nn.sigmoid(temp_sum1)
        k_t = f_t * c_prev

        temp3 = jnp.dot(h_prev, self.U_g) + self.b1_g
        temp4 = jnp.dot(x_t, self.W_g) + self.b2_g
        temp_sum2 = temp3 + temp4
        g_t = jnp.tanh(temp_sum2)

        temp5 = jnp.dot(h_prev, self.U_i) + self.b1_i
        temp6 = jnp.dot(x_t, self.W_i) + self.b2_i
        temp_sum3 = temp5 + temp6
        i_t = jnp.tanh(temp_sum3)

        j_t = g_t * i_t

        c_t = j_t + k_t

        temp7 = jnp.dot(h_prev, self.U_o) + self.b1_o
        temp8 = jnp.dot(x_t, self.W_o) + self.b2_o
        temp_sum4 = temp7 + temp8
        o_t = jnp.tanh(temp_sum4)

        h_t = o_t * jnp.tanh(c_t)

        return h_t, c_t


class LSTMLayer:
    def __init__(self, input_dim: int, hidden_dim: int, prng_key):

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell = LSTMCell(input_dim, hidden_dim, prng_key)

    def __call__(self, x, h_0=None, c_0=None):

        batch_size, seq_len, _ = x.shape
        h_t = jnp.zeros((batch_size, self.hidden_dim)) if h_0 is None else h_0
        c_t = jnp.zeros((batch_size, self.hidden_dim)) if c_0 is None else c_0

        outputs = list()

        for t in range(seq_len):
            h_t, c_t = self.cell(x[:, t, :], h_t, c_t)
            outputs.append(h_t)

        outputs = jnp.stack(outputs, axis=1)

        return h_t, c_t, outputs


class BahdanauAttention:
    def __init__(self, hidden_dim: int, attention_dim: int, prng_key):
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim

        k1, k2, k3, k4, k5 = random.split(prng_key, 5)
        self.W_h = random.normal(k1, (hidden_dim, attention_dim))
        self.b_h = random.normal(k2, (hidden_dim,))
        self.W_c = random.normal(k3, (hidden_dim, attention_dim))
        self.b_c = random.normal(k4, (hidden_dim,))
        self.V = random.normal(k5, (attention_dim, 1))

    def __call__(self, hidden_state, encoder_outputs):

        batch_size, seq_len, _ = encoder_outputs.shape

        # (batch_size, hidden_dim) -> (batch_size, 1, hidden_dim)
        hidden_state = jnp.expand_dims(hidden_state, 1)
        # (batch_size, 1, hidden_dim) -> (batch_size, seq_len, hidden_dim)
        hidden_state = jnp.broadcast_to(hidden_state,
                                        (batch_size, seq_len, self.hidden_dim))

        # (batch_size, seq_len, attention_dim)
        temp1 = jnp.dot(encoder_outputs, self.W_h) + self.b_h

        # (batch_size, seq_len, attention_dim)
        temp2 = jnp.dot(hidden_state, self.W_c) + self.b_c
        temp_sum = temp1 + temp2
        temp_out = jnp.tanh(temp_sum)
        score = jnp.squeeze(jnp.dot(temp_out, self.V), -1)
        attention_weights = jax.nn.softmax(score)

        context_vector = batch_matmul(jnp.expand_dims(attention_weights, 1),
                                      encoder_outputs)
        context_vector = jnp.squeeze(context_vector, 1)
        return context_vector, attention_weights


class DecoderWithBahdanauAttention:
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 prng_key):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        k1, k2, k3, k4, k5 = random.split(prng_key, 5)

        self.embedding = random.normal(k1, (vocab_size, embed_dim))
        self.attention = BahdanauAttention(hidden_dim, hidden_dim, k2)
        self.lstm = LSTMCell(embed_dim + hidden_dim, hidden_dim, k3)
        self.W_fc = random.normal(k4, (hidden_dim, vocab_size))
        self.b_fc = random.normal(k5, (vocab_size,))

    def __call__(self, input_token, hidden_state, cell_state, encoder_outputs):

        input_embedding = self.embedding[input_token]

        context_vector, attention_weights = self.attention(hidden_state,
                                                           encoder_outputs)

        lstm_input = jnp.concatenate((input_embedding, context_vector),
                                     axis=-1)

        hidden_state, cell_state = self.lstm(lstm_input, hidden_state,
                                             cell_state)

        output = jnp.dot(hidden_state, self.W_fc) + self.b_fc

        return output, hidden_state, cell_state, attention_weights


class MultiHeadAttention:
    def __init__(self, embed_dim: int, num_heads: int, prng_key):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        msg = "Embedding dimension MUST be divisible by number of heads"
        assert embed_dim % num_heads == 0, msg

        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        k1, k2, k3, k4 = random.split(prng_key, 4)

        self.query = random.normal(k1, (embed_dim, embed_dim))
        self.key = random.normal(k2, (embed_dim, embed_dim))
        self.value = random.normal(k3, (embed_dim, embed_dim))

        self.out = random.normal(k4, (embed_dim, embed_dim))

    def __call__(self, x):

        batch_size, seq_len, _ = x.shape

        Q = jnp.dot(x, self.query)
        K = jnp.dot(x, self.key)
        V = jnp.dot(x, self.value)

        # Original shape is (batch_size, seq_len, embed_dim)
        # Reshape to (batch_size, seq_len, num_heads, head_dim)
        # Permute dimensions to (batch_size, num_heads, seq_len, head_dim)
        Q = reshape(Q, (batch_size, seq_len, self.num_heads, self.head_dim))
        Q = jnp.permute_dims(Q, (0, 2, 1, 3))
        K = reshape(K, (batch_size, seq_len, self.num_heads, self.head_dim))
        K = jnp.permute_dims(K, (0, 2, 1, 3))
        V = reshape(V, (batch_size, seq_len, self.num_heads, self.head_dim))
        V = jnp.permute_dims(V, (0, 2, 1, 3))

        # Get attention scores of shape:
        # (batch_size, num_heads, seq_len, seq_len)
        attention_scores = jnp.dot(Q, jnp.matrix_transpose(K)) * self.scale
        attention_weights = jax.nn.softmax(attention_scores)

        # Attention output will have shape:
        # (batch_size, num_heads, seq_len, seq_len) *
        # (batch_size, num_heads, seq_len, head_dim) =
        # (batch_size, num_heads, seq_len, head_dim)
        attention_output = jnp.dot(attention_weights, V)

        # Restore shape to (batch_size, seq_len, num_heads, head_dim)
        attention_output = jnp.permute_dims(attention_output, (0, 2, 1, 3))

        # Reshape output to (batch_size, seq_len, embed_dim)
        attention_output = reshape(attention_output,
                                   (batch_size, seq_len, self.embed_dim))

        output = jnp.dot(attention_output, self.out)
        return output
