import jax      # type: ignore
import jax.numpy as jnp     # type: ignore
from jax import random     # type: ignore
from jax.lax import reshape, batch_matmul     # type: ignore


class LSTMCell:
    def __init__(self):
        pass

    def __call__(self, x_t, h_prev, c_prev, params):

        temp1 = jnp.dot(h_prev, params['lstm']['cell']['U_f']) + \
            params['lstm']['cell']['b1_f']
        temp2 = jnp.dot(x_t, params['lstm']['cell']['W_f']) + \
            params['lstm']['cell']['b1_f']
        temp_sum1 = temp1 + temp2
        f_t = jax.nn.sigmoid(temp_sum1)
        k_t = f_t * c_prev

        temp3 = jnp.dot(h_prev, params['lstm']['cell']['U_g']) + \
            params['lstm']['cell']['b1_g']
        temp4 = jnp.dot(x_t, params['lstm']['cell']['W_g']) + \
            params['lstm']['cell']['b2_g']
        temp_sum2 = temp3 + temp4
        g_t = jnp.tanh(temp_sum2)

        temp5 = jnp.dot(h_prev, params['lstm']['cell']['U_i']) + \
            params['lstm']['cell']['b1_i']
        temp6 = jnp.dot(x_t, params['lstm']['cell']['W_i']) + \
            params['lstm']['cell']['b2_i']
        temp_sum3 = temp5 + temp6
        i_t = jnp.tanh(temp_sum3)

        j_t = g_t * i_t

        c_t = j_t + k_t

        temp7 = jnp.dot(h_prev, params['lstm']['cell']['U_o']) + \
            params['lstm']['cell']['b1_o']
        temp8 = jnp.dot(x_t, params['lstm']['cell']['W_o']) + \
            params['lstm']['cell']['b2_o']
        temp_sum4 = temp7 + temp8
        o_t = jnp.tanh(temp_sum4)

        h_t = o_t * jnp.tanh(c_t)

        return h_t, c_t


class LSTMLayer:
    def __init__(self, hidden_dim):
        self.cell = LSTMCell()
        self.hidden_dim = hidden_dim

    def __call__(self, x, params, h_0=None, c_0=None,):

        batch_size, seq_len, _ = x.shape
        h_t = jnp.zeros((batch_size, self.hidden_dim)) if h_0 is None else h_0
        c_t = jnp.zeros((batch_size, self.hidden_dim)) if c_0 is None else c_0

        outputs = list()

        for t in range(seq_len):
            h_t, c_t = self.cell(x[:, t, :], h_t, c_t, params)
            outputs.append(h_t)

        outputs = jnp.stack(outputs, axis=1)

        return h_t, c_t, outputs


class Encoder:
    def __init__(self, hidden_dim):
        self.lstm = LSTMLayer(hidden_dim)

    def __call__(self, params, x):

        embedding = params['embedding'][x]

        hidden, cell, encoder_outputs = self.lstm(embedding, params)

        return hidden, cell, encoder_outputs


class BahdanauAttention:
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim

    def __call__(self, hidden_state, encoder_outputs, decoder_params):

        batch_size, seq_len, _ = encoder_outputs.shape

        # (batch_size, hidden_dim) -> (batch_size, 1, hidden_dim)
        hidden_state = jnp.expand_dims(hidden_state, 1)
        # (batch_size, 1, hidden_dim) -> (batch_size, seq_len, hidden_dim)
        hidden_state = jnp.broadcast_to(hidden_state,
                                        (batch_size, seq_len, self.hidden_dim))

        # (batch_size, seq_len, attention_dim)
        temp1 = jnp.dot(encoder_outputs, decoder_params['attention']['W_h']) \
            + decoder_params['attention']['b_h']

        # (batch_size, seq_len, attention_dim)
        temp2 = jnp.dot(hidden_state, decoder_params['attention']['W_c']) + \
            decoder_params['attention']['b_c']
        temp_sum = temp1 + temp2
        temp_out = jnp.tanh(temp_sum)
        score = jnp.squeeze(jnp.dot(temp_out,
                                    decoder_params['attention']['V']), -1)
        attention_weights = jax.nn.softmax(score)

        context_vector = batch_matmul(jnp.expand_dims(attention_weights, 1),
                                      encoder_outputs)
        context_vector = jnp.squeeze(context_vector, 1)
        return context_vector, attention_weights


class DecoderWithBahdanauAttention:
    def __init__(self, hidden_dim):
        self.attention = BahdanauAttention(hidden_dim)
        self.lstm = LSTMLayer(hidden_dim)

    def __call__(self, input_token, hidden_state, cell_state, encoder_outputs,
                 params):

        input_embedding = params['embedding'][input_token]

        context_vector, attention_weights = self.attention(
             hidden_state, encoder_outputs, params
        )

        context_vector = jnp.expand_dims(context_vector, axis=1)
        input_embedding = jnp.expand_dims(input_embedding, axis=1)

        lstm_input = jnp.concatenate((input_embedding, context_vector),
                                     axis=-1)

        hidden_state, cell_state, _ = self.lstm(lstm_input, params,
                                                hidden_state, cell_state)

        output = jnp.dot(hidden_state, params['W_fc']) + \
            params['b_fc']

        return output, hidden_state, cell_state, attention_weights


def create_model(hidden_dim):
    encoder = Encoder(hidden_dim)
    decoder = DecoderWithBahdanauAttention(hidden_dim)
    return encoder, decoder


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
