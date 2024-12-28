import os
import joblib   # type: ignore
import argparse
from jax import random     # type: ignore
from common_utils import load_tokenizer


def create_lstm_cell_parameters(prng_key, embed_dim, hidden_dim, params):

    k1, k2, k3, k4, k5, k6, k7, k8 = random.split(prng_key, 8)

    params['lstm']['cell'] = dict()

    params['lstm']['cell']['U_f'] = random.normal(k1, (hidden_dim, hidden_dim))
    params['lstm']['cell']['b1_f'] = random.normal(k1, (hidden_dim,))

    params['lstm']['cell']['W_f'] = random.normal(k2, (embed_dim, hidden_dim))
    params['lstm']['cell']['b2_f'] = random.normal(k2, (hidden_dim,))

    params['lstm']['cell']['U_g'] = random.normal(k3, (hidden_dim, hidden_dim))
    params['lstm']['cell']['b1_g'] = random.normal(k3, (hidden_dim,))

    params['lstm']['cell']['W_g'] = random.normal(k4, (embed_dim, hidden_dim))
    params['lstm']['cell']['b2_g'] = random.normal(k4, (hidden_dim,))

    params['lstm']['cell']['U_i'] = random.normal(k5, (hidden_dim, hidden_dim))
    params['lstm']['cell']['b1_i'] = random.normal(k5, (hidden_dim,))

    params['lstm']['cell']['W_i'] = random.normal(k6, (embed_dim, hidden_dim))
    params['lstm']['cell']['b2_i'] = random.normal(k6, (hidden_dim,))

    params['lstm']['cell']['U_o'] = random.normal(k7, (hidden_dim, hidden_dim))
    params['lstm']['cell']['b1_o'] = random.normal(k7, (hidden_dim,))

    params['lstm']['cell']['W_o'] = random.normal(k8, (embed_dim, hidden_dim))
    params['lstm']['cell']['b2_o'] = random.normal(k8, (hidden_dim,))

    return params


def create_lstm_layer_parameters(prng_key, embed_dim, hidden_dim, params):
    params['lstm'] = dict()

    params = create_lstm_cell_parameters(
        prng_key, embed_dim, hidden_dim, params
    )

    return params


def create_encoder_parameters(prng_key, vocab_size, embed_dim, hidden_dim):
    params = dict()

    k1, k2 = random.split(prng_key)
    params['embedding'] = random.normal(k1, (vocab_size, embed_dim))
    params = create_lstm_layer_parameters(k2, embed_dim, hidden_dim, params)

    return params


def create_bahdanau_attention_parameters(prng_key, hidden_dim, attention_dim,
                                         params):

    k1, k2, k3 = random.split(prng_key, 3)

    params['attention'] = dict()

    params['attention']['W_h'] = random.normal(k1, (hidden_dim, attention_dim))
    params['attention']['b_h'] = random.normal(k1, (hidden_dim,))

    params['attention']['W_c'] = random.normal(k2, (hidden_dim, attention_dim))
    params['attention']['b_c'] = random.normal(k2, (hidden_dim,))

    params['attention']['V'] = random.normal(k3, (attention_dim, 1))

    return params


def create_decoder_parameters(prng_key, vocab_size, embed_dim, hidden_dim):

    params = dict()
    k1, k2, k3, k4, k5 = random.split(prng_key, 5)

    params['embedding'] = random.normal(k1, (vocab_size, embed_dim))

    params = create_bahdanau_attention_parameters(
        k2, hidden_dim, hidden_dim, params
    )

    params['lstm'] = dict()
    params = create_lstm_cell_parameters(k3, embed_dim + hidden_dim,
                                         hidden_dim, params)

    params['W_fc'] = random.normal(k4, (hidden_dim, vocab_size))
    params['b_fc'] = random.normal(k5, (vocab_size,))

    return params


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_filepath',
                        help='Path to tokenizer joblib file',
                        type=str, default='./output/tokenizer.joblib')
    parser.add_argument('--embed_dim',
                        help='Dimension of Input Embedding',
                        type=int, default=64)
    parser.add_argument('--hidden_dim',
                        help='Number of neurons in hidden layers',
                        type=int, default=64)
    parser.add_argument('--random_state',
                        help='Random state for initialization',
                        type=int, default=42)
    parser.add_argument('--output_dir',
                        help='Directory to store output',
                        type=str, default='./output')
    return parser.parse_args()


def create_model_parameters(args):

    tokenizer_filepath = args.tokenizer_filepath
    embed_dim = args.embed_dim
    hidden_dim = args.hidden_dim
    random_state = args.random_state
    output_dir = args.output_dir
    prng_key = random.PRNGKey(random_state)
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = load_tokenizer(tokenizer_filepath)
    vocab_size = tokenizer.vocab_size

    k1, k2 = random.split(prng_key, 2)

    encoder_params = create_encoder_parameters(
        k1, vocab_size, embed_dim, hidden_dim
    )

    decoder_params = create_decoder_parameters(
        k2, vocab_size, embed_dim, hidden_dim
    )

    encoder_filepath = os.path.join(output_dir, 'encoder_base.joblib')
    joblib.dump(encoder_params, encoder_filepath)

    decoder_filepath = os.path.join(output_dir, 'decoder_base.joblib')
    joblib.dump(decoder_params, decoder_filepath)


def main():
    args = get_arguments()
    create_model_parameters(args)


if __name__ == '__main__':
    main()
