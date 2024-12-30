import jax      # type: ignore
import joblib   # type: ignore
import argparse
import numpy as np
import jax.numpy as jnp     # type: ignore
from common_utils import load_tokenizer
from jax_implementation.model import create_model


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--factor_input',
                        help='Polynomial that needs to be expanded as a '
                        'string',
                        type=str, default='-8*j*(-8*j-3)')
    parser.add_argument('--encoder_params_file',
                        help='Path to file containing encoder parameters',
                        type=str, default='./output/encoder_5.joblib')
    parser.add_argument('--decoder_params_file',
                        help='Path to file containing encoder parameters',
                        type=str, default='./output/decoder_5.joblib')
    parser.add_argument('--tokenizer_filepath',
                        type=str,
                        help='Path to tokenizer which is to be used',
                        default='./output/tokenizer.joblib')
    parser.add_argument('--hidden_size',
                        type=int,
                        help='Number of Neurons in Hidden Layers',
                        default=64)
    return parser.parse_args()


def forward(factor_input_ids, tokenizer, encoder, encoder_params, decoder,
            decoder_params):

    decoder_hidden_state, decoder_cell_state, encoder_outputs = \
            encoder(encoder_params, factor_input_ids)

    decoder_input = jnp.asarray([tokenizer.sos_token_id])

    def loop_body(t, carry):
        best_guesses, decoder_input, decoder_hidden_state, \
            decoder_cell_state, = carry

        # Decoder forward pass
        logits, decoder_hidden_state, decoder_cell_state, _ = \
            decoder(decoder_input, decoder_hidden_state,
                    decoder_cell_state, encoder_outputs, decoder_params)

        best_guess = jnp.reshape(jnp.argmax(logits, axis=1), -1)
        best_guesses = best_guesses.at[t].set(best_guess[0])

        return best_guesses, best_guess, decoder_hidden_state, \
            decoder_cell_state

    # execute loop body
    best_guesses = jnp.zeros((tokenizer.MAX_SEQUENCE_LENGTH,),
                             dtype=jnp.int32)
    carry = (best_guesses, decoder_input, decoder_hidden_state,
             decoder_cell_state)

    carry = jax.lax.fori_loop(1, tokenizer.MAX_SEQUENCE_LENGTH, loop_body,
                              carry)

    best_guesses, _, _, _ = carry

    return np.array(best_guesses)


def get_prediction(factor_input: str, tokenizer, encoder_params: tuple,
                   decoder_params: tuple, hidden_size: int,
                   inference_function):

    factor_input_ids = tokenizer.encode_expression(
        factor_input, tokenizer.MAX_SEQUENCE_LENGTH
        )

    factor_input_ids = jnp.reshape(
        jnp.asarray(factor_input_ids, dtype=jnp.int32),
        (1, -1)
        )

    encoder, decoder = create_model(hidden_size)

    best_guesses = forward(
        factor_input_ids, tokenizer, encoder, encoder_params, decoder,
        decoder_params
        )
    print(best_guesses)
    output = tokenizer.decode_expression(best_guesses)
    print(f'{factor_input}={output}')

    # print(type(factor_input_ids))


def single_inference(args):

    factor_input = args.factor_input
    encoder_params_file = args.encoder_params_file
    decoder_params_file = args.decoder_params_file
    tokenizer_filepath = args.tokenizer_filepath
    hidden_size = args.hidden_size

    tokenizer = load_tokenizer(tokenizer_filepath)
    encoder_params = joblib.load(encoder_params_file)
    decoder_params = joblib.load(decoder_params_file)

    jit_forward = jax.jit(forward)

    get_prediction(factor_input, tokenizer, encoder_params, decoder_params,
                   hidden_size, jit_forward)


def main():
    args = get_arguments()
    single_inference(args)


if __name__ == '__main__':
    main()
