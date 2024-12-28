import os
import joblib   # type: ignore
import argparse
import numpy as np
import pandas as pd     # type: ignore
from jax import random      # type: ignore
import jax.numpy as jnp     # type: ignore
from dataset import PolynomialDataset
from torch.utils.data import DataLoader
from jax_implementation.model import create_model
from common_utils import load_tokenizer, collate_fn


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filepath',
                        type=str, help='Path to Input File',
                        default='./output/validation.csv')
    parser.add_argument('--random_state',
                        help='Random state for weights initialization',
                        type=int, default=42)
    parser.add_argument('--embed_size',
                        help='Size of embedding',
                        type=int, default=128)
    parser.add_argument('--hidden_size',
                        type=int,
                        help='Number of Neurons in Hidden Layers',
                        default=128)
    parser.add_argument('--output_dir',
                        type=str,
                        help='Directory to save output',
                        default='./output')
    parser.add_argument('--batch_size',
                        help='Batch size for model training',
                        type=int, default=128)
    parser.add_argument('--tokenizer_filepath',
                        type=str,
                        help='Path to tokenizer which is to be used',
                        default='./output/tokenizer.joblib')
    parser.add_argument('--encoder_filepath',
                        help='File that cotains model encoder',
                        type=str, default='./output/encoder_2.joblib')
    parser.add_argument('--decoder_filepath',
                        help='File that contains model decoder',
                        type=str, default='./output/decoder_2.joblib')
    return parser.parse_args()


def load_model(encoder_filepath: str, decoder_filepath: str, model):

    encoder_params = joblib.load(encoder_filepath)

    model.encoder.embedding = encoder_params['embedding']
    model.encoder.lstm.cell.U_f = encoder_params['lstm.cell.U_f']
    model.encoder.lstm.cell.b1_f = encoder_params['lstm.cell.b1_f']
    model.encoder.lstm.cell.W_f = encoder_params['lstm.cell.W_f']
    model.encoder.lstm.cell.b2_f = encoder_params['lstm.cell.b2_f']
    model.encoder.lstm.cell.U_g = encoder_params['lstm.cell.U_g']
    model.encoder.lstm.cell.b1_g = encoder_params['lstm.cell.b1_g']
    model.encoder.lstm.cell.W_g = encoder_params['lstm.cell.W_g']
    model.encoder.lstm.cell.b2_g = encoder_params['lstm.cell.b2_g']
    model.encoder.lstm.cell.U_i = encoder_params['lstm.cell.U_i']
    model.encoder.lstm.cell.b1_i = encoder_params['lstm.cell.b1_i']
    model.encoder.lstm.cell.W_i = encoder_params['lstm.cell.W_i']
    model.encoder.lstm.cell.b2_i = encoder_params['lstm.cell.b2_i']
    model.encoder.lstm.cell.U_o = encoder_params['lstm.cell.U_o']
    model.encoder.lstm.cell.b1_o = encoder_params['lstm.cell.b1_o']
    model.encoder.lstm.cell.W_o = encoder_params['lstm.cell.W_o']
    model.encoder.lstm.cell.b2_o = encoder_params['lstm.cell.b2_o']

    decoder_params = joblib.load(decoder_filepath)

    model.decoder.embedding = decoder_params['embedding']
    model.decoder.attention.W_h = decoder_params['attention.W_h']
    model.decoder.attention.b_h = decoder_params['attention.b_h']
    model.decoder.attention.W_c = decoder_params['attention.W_c']
    model.decoder.attention.b_c = decoder_params['attention.b_c']
    model.decoder.attention.V = decoder_params['attention.V']

    model.decoder.lstm.U_f = decoder_params['lstm.U_f']
    model.decoder.lstm.b1_f = decoder_params['lstm.b1_f']
    model.decoder.lstm.W_f = decoder_params['lstm.W_f']
    model.decoder.lstm.b2_f = decoder_params['lstm.b2_f']
    model.decoder.lstm.U_g = decoder_params['lstm.U_g']
    model.decoder.lstm.b1_g = decoder_params['lstm.b1_g']
    model.decoder.lstm.W_g = decoder_params['lstm.W_g']
    model.decoder.lstm.b2_g = decoder_params['lstm.b2_g']
    model.decoder.lstm.U_i = decoder_params['lstm.U_i']
    model.decoder.lstm.b1_i = decoder_params['lstm.b1_i']
    model.decoder.lstm.W_i = decoder_params['lstm.W_i']
    model.decoder.lstm.b2_i = decoder_params['lstm.b2_i']
    model.decoder.lstm.U_o = decoder_params['lstm.U_o']
    model.decoder.lstm.b1_o = decoder_params['lstm.b1_o']
    model.decoder.lstm.W_o = decoder_params['lstm.W_o']
    model.decoder.lstm.b2_o = decoder_params['lstm.b2_o']

    model.decoder.W_fc = decoder_params['W_fc']
    model.decoder.b_fc = decoder_params['b_fc']

    return model


def get_batched_predictions(batch, model, tokenizer, prng_key, expansions,
                            teacher_force_ratio: float = 0.5):

    input_ids = jnp.asarray(batch['input_ids'], dtype=jnp.int32)
    target_ids = jnp.asarray(batch['target_ids'], dtype=jnp.int32)

    decoder_hidden_state, decoder_cell_state, encoder_outputs = \
        model.encoder(input_ids)

    batch_size = input_ids.shape[0]
    target_len = target_ids.shape[1]

    outputs = jnp.zeros((batch_size, target_len, tokenizer.vocab_size))

    decoder_input = target_ids[:, 0]
    _, target_len = target_ids.shape

    for t in range(1, target_len):

        output, decoder_hidden_state, decoder_cell_state, _ = \
            model.decoder(decoder_input, decoder_hidden_state,
                          decoder_cell_state, encoder_outputs)

        # Store output
        outputs = outputs.at[t].set(output)

        # Get the best guess
        best_guess = jnp.argmax(output, axis=1)

        # Decide whether to use teacher forcing
        prng_key, subkey = random.split(prng_key)
        use_teacher_force = random.uniform(subkey) < teacher_force_ratio

        decoder_input = jnp.where(use_teacher_force, target_ids[:, t],
                                  best_guess)

    for output in outputs:
        print(output)
        expansions.append(
            tokenizer.decode_expression(np.array(output))
        )

    return expansions


def inference(args):

    input_file = args.input_filepath
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    random_state = args.random_state
    embed_size = args.embed_size
    hidden_size = args.hidden_size
    batch_size = args.batch_size
    encoder_filepath = args.encoder_filepath
    decoder_filepath = args.decoder_filepath
    tokenizer_filepath = args.tokenizer_filepath
    tokenizer = load_tokenizer(tokenizer_filepath)

    df = pd.read_csv(input_file)
    df = df.iloc[:1024, :]

    factors = df['factor'].tolist()
    expansions = df['expansion'].tolist()

    val_dataset = PolynomialDataset(factors, expansions, tokenizer,
                                    tokenizer.MAX_SEQUENCE_LENGTH, 'jax')

    val_dataloader = DataLoader(val_dataset, shuffle=False,
                                batch_size=batch_size, collate_fn=collate_fn)

    prng_key = random.PRNGKey(random_state)
    model = create_model(tokenizer.vocab_dict, embed_size, hidden_size,
                         prng_key)

    model = load_model(encoder_filepath, decoder_filepath, model)

    expansions = list()

    for i, batch in enumerate(val_dataloader):

        expansions = get_batched_predictions(
            batch, model, tokenizer, prng_key, expansions
        )

    factors = df['factors'].tolist()

    for factor, expansion in list(zip(factors, expansions)):
        print(f'{factor}={expansion}')


def main():
    args = get_arguments()
    inference(args)


if __name__ == '__main__':
    main()
