import os
import jax      # type: ignore
import optax    # type: ignore
import joblib   # type: ignore
import argparse
import pandas as pd      # type: ignore
from jax import random      # type: ignore
import jax.numpy as jnp     # type: ignore
from dataset import PolynomialDataset
from common_utils import load_tokenizer
from torch.utils.data import DataLoader     # type: ignore
from jax_implementation.model import create_model
from optax import softmax_cross_entropy_with_integer_labels     # type: ignore


def get_training_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filepath',
                        type=str, help='Path to Input File',
                        default='./output/training.csv')
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
    parser.add_argument('--learning_rate',
                        type=int,
                        help='Learning Rate at which the model is to be '
                        'trained', default=2e-4)
    parser.add_argument('--output_dir',
                        type=str,
                        help='Directory to save output',
                        default='./output')
    parser.add_argument('--batch_size',
                        help='Batch size for model training',
                        type=int, default=128)
    parser.add_argument('--epochs',
                        type=int,
                        help='Number of Epochs to train the model',
                        default=50)
    parser.add_argument('--tokenizer_filepath',
                        type=str,
                        help='Path to tokenizer which is to be used',
                        default='./output/tokenizer.joblib')
    return parser.parse_args()


def collect_model_params(model):

    encoder_params = (
        # Encoder Params
        model.encoder.embedding,
        model.encoder.lstm.cell.U_f,
        model.encoder.lstm.cell.b1_f,
        model.encoder.lstm.cell.W_f,
        model.encoder.lstm.cell.b2_f,
        model.encoder.lstm.cell.U_g,
        model.encoder.lstm.cell.b1_g,
        model.encoder.lstm.cell.W_g,
        model.encoder.lstm.cell.b2_g,
        model.encoder.lstm.cell.U_i,
        model.encoder.lstm.cell.b1_i,
        model.encoder.lstm.cell.W_i,
        model.encoder.lstm.cell.b2_i,
        model.encoder.lstm.cell.U_o,
        model.encoder.lstm.cell.b1_o,
        model.encoder.lstm.cell.W_o,
        model.encoder.lstm.cell.b2_o,
    )

    decoder_params = (
        # Decoder Params
        model.decoder.embedding,
        model.decoder.attention.W_h,
        model.decoder.attention.b_h,
        model.decoder.attention.W_c,
        model.decoder.attention.b_c,
        model.decoder.attention.V,

        model.decoder.lstm.U_f,
        model.decoder.lstm.b1_f,
        model.decoder.lstm.W_f,
        model.decoder.lstm.b2_f,
        model.decoder.lstm.U_g,
        model.decoder.lstm.b1_g,
        model.decoder.lstm.W_g,
        model.decoder.lstm.b2_g,
        model.decoder.lstm.U_i,
        model.decoder.lstm.b1_i,
        model.decoder.lstm.W_i,
        model.decoder.lstm.b2_i,
        model.decoder.lstm.U_o,
        model.decoder.lstm.b1_o,
        model.decoder.lstm.W_o,
        model.decoder.lstm.b2_o,

        model.decoder.W_fc,
        model.decoder.b_fc
    )

    model_params = (encoder_params, decoder_params)

    return model_params


def train_step(batch, model, optimizer, optimizer_state, model_params):

    input_ids = jnp.asarray(batch['input_ids'], dtype=jnp.int32)
    target_ids = jnp.asarray(batch['target_ids'], dtype=jnp.int32)

    def criterion(model_params):

        decoder_hidden_state, decoder_cell_state, encoder_outputs = \
            model.encoder(input_ids)

        decoder_input = target_ids[:, 0]
        _, target_len = target_ids.shape

        loss = 0

        for t in range(1, target_len):

            logits, decoder_hidden_state, decoder_cell_state, _ = \
                model.decoder(decoder_input, decoder_hidden_state,
                              decoder_cell_state, encoder_outputs)

            ground_truth = target_ids[:, t]

            loss += softmax_cross_entropy_with_integer_labels(
                logits, ground_truth
            ).mean()

            decoder_input = ground_truth

        return loss / (target_len - 1)

    grad_criterion = jax.value_and_grad(criterion)

    loss, grads = grad_criterion(model_params)

    updates, new_optimizer_state = optimizer.update(grads, optimizer_state)
    model_params = optax.apply_updates(model_params, updates)
    return loss, model_params, new_optimizer_state


def train_model(epochs, number_of_batches, embed_size, hidden_size, vocab_dict,
                learning_rate, random_state, train_dataloader, output_dir):

    prng_key = random.PRNGKey(random_state)
    model = create_model(vocab_dict, embed_size, hidden_size,
                         prng_key)
    optimizer = optax.adam(learning_rate)
    model_params = collect_model_params(model)
    optimizer_state = optimizer.init(model_params)

    best_loss = float('inf')

    for epoch in range(epochs):
        total_loss = 0

        for i, batch in enumerate(train_dataloader):

            loss, model_params, optimizer_state = train_step(
                batch, model, optimizer, optimizer_state,
                model_params
            )

            total_loss += loss

            if (i + 1) % (number_of_batches // 10) == 0:
                print(f'Processed {i + 1} batches!')

        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model_params(model_params, output_dir, str(epoch))

    print(f'Minimum Loss = {best_loss:.4f}')

    return model_params


def save_model_params(model_params: tuple, output_dir: str, epoch: str):

    encoder_parameters, decoder_parameters = model_params

    # save encoder_params
    encoder_params = dict()
    encoder_params['embedding'] = encoder_parameters[0]
    encoder_params['lstm.cell.U_f'] = encoder_parameters[1]
    encoder_params['lstm.cell.b1_f'] = encoder_parameters[2]
    encoder_params['lstm.cell.W_f'] = encoder_parameters[3]
    encoder_params['lstm.cell.b2_f'] = encoder_parameters[4]
    encoder_params['lstm.cell.U_g'] = encoder_parameters[5]
    encoder_params['lstm.cell.b1_g'] = encoder_parameters[6]
    encoder_params['lstm.cell.W_g'] = encoder_parameters[7]
    encoder_params['lstm.cell.b2_g'] = encoder_parameters[8]
    encoder_params['lstm.cell.U_i'] = encoder_parameters[9]
    encoder_params['lstm.cell.b1_i'] = encoder_parameters[10]
    encoder_params['lstm.cell.W_i'] = encoder_parameters[11]
    encoder_params['lstm.cell.b2_i'] = encoder_parameters[12]
    encoder_params['lstm.cell.U_o'] = encoder_parameters[13]
    encoder_params['lstm.cell.b1_o'] = encoder_parameters[14]
    encoder_params['lstm.cell.W_o'] = encoder_parameters[15]
    encoder_params['lstm.cell.b2_o'] = encoder_parameters[16]

    encoder_filepath = os.path.join(output_dir, f'encoder_{epoch}.joblib')
    joblib.dump(encoder_params, encoder_filepath)

    decoder_params = dict()
    decoder_params['embedding'] = decoder_parameters[0]
    decoder_params['attention.W_h'] = decoder_parameters[1]
    decoder_params['attention.b_h'] = decoder_parameters[2]
    decoder_params['attention.W_c'] = decoder_parameters[3]
    decoder_params['attention.b_c'] = decoder_parameters[4]
    decoder_params['attention.V'] = decoder_parameters[5]

    decoder_params['lstm.U_f'] = decoder_parameters[6]
    decoder_params['lstm.b1_f'] = decoder_parameters[7]
    decoder_params['lstm.W_f'] = decoder_parameters[8]
    decoder_params['lstm.b2_f'] = decoder_parameters[9]
    decoder_params['lstm.U_g'] = decoder_parameters[10]
    decoder_params['lstm.b1_g'] = decoder_parameters[11]
    decoder_params['lstm.W_g'] = decoder_parameters[12]
    decoder_params['lstm.b2_g'] = decoder_parameters[13]
    decoder_params['lstm.U_i'] = decoder_parameters[14]
    decoder_params['lstm.b1_i'] = decoder_parameters[15]
    decoder_params['lstm.W_i'] = decoder_parameters[16]
    decoder_params['lstm.b2_i'] = decoder_parameters[17]
    decoder_params['lstm.U_o'] = decoder_parameters[18]
    decoder_params['lstm.b1_o'] = decoder_parameters[19]
    decoder_params['lstm.W_o'] = decoder_parameters[20]
    decoder_params['lstm.b2_o'] = decoder_parameters[21]
    decoder_params['W_fc'] = decoder_parameters[22]
    decoder_params['b_fc'] = decoder_parameters[23]

    decoder_filepath = os.path.join(output_dir, f'decoder_{epoch}.joblib')
    joblib.dump(decoder_params, decoder_filepath)


def train(args):
    input_file = args.input_filepath
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    random_state = args.random_state
    embed_size = args.embed_size
    hidden_size = args.hidden_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size
    tokenizer_filepath = args.tokenizer_filepath
    tokenizer = load_tokenizer(tokenizer_filepath)

    df = pd.read_csv(input_file)

    factors = df['factor'].tolist()
    expansions = df['expansion'].tolist()

    train_dataset = PolynomialDataset(factors, expansions, tokenizer,
                                      tokenizer.MAX_SEQUENCE_LENGTH, 'jax')

    train_dataloader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=batch_size)
    number_of_batches = len(train_dataloader)
    print(f'Number of samples = {number_of_batches * batch_size}')

    model_params = train_model(
        epochs, number_of_batches, embed_size, hidden_size,
        tokenizer.vocab_dict, learning_rate, random_state, train_dataloader,
        output_dir
    )

    save_model_params(model_params, output_dir, 'FT')


def main():
    args = get_training_arguments()
    train(args)


if __name__ == '__main__':
    main()
