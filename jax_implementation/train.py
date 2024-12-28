import os
import jax      # type: ignore
import optax    # type: ignore
import joblib   # type: ignore
import argparse
import pandas as pd      # type: ignore
from jax import random      # type: ignore
import jax.numpy as jnp     # type: ignore
from dataset import PolynomialDataset
from common_utils import load_tokenizer, collate_fn
from torch.utils.data import DataLoader     # type: ignore
from jax_implementation.model import create_model


def get_training_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filepath',
                        type=str, help='Path to Input File',
                        default='./output/training.csv')
    parser.add_argument('--model_parameters_dir',
                        help='Directory to load model checkpoints',
                        type=str, default='./output')
    parser.add_argument('--random_state',
                        help='Random state for weights initialization',
                        type=int, default=42)
    parser.add_argument('--embed_size',
                        help='Size of embedding',
                        type=int, default=32)
    parser.add_argument('--hidden_size',
                        type=int,
                        help='Number of Neurons in Hidden Layers',
                        default=32)
    parser.add_argument('--learning_rate',
                        type=int,
                        help='Learning Rate at which the model is to be '
                        'trained', default=1e-3)
    parser.add_argument('--output_dir',
                        type=str,
                        help='Directory to save output',
                        default='./output')
    parser.add_argument('--batch_size',
                        help='Batch size for model training',
                        type=int, default=512)
    parser.add_argument('--epochs',
                        type=int,
                        help='Number of Epochs to train the model',
                        default=5)
    parser.add_argument('--tokenizer_filepath',
                        type=str,
                        help='Path to tokenizer which is to be used',
                        default='./output/tokenizer.joblib')
    parser.add_argument('--teacher_force_ratio',
                        type=float, default=0.5)
    return parser.parse_args()


def save_model_params(model_params, output_dir: str,
                      epoch: str):

    encoder_params, decoder_params = model_params

    encoder_filepath = os.path.join(output_dir, f'encoder_{epoch}.joblib')
    joblib.dump(encoder_params, encoder_filepath)

    decoder_filepath = os.path.join(output_dir, f'decoder_{epoch}.joblib')
    joblib.dump(decoder_params, decoder_filepath)


def load_model_params(model_parameters_dir: str):

    encoder_filepath = os.path.join(model_parameters_dir,
                                    'encoder_base.joblib')
    encoder_params = joblib.load(encoder_filepath)

    decoder_filepath = os.path.join(model_parameters_dir,
                                    'decoder_base.joblib')
    decoder_params = joblib.load(decoder_filepath)

    model_params = (encoder_params, decoder_params)

    return model_params


def cross_entropy_loss(logits, target):
    """
    Compute the cross-entropy loss between logits and target labels.

    Args:
        logits (Array): Logits of shape (target_len, batch_size, vocab_size).
        target (Array): Target IDs of shape (target_len, batch_size).

    Returns:
        float: Average cross-entropy loss over the batch.
    """
    # Reshape logits and targets for loss calculation
    target_len, batch_size, vocab_size = logits.shape
    logits_flat = logits.reshape(target_len * batch_size, vocab_size)
    target_flat = target.reshape(-1)

    # Compute cross-entropy loss
    log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
    true_log_probs = log_probs[jnp.arange(logits_flat.shape[0]), target_flat]
    loss = -jnp.mean(true_log_probs)
    return loss


def train_step(batch, encoder, decoder, optimizer, optimizer_state,
               model_params, prng_key, vocab_size,
               teacher_force_ratio: float = 0.5):
    """
    Perform a single training step.

    Args:
        batch (tuple): A tuple (input_ids, target_ids).
        encoder (Callable): Encoder model.
        decoder (Callable): Decoder model.
        optimizer (optax.GradientTransformation): Optimizer.
        optimizer_state (optax.OptState): Optimizer state.
        model_params (tuple): A tuple (encoder_params, decoder_params).
        prng_key (jax.random.PRNGKey): RNG key for randomness.
        vocab_size (int): Vocabulary size.
        teacher_force_ratio (float): Probability of using teacher forcing.

    Returns:
        tuple: Loss, updated model parameters, updated optimizer state.
    """
    input_ids, target_ids = batch
    input_ids = jnp.asarray(input_ids, dtype=jnp.int32)
    target_ids = jnp.asarray(target_ids, dtype=jnp.int32)

    def criterion(model_params):

        encoder_params, decoder_params = model_params

        # Encoder forward pass
        decoder_hidden_state, decoder_cell_state, encoder_outputs = \
            encoder(encoder_params, input_ids)

        batch_size, target_len = target_ids.shape
        outputs = jnp.zeros((batch_size, target_len, vocab_size),
                            dtype=jnp.float32)

        decoder_input = target_ids[:, 0]  # First token (<SOS>)

        # Loop through the target sequence
        def loop_body(t, carry):
            outputs, decoder_input, decoder_hidden_state, decoder_cell_state, \
                rng_key = carry

            # Decoder forward pass
            logits, decoder_hidden_state, decoder_cell_state, _ = \
                decoder(decoder_input, decoder_hidden_state,
                        decoder_cell_state, encoder_outputs, decoder_params)

            outputs = outputs.at[:, t, :].set(logits)
            best_guess = jnp.argmax(logits, axis=1)

            # Determine whether to use teacher forcing
            rng_key, subkey = random.split(rng_key)
            use_teacher_force = random.uniform(subkey, shape=(batch_size,)) < \
                teacher_force_ratio
            decoder_input = jnp.where(use_teacher_force, target_ids[:, t],
                                      best_guess)

            return outputs, decoder_input, decoder_hidden_state, \
                decoder_cell_state, rng_key

        # Initialize carry
        rng_key = prng_key
        carry = (outputs, decoder_input, decoder_hidden_state,
                 decoder_cell_state, rng_key)

        # Perform the loop
        carry = jax.lax.fori_loop(1, target_len, loop_body, carry)

        outputs, _, _, _, _ = carry
        return cross_entropy_loss(outputs, target_ids)

    # Compute gradients
    grad_criterion = jax.value_and_grad(criterion, has_aux=False)
    loss, grads = grad_criterion(model_params)

    # Update parameters
    updates, new_optimizer_state = optimizer.update(grads, optimizer_state)
    new_params = optax.apply_updates(model_params, updates)

    return loss, new_params, new_optimizer_state


def train_model(model_params, epochs, number_of_batches,
                hidden_size, vocab_dict,
                learning_rate, random_state, train_dataloader, output_dir,
                teacher_force_ratio):

    prng_key = random.PRNGKey(random_state)
    encoder, decoder = create_model(hidden_size)
    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(model_params)

    best_loss = float('inf')

    for epoch in range(epochs):
        total_loss = 0

        for i, batch in enumerate(train_dataloader):

            loss, model_params, optimizer_state = train_step(
                batch, encoder, decoder, optimizer, optimizer_state,
                model_params, prng_key, len(vocab_dict), teacher_force_ratio
            )

            total_loss += loss

            # if (i + 1) % (number_of_batches // 10) == 0:
            #     print(f'Processed {i + 1} batches!')

        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model_params(model_params, output_dir, str(epoch))

    print(f'Minimum Loss = {best_loss:.4f}')

    return model_params


def train(args):
    input_file = args.input_filepath
    model_parameters_dir = args.model_parameters_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    random_state = args.random_state
    hidden_size = args.hidden_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size
    tokenizer_filepath = args.tokenizer_filepath
    tokenizer = load_tokenizer(tokenizer_filepath)
    teacher_force_ratio = args.teacher_force_ratio

    df = pd.read_csv(input_file)
    df = df.iloc[:25600, :]

    factors = df['factor'].tolist()
    expansions = df['expansion'].tolist()

    train_dataset = PolynomialDataset(factors, expansions, tokenizer,
                                      tokenizer.MAX_SEQUENCE_LENGTH, 'jax')

    train_dataloader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=batch_size, collate_fn=collate_fn)
    number_of_batches = len(train_dataloader)
    # print(f'Number of samples = {number_of_batches * batch_size}')

    model_params = load_model_params(model_parameters_dir)

    model_params = train_model(
        model_params, epochs, number_of_batches, hidden_size,
        tokenizer.vocab_dict, learning_rate, random_state, train_dataloader,
        output_dir, teacher_force_ratio
    )

    save_model_params(model_params, output_dir, 'FT')


def main():
    args = get_training_arguments()
    train(args)


if __name__ == '__main__':
    main()
