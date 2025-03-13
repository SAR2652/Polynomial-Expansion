import os
import jax      # type: ignore
import optax    # type: ignore
import argparse
import functools
import pandas as pd      # type: ignore
from jax import random      # type: ignore
import jax.numpy as jnp     # type: ignore
from dataset import PolynomialDataset
from torch.utils.data import DataLoader     # type: ignore
from flax.jax_utils import replicate, unreplicate
from flax.training import train_state, checkpoints
from common_utils import load_tokenizer, collate_fn
from jax_implementation.model import CrossAttentionModelFLAX


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
    parser.add_argument('--embed_dim',
                        help='Size of embedding',
                        type=int, default=64)
    parser.add_argument('--hidden_dim',
                        type=int,
                        help='Number of Neurons in Hidden Layers',
                        default=64)
    parser.add_argument('--num_heads',
                        help='Number of Attention Heads',
                        type=int, default=4)
    parser.add_argument('--learning_rate',
                        type=float,
                        help='Learning Rate at which the model is to be '
                        'trained', default=1e-4)
    parser.add_argument('--output_dir',
                        type=str,
                        help='Directory to save output',
                        default='./output')
    parser.add_argument('--batch_size',
                        help='Batch size for model training',
                        type=int, default=768)
    parser.add_argument('--epochs',
                        type=int,
                        help='Number of Epochs to train the model',
                        default=1000)
    parser.add_argument('--tokenizer_filepath',
                        type=str,
                        help='Path to tokenizer which is to be used',
                        default='./output/tokenizer.joblib')
    parser.add_argument('--teacher_force_ratio',
                        type=float, default=0.5)
    parser.add_argument('--bidirectional',
                        action='store_true',
                        help='Use bidirectional model')
    parser.add_argument('--continue_from_ckpt',
                        action='store_true',
                        help='Continue training from a checkpoint')
    parser.add_argument('--ckpt_file',
                        type=str, default='./output/checkpoint',
                        help='Path to checkpoint file')
    return parser.parse_args()


def init_train_state(model, random_key, batch_size, seq_len, learning_rate
                     ) -> train_state.TrainState:

    dummy_inputs = jnp.ones((batch_size, seq_len),
                            dtype=jnp.int32)
    dummy_targets = jnp.ones((batch_size, seq_len),
                             dtype=jnp.int32)

    # Initialize the Model
    params = model.init(random_key, dummy_inputs, dummy_targets)['params']
    # Create the optimizer
    optimizer = optax.adam(learning_rate)
    # Create a State
    return train_state.TrainState.create(
        apply_fn=model.apply,
        tx=optimizer,
        params=params
    )


@jax.pmap
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


@functools.partial(jax.pmap, axis_name='num_devices')
def train_step(state: train_state.TrainState, inputs: jnp.ndarray,
               targets: jnp.ndarray):

    targets = targets.reshape(targets.shape[0], -1)

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, inputs)
        logits = logits.reshape(logits.shape[0], -1, logits.shape[-1])
        loss = optax.softmax_cross_entropy_with_integer_labels(logits,
                                                               targets)
        print(f'Process Loss = {loss.shape}')
        loss = loss.mean()
        print(f'Process Mean Loss = {loss.shape}')
        return loss, logits

    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = gradient_fn(state.params)

    loss = jax.lax.pmean(loss, axis_name='num_devices')
    print(f'AllReduce Loss = {loss}')
    grads = jax.lax.pmean(grads, axis_name='num_devices')

    return state, loss, grads


def train_model(args):
    input_file = args.input_filepath
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    random_state = args.random_state
    embed_size = args.embed_dim
    hidden_size = args.hidden_dim
    num_heads = args.num_heads
    learning_rate = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size
    tokenizer_filepath = args.tokenizer_filepath
    tokenizer = load_tokenizer(tokenizer_filepath)
    teacher_force_ratio = args.teacher_force_ratio
    bidirectional = args.bidirectional
    continue_from_ckpt = args.continue_from_ckpt
    ckpt_file = args.ckpt_file
    num_devices = jax.local_device_count()
    print(f'Number of Devices = {num_devices}')

    df = pd.read_csv(input_file)

    factors = df['factor'].tolist()
    expansions = df['expansion'].tolist()

    train_dataset = PolynomialDataset(factors, expansions, tokenizer)

    train_dataloader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=batch_size, collate_fn=collate_fn)

    model = CrossAttentionModelFLAX(
        embed_size, hidden_size, tokenizer.vocab_size, num_heads,
        tokenizer.sos_token_id, bidirectional, teacher_force_ratio
    )

    prng_key = random.PRNGKey(random_state)
    dummy_inputs = jnp.ones((batch_size, tokenizer.MAX_SEQUENCE_LENGTH),
                            dtype=jnp.int32)
    dummy_targets = jnp.ones((batch_size, tokenizer.MAX_SEQUENCE_LENGTH),
                             dtype=jnp.int32)
    params = model.init(prng_key, dummy_inputs, dummy_targets)
    param_shapes = jax.tree_map(lambda x: x.shape, params)
    print(f"Model parameter shapes: {param_shapes}")

    state = init_train_state(model, prng_key, batch_size,
                             tokenizer.MAX_SEQUENCE_LENGTH, learning_rate)
    state = replicate(state)

    if continue_from_ckpt and os.path.exists(ckpt_file):
        state = checkpoints.restore_checkpoint(ckpt_file, state)

    name = 'best_model_saca'
    if bidirectional:
        name += '_bidirect'
    name += '_'

    min_loss = float('inf')
    for epoch in range(epochs):

        running_loss = 0
        for i, batch in enumerate(train_dataloader):

            inputs, targets, _, _ = batch
            inputs = jnp.array(inputs, dtype=jnp.int32)
            targets = jnp.array(targets, dtype=jnp.int32)

            inputs = inputs.reshape(num_devices, -1,
                                    tokenizer.MAX_SEQUENCE_LENGTH)
            targets = targets.reshape(num_devices, -1,
                                      tokenizer.MAX_SEQUENCE_LENGTH)

            state, loss, grads = train_step(state, inputs, targets)
            state = update_model(state, grads)

            running_loss += loss.mean().item()
            if (i + 1) % (len(train_dataloader) // 100) == 0:
                print(f'Running Loss after {i + 1} batches = '
                      f'{running_loss:.4f}')

        avg_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

        if avg_loss < min_loss:
            min_loss = avg_loss
            temp_state = unreplicate(state)
            checkpoints.save_checkpoint(output_dir, temp_state, epoch + 1,
                                        name, 1, overwrite=True)


def main():
    args = get_training_arguments()
    train_model(args)


if __name__ == '__main__':
    main()
