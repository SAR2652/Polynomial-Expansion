import os
import jax
import argparse
import numpy as np
import pandas as pd
import jax.numpy as jnp
from dataset import PolynomialDataset
from torch.utils.data import DataLoader
from common_utils import load_tokenizer, collate_fn
from jax_implementation.model import CrossAttentionModelFLAX
from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filepath',
                        help='CSV file containing validation data',
                        type=str, default='./validation.csv')
    parser.add_argument('--ckpt_dir',
                        help='Model checkpoint filepath',
                        type=str, default='./output/results/best_model_saca'
                        '_bidirect_217')
    parser.add_argument('--embed_dim',
                        help='Dimension of Embeddings',
                        type=int, default=64)
    parser.add_argument('--hidden_dim',
                        help='Hidden layer dimensions',
                        type=int, default=64)
    parser.add_argument('--batch_size',
                        help='Batch size for inference',
                        type=int, default=1)
    parser.add_argument('--tokenizer_filepath',
                        type=str,
                        help='Path to tokenizer which is to be used',
                        default='./output/tokenizer.joblib')
    parser.add_argument('--random_state',
                        help='Random state for weights initialization',
                        type=int, default=42)
    parser.add_argument('--bidirectional',
                        help='Enable bidirectional mode in LSTM',
                        action='store_true')
    parser.add_argument('--num_heads',
                        help='Number of Attention Heads',
                        type=int, default=4)
    return parser.parse_args()


def batched_inference(args):
    input_filepath = args.input_filepath
    ckpt_dir = os.path.abspath(args.ckpt_dir)
    embed_dim = args.embed_dim
    hidden_dim = args.hidden_dim
    batch_size = args.batch_size
    tokenizer_filepath = args.tokenizer_filepath
    bidirectional = args.bidirectional
    num_heads = args.num_heads

    tokenizer = load_tokenizer(tokenizer_filepath)

    # key = jax.random.PRNGKey(args.random_state)

    df = pd.read_csv(input_filepath)
    df = df.iloc[10:15, :]  # Using a subset of the data

    factors = df['factor'].tolist()
    expansions = df['expansion'].tolist()

    dataset = PolynomialDataset(factors, expansions, tokenizer)
    dataloader = DataLoader(dataset, shuffle=False,
                            batch_size=batch_size, collate_fn=collate_fn)

    # Initialize the Flax model
    model = CrossAttentionModelFLAX(
        embed_dim, hidden_dim, tokenizer.vocab_size, num_heads,
        tokenizer.sos_token_id, bidirectional
    )

    orbax_checkpointer = PyTreeCheckpointer()
    checkpoint_manager = CheckpointManager(ckpt_dir, orbax_checkpointer)
    step = checkpoint_manager.latest_step()
    checkpoint = checkpoint_manager.restore(step)
    state = checkpoint['state']
    params = state['params']

    expressions = []

    for batch in dataloader:
        inputs, targets, f, e = batch

        inputs = jnp.array(inputs, dtype=jnp.int32)
        targets = jnp.array(targets, dtype=jnp.int32)

        # Model inference
        logits = model.apply({'params': params}, inputs)
        # print(f'Logits Shape = {logits.shape}')

        # Softmax and decoding
        probs = jax.nn.softmax(logits, axis=-1)
        # print(f'Probs Shape = {probs.shape}')
        best_guesses_gpu = jnp.argmax(probs, axis=-1)
        # print(f'Best Guesses Shape = {best_guesses_gpu.shape}')
        best_guesses_cpu = np.asarray(best_guesses_gpu)
        # print(f'Best Guesses = {best_guesses_cpu}')

        curr_expressions = tokenizer.batch_decode_expressions(best_guesses_cpu)
        expressions.extend(curr_expressions)

    # Print predictions
    for i in range(len(expressions)):
        print(f'{factors[i]}={expressions[i]}')


def main():
    args = get_arguments()
    batched_inference(args)


if __name__ == '__main__':
    main()
