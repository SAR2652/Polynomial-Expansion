import os
import jax
import json
import argparse
import numpy as np
from jax import random
import jax.numpy as jnp
from typing import Union
import orbax.checkpoint as ocp
from flax.core import FrozenDict
from common_utils import load_tokenizer
from jax_implementation.utils import init_train_state
from jax_implementation.model import CrossAttentionModelFLAX


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir',
                        help='Directory containing checkpoints',
                        type=str, default='output/ddp_kv/output/checkpoints')
    parser.add_argument('--output_dir',
                        help='Directory to store output',
                        type=str, default='./output')
    parser.add_argument('--tokenizer_filepath',
                        type=str,
                        help='Path to tokenizer which is to be used',
                        default='./output/tokenizer.joblib')
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
    parser.add_argument('--bidirectional',
                        action='store_true',
                        help='Use bidirectional model')
    parser.add_argument('--quantization_dtype',
                        type=str, default='int8',
                        help='Use float16 quantization instead of int8 (except'
                        ' for embeddings)')
    return parser.parse_args()


def quantize_tensor(tensor: jnp.ndarray, num_bits: int = 8,
                    dtype: str = 'int8'):
    """Quantizes a float32 JAX tensor to int8 using uniform affine
    quantization."""

    if dtype == 'float16':
        return {
            'quantized': tensor.astype(jnp.float16),
            'dtype': 'float16'
        }

    qmin = -2 ** (num_bits - 1)
    qmax = 2 ** (num_bits - 1) - 1

    min_val = jnp.min(tensor)
    max_val = jnp.max(tensor)

    scale = jnp.where(max_val != min_val, (max_val - min_val) / (qmax - qmin),
                      1.0)
    zero_point = jnp.where(
        max_val != min_val,
        jnp.round(qmin - min_val / scale).astype(jnp.int32),
        0
    )

    quantized = jnp.where(
        max_val != min_val,
        jnp.clip(jnp.round(tensor / scale + zero_point),
                 qmin, qmax).astype(jnp.int8),
        jnp.zeros_like(tensor, dtype=jnp.int8)
    )

    return {
        'quantized': quantized,
        'scale': scale,
        'zero_point': zero_point
    }


def recursively_quantize(params: Union[dict, FrozenDict], parent_key: str = '',
                         dtype: str = 'int8') -> dict:
    """Recursively quantizes all JAX float32 arrays except embeddings."""
    quantized_params = {}

    for k, v in params.items():
        full_key = f"{parent_key}/{k}" if parent_key else k

        if isinstance(v, (dict, FrozenDict)):
            quantized_params[k] = recursively_quantize(v, full_key, dtype)

        elif isinstance(v, jnp.ndarray) and v.dtype == jnp.float32:
            if "embedding" in full_key.lower():
                # Keep original embeddings (will be written as float16 later)
                quantized_params[k] = v
            else:
                # Quantize other weights
                quantized_params[k] = quantize_tensor(v, dtype=dtype)
        else:
            # Copy non-float32 values as-is
            quantized_params[k] = v

    return quantized_params


def export_quantized_params_to_bin_json(quantized_params: dict,
                                        output_dir: str, dtype: str):
    os.makedirs(output_dir, exist_ok=True)

    # Flatten the nested dict to key path -> leaf dict
    def flatten_dict(d, parent_key=''):
        items = {}
        for k, v in d.items():
            full_key = f"{parent_key}/{k}" if parent_key else k
            if isinstance(v, dict) or isinstance(v, FrozenDict):
                items.update(flatten_dict(v, full_key))
            else:
                items[full_key] = v
        return items

    flat_params = flatten_dict(quantized_params)

    # print(f'Flat params = {flat_params}')

    # We want to group keys by prefix up to leaf dictionary, so filter keys by
    # removing 'quantized', 'scale', 'zero_point'
    # We'll rebuild leaves by grouping on the part before last '/' + last key
    # is one of ['quantized', 'scale', 'zero_point']

    # Group by leaf prefix
    leaves = {}
    for k, v in flat_params.items():
        *prefix, last = k.split('/')
        prefix_key = '/'.join(prefix)
        if prefix_key not in leaves:
            leaves[prefix_key] = {}
        leaves[prefix_key][last] = v

    # print(f'Leaves = {leaves}')

    # Prepare binary buffer and metadata
    bin_data = bytearray()
    metadata = {}
    offset = 0

    for leaf_name, leaf_data in leaves.items():

        if "embedding" in leaf_name.lower():
            # Keep embeddings in float16
            raw_np = np.array(leaf_data['embedding']).flatten() \
                .astype(np.float16)
            dtype = "float16"
        else:
            quantized = leaf_data['quantized']
            dtype = leaf_data.get('dtype', 'int8')

            if dtype == "float16":
                raw_np = np.array(quantized).flatten().astype(np.float16)
            else:
                raw_np = np.array(quantized).flatten().astype(np.int8)

        size = raw_np.size
        bin_data += raw_np.tobytes()

        metadata_entry = {
            'shape': list(quantized.shape),
            'dtype': dtype,
            'offset': offset,
            'size': size
        }

        if dtype == "int8":
            metadata_entry.update({
                'scale': float(leaf_data['scale']),
                'zero_point': int(leaf_data['zero_point'])
            })

        metadata[leaf_name] = metadata_entry
        offset += raw_np.nbytes

    # print(f'Metadata = {metadata}')

    # Write binary file
    with open(os.path.join(output_dir, f'weights_{dtype}.bin'), 'wb') as f:
        f.write(bin_data)

    # Write metadata JSON
    with open(os.path.join(output_dir, f'metadata_{dtype}.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved {len(leaves)} tensors to {output_dir}/weights.bin and "
          "metadata.json")


def quantize_weights_to_int8(args):

    ckpt_dir = args.ckpt_dir
    output_dir = args.output_dir
    ckpt_dir = os.path.abspath(ckpt_dir)
    tokenizer_filepath = args.tokenizer_filepath
    tokenizer = load_tokenizer(tokenizer_filepath)
    quantization_dtype = args.quantization_dtype

    # hyperparameters
    random_state = args.random_state
    embed_size = args.embed_dim
    hidden_size = args.hidden_dim
    num_heads = args.num_heads
    bidirectional = args.bidirectional

    # initialize checkpoint manager
    checkpoint_manager = ocp.CheckpointManager(ckpt_dir)
    step = checkpoint_manager.latest_step()

    model = CrossAttentionModelFLAX(
        embed_size, hidden_size, tokenizer.vocab_size, num_heads,
        tokenizer.sos_token_id, bidirectional
    )

    # initialize random key and training state
    prng_key = random.PRNGKey(random_state)

    train_state = init_train_state(model, prng_key,
                                   seq_len=tokenizer.MAX_SEQUENCE_LENGTH)

    # get PyTree object to load checkpoint
    abstract_state = jax.tree_util.tree_map(
        ocp.utils.to_shape_dtype_struct, train_state
    )
    state = checkpoint_manager.restore(
        step, args=ocp.args.StandardRestore(abstract_state)
    )
    params = state.params

    # quantize parameters and export to JSON & BIN files
    quantized_params = recursively_quantize(params,
                                            dtype=quantization_dtype)
    export_quantized_params_to_bin_json(quantized_params, output_dir,
                                        quantization_dtype)


def main():
    args = get_arguments()
    quantize_weights_to_int8(args)


if __name__ == "__main__":
    main()
