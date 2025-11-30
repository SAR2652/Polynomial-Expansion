import os
import jax
import json
import argparse
import numpy as np
from jax import random
import jax.numpy as jnp
from typing import Union
import orbax.checkpoint as ocp
from src.common_utils import load_tokenizer
from src.jax_implementation.utils import init_train_state
from src.jax_implementation.model import CrossAttentionModelFLAX


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
    return parser.parse_args()


def quantize_tensor(tensor: jnp.ndarray, num_bits=8):
    """Quantizes a float32 JAX tensor to int8 using uniform affine
    quantization."""
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


def quantize_tensor_to_int32(tensor: jnp.ndarray):
    """Quantizes a float32 JAX tensor to int32 using uniform affine
    quantization."""
    # qmin = -2 ** 31
    # qmax = 2 ** 31 - 1
    # For int32, qmax - qmin = 2**32 - 1 = 4,294,967,295, which is too large
    # for float32 precision in JAX's JIT context. This triggers an overflow
    # during the division.

    # Use a safe range like qmin = -2**15, qmax = 2**15 - 1
    # (i.e., int16 range) for scale

    qmin = -2 ** 15
    qmax = 2 ** 15 - 1

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
        jnp.clip(jnp.round(tensor / scale + zero_point), qmin, qmax)
        .astype(jnp.int32),
        jnp.zeros_like(tensor, dtype=jnp.int32)
    )

    return {
        'quantized': quantized,
        'scale': scale,
        'zero_point': zero_point
    }


def recursively_quantize(params: Union[dict], parent_key=''
                         ) -> dict:
    """Recursively quantizes all JAX float32 arrays:
       - embeddings → float16
       - weights → int8
       - bias → int32
    """
    quantized_params = {}

    for k, v in params.items():
        full_key = f"{parent_key}/{k}" if parent_key else k

        if isinstance(v, (dict)):
            quantized_params[k] = recursively_quantize(v, full_key)

        elif isinstance(v, jnp.ndarray) and v.dtype == jnp.float32:
            if "embedding" in full_key.lower():
                quantized_params[k] = v  # keep float32 for float16 export
            elif "bias" in full_key.lower():
                quantized_params[k] = quantize_tensor_to_int32(v)
            else:
                quantized_params[k] = quantize_tensor(v)
        else:
            quantized_params[k] = v

    return quantized_params


def export_quantized_params_to_bin_json(quantized_params, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    bin_data = bytearray()
    metadata = {}
    offset = 0

    def process_node(node, meta_subtree):
        nonlocal bin_data, offset

        for k, v in node.items():
            if isinstance(v, dict) and 'quantized' in v:
                # Determine dtype
                dtype = 'int32' if v['quantized'].dtype == jnp.int32 else \
                    'int8'
                quantized_np = np.array(v['quantized']).flatten().astype(
                    np.int32 if dtype == 'int32' else np.int8)
                size = quantized_np.size
                bin_data += quantized_np.tobytes()

                meta_subtree[k] = {
                    'shape': list(v['quantized'].shape),
                    'scale': float(v['scale']),
                    'zero_point': int(v['zero_point']),
                    'dtype': dtype,
                    'offset': offset,
                    'size': size
                }
                offset += quantized_np.nbytes

            elif isinstance(v, jnp.ndarray):
                # Embedding or float32 weights (e.g., float16 export)
                raw_np = np.array(v).flatten().astype(np.float16)
                size = raw_np.size
                bin_data += raw_np.tobytes()
                scale = np.max(raw_np) - np.min(raw_np)

                meta_subtree[k] = {
                    'shape': list(v.shape),
                    'dtype': "float16",
                    'scale': float(scale),
                    'offset': offset,
                    'size': size
                }

                offset += raw_np.nbytes

            elif isinstance(v, dict):
                # Recurse into nested dict
                meta_subtree[k] = {}
                process_node(v, meta_subtree[k])

            else:
                # Non-array leaf (e.g., constants or unsupported types)
                meta_subtree[k] = str(v)

    process_node(quantized_params, metadata)

    # Write binary file
    with open(os.path.join(output_dir, 'weights.bin'), 'wb') as f:
        f.write(bin_data)

    # Write hierarchical metadata JSON
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

    print("Saved hierarchical metadata and weights to "
          f"{output_dir}/weights.bin and metadata.json")


def quantize_weights_to_int8(args):

    ckpt_dir = args.ckpt_dir
    output_dir = args.output_dir
    ckpt_dir = os.path.abspath(ckpt_dir)
    tokenizer_filepath = args.tokenizer_filepath
    tokenizer = load_tokenizer(tokenizer_filepath)

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
    quantized_params = recursively_quantize(params)
    export_quantized_params_to_bin_json(quantized_params, output_dir)


def main():
    args = get_arguments()
    quantize_weights_to_int8(args)


if __name__ == "__main__":
    main()
