import jax
import numpy as np
import jax.numpy as jnp
from typing import Union, Tuple, Literal
from torch.utils.data import DataLoader
from flax.training import train_state
from flax.jax_utils import replicate


def eval_step(model, params, inputs):
    print(inputs.shape)
    logits = model.apply({'params': params}, inputs, targets=None, eval=True)
    probs = jax.nn.softmax(logits, axis=-1)
    preds = jnp.argmax(probs, axis=-1)
    return preds, probs


def train_epoch_or_evaluate(
        state_or_model: Union[train_state.TrainState, Tuple],
        dataloader: DataLoader, tokenizer, ddp: bool,
        step_function, update_model=None, num_devices: int = 1,
        mode: Literal["train", "eval", "infer"] = "train",
        max_seq_len: int = 30, vocab_size: int = 31):

    if isinstance(state_or_model, Tuple):
        model, params = state_or_model
        if ddp:
            replicated_params = replicate(params)
            print('Replicated params!')
    else:
        state = state_or_model

    if mode == "train":
        running_loss = 0
        assert update_model is not None, "update_model() must have a " \
            "function as value in 'train' mode"

    if mode in ["eval", "infer"]:
        predictions = np.empty((0, max_seq_len), dtype=np.int32)
        probabilities = np.empty((0, max_seq_len, vocab_size),
                                 dtype=np.float32)

        if mode == "eval":
            ground_truth = list()

    for i, batch in enumerate(dataloader, 0):

        inputs, targets, _, expansions = batch

        if mode != "infer":
            assert all(x is not None for x in targets), \
                "Targets can be None ONLY in inference mode!"

        inputs = jnp.array(inputs, dtype=jnp.int32)
        targets = jnp.array(targets, dtype=jnp.int32)

        if ddp:

            if mode == "train":
                inputs = inputs.reshape(num_devices, -1,
                                        tokenizer.MAX_SEQUENCE_LENGTH)
                targets = targets.reshape(num_devices, -1,
                                          tokenizer.MAX_SEQUENCE_LENGTH)

            else:
                # batch_size = inputs.shape[0]
                inputs = inputs.reshape(num_devices, -1,
                                        tokenizer.MAX_SEQUENCE_LENGTH)

        if mode == "train":
            state, loss, grads = step_function(state, inputs, targets)
            running_loss += loss.mean().item()

            if (i + 1) % (len(dataloader) // 1) == 0:
                print(f'Running Loss after {i + 1} batches = '
                      f'{running_loss:.4f}')

            state = update_model(state, grads)

        else:

            print(inputs.shape)

            if ddp:
                batch_preds, batch_probs = step_function(
                    model, replicated_params, inputs
                )
            else:
                batch_preds, batch_probs = step_function(model, params, inputs)

            batch_preds_np = np.asarray(batch_preds)
            batch_probs_np = np.asarray(batch_probs)

            # Handle DDP output shapes: (num_devices, batch_size_per_device,
            # ...)
            if ddp:
                batch_preds_np = batch_preds_np.reshape(
                    -1, batch_preds_np.shape[-1]
                )
                batch_probs_np = batch_probs_np.reshape(
                    -1, batch_probs_np.shape[-2], batch_probs_np.shape[-1]
                )

            predictions = np.concatenate([predictions, batch_preds_np], axis=0)
            probabilities = np.concatenate([probabilities, batch_probs_np],
                                           axis=0)
            if mode == "eval":
                ground_truth.extend(expansions)

    if mode == "train":
        return state, running_loss
    else:
        return_vals = [predictions, probabilities]

        if mode == "eval":
            return_vals.append(ground_truth)

        return tuple(return_vals)
