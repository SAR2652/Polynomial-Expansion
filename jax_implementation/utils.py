import jax
import functools
import numpy as np
import jax.numpy as jnp
from typing import Union, Tuple, Literal
from torch.utils.data import DataLoader
from flax.training import train_state
from flax.jax_utils import replicate


@functools.partial(jax.jit, static_argnums=0)
def eval_step(model, params, inputs):
    logits = model.apply({'params': params}, inputs, targets=None, eval=True)
    probs = jax.nn.softmax(logits, axis=-1)
    preds = jnp.argmax(probs, axis=-1)
    return preds, probs


def is_replicated(params):
    """Checks if the parameters are replicated across devices."""
    leaves, _ = jax.tree_util.tree_flatten(params)
    for leaf in leaves:
        if not hasattr(leaf, "shape"):
            continue
        if leaf.shape[0] == jax.local_device_count():
            return True
    return False


def train_epoch_or_evaluate(
        state_or_model: Union[train_state.TrainState, Tuple],
        dataloader: DataLoader, tokenizer, ddp: bool,
        step_function, update_model=None, num_devices: int = 1,
        mode: Literal["train", "eval", "infer"] = "train",
        curr_epoch: int = None, warmup_epochs: int = None):

    if isinstance(state_or_model, tuple):
        model, params = state_or_model
        replicate_flag = False
        if ddp and not is_replicated(params):
            replicated_params = replicate(params)
            replicate_flag = True
    else:
        state = state_or_model

    if mode == "train":
        running_loss = 0
        assert update_model is not None, "update_model() must have a " \
            "function as value in 'train' mode"

    if mode in ["eval", "infer"]:
        predictions_list, probabilities_list = [], []

        if mode == "eval":
            ground_truth_list = list()

    for i, batch in enumerate(dataloader, 0):

        inputs, targets, _, _ = batch

        if mode != "infer":
            assert all(x is not None for x in targets), \
                "Targets can be None ONLY in inference mode!"

        inputs = jnp.array(inputs, dtype=jnp.int32)
        targets = jnp.array(targets, dtype=jnp.int32)

        if ddp:

            inputs = inputs.reshape(num_devices, -1,
                                    tokenizer.MAX_SEQUENCE_LENGTH)

            if mode in ["train", "eval"]:
                targets = targets.reshape(num_devices, -1,
                                          tokenizer.MAX_SEQUENCE_LENGTH)

        if mode == "train":
            state, loss, grads = step_function(state, inputs, targets,
                                               curr_epoch, warmup_epochs)
            running_loss += loss.mean().item()

            if (i + 1) % (len(dataloader) // 100) == 0:
                print(f'Running Loss after {i + 1} batches = '
                      f'{running_loss:.4f}')

            state = update_model(state, grads)

        else:

            if ddp and replicate_flag:
                batch_preds, batch_probs = step_function(
                    model, replicated_params, inputs
                )
            else:
                batch_preds, batch_probs = step_function(model, params, inputs)

            print(f'Processed {i + 1} batches for evaluation')

            # Handle DDP output shapes: (num_devices, batch_size_per_device,
            # ...)
            if ddp:
                batch_preds = batch_preds.reshape(
                    -1, batch_preds.shape[-1]
                )
                batch_probs = batch_probs.reshape(
                    -1, batch_probs.shape[-2], batch_probs.shape[-1]
                )

            predictions_list.append(batch_preds)
            probabilities_list.append(batch_probs)

            if mode == "eval":
                ground_truth_list.extend(targets)

    if mode == "train":
        return state, running_loss
    else:
        predictions_jnp = jnp.concatenate(predictions_list, axis=0)
        probabilities_jnp = jnp.concatenate(probabilities_list, axis=0)

        predictions = np.asarray(jax.device_get(predictions_jnp))
        probabilities = np.asarray(jax.device_get(probabilities_jnp))

        return_vals = [predictions, probabilities]

        if mode == "eval":
            ground_truth_jnp = jnp.concatenate(ground_truth_list, axis=0)
            # print(ground_truth_jnp.shape)
            ground_truth = np.asarray(jax.device_get(ground_truth_jnp))
            # print(ground_truth.shape)
            return_vals.append(ground_truth)

        return tuple(return_vals)
