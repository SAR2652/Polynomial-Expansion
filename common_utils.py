import torch    # type: ignore
import jax.numpy as jnp     # type: ignore
from functools import partial
from typing import Tuple, Iterable


def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """ A helper functions that loads the file into a tuple of strings

    :param file_path: path to the data file
    :return factors: (LHS) inputs to the model
            expansions: (RHS) group truth
    """
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])
    return factors, expansions


class Tokenizer:
    sos_token = '<s>'
    eos_token = '</s>'
    pad_token = '<pad>'
    sos_token_id = 0
    eos_token_id = 1
    pad_token_id = 2
    current_token_idx = 3
    vocab_dict = dict()
    vocab_dict[sos_token] = sos_token_id
    vocab_dict[eos_token] = eos_token_id
    vocab_dict[pad_token] = pad_token_id

    def __init__(self, framework: str='pytorch'):
        self.vocab_size = len(self.vocab_dict)
        self.id_dict = dict((v, k) for k, v in self.vocab_dict.items())
        if framework == 'pytorch':
            tensor_long = partial(torch.tensor, dtype=torch.long)
        elif framework == 'jax':
            tensor_long = partial(jnp.asarray, dtype=jnp.int64)

        self.return_type_convert = tensor_long

    def expand_vocabulary(self, expressions: Iterable):
        """Create Vocabulary, i.e. mapping for each unique token and its
        corresponding index"""
        for expression in expressions:
            tokens = list(expression)
            for token in tokens:
                if token not in self.vocab_dict.keys():
                    self.vocab_dict[token] = self.current_token_idx
                    self.id_dict[self.current_token_idx] = token
                    self.current_token_idx += 1
        self.vocab_size = len(self.vocab_dict)
        
    def convert_tokens_to_ids(self, expression):
        """Convert Tokens into their corresponding Integer mappings"""
        tokens = list(expression)
        input_ids = [self.vocab_dict[token] for token in tokens]
        return input_ids

    def encode(self, factor, expansion, max_seq_length):
        """Tokenize a single factor and its corresponding expansion and
        append EOS token"""
        factor_input_ids = self.convert_tokens_to_ids(factor)
        expansion_label_ids = self.convert_tokens_to_ids(expansion)
        factor_input_ids.insert(self.sos_token_id, 0)
        expansion_label_ids.insert(self.sos_token_id, 0)
        factor_input_ids.append(self.eos_token_id)
        expansion_label_ids.append(self.eos_token_id)
        factor_padding_length = max_seq_length - len(factor_input_ids)
        expansion_padding_length = max_seq_length - len(expansion_label_ids)
        factor_input_ids.extend([self.pad_token_id] * factor_padding_length)
        expansion_label_ids.extend([self.pad_token_id] *
                                   expansion_padding_length)
        factor_inputs = self.return_type_convert(factor_input_ids)
        expansion_inputs = self.return_type_convert(expansion_label_ids)
        return factor_inputs, expansion_inputs

    def encode_expression(self, expression, max_seq_length):
        """Encode a single expression into its corresponding numeric ids"""
        input_ids = self.convert_tokens_to_ids(expression)
        input_ids.insert(self.sos_token_id, 0)
        input_ids.append(self.eos_token_id)
        padding_length = max_seq_length - len(input_ids)
        input_ids.extend([self.pad_token_id] * padding_length)
        return self.return_type_convert(input_ids)

    def decode_expression(self, expression):
        """Convert IDs to their corresponding tokens"""
        special_token_ids = [self.sos_token_id, self.eos_token_id,
                             self.pad_token_id]
        return ''.join([self.id_dict[id] for id in expression if id not in
                        special_token_ids])

    def validate(self):
        for k, v in self.vocab_dict.items():
            if self.id_dict[v] != k:
                return False
        return True


def score(true_expansion: str, pred_expansion: str) -> int:
    """ the scoring function - this is how the model will be evaluated

    :param true_expansion: group truth string
    :param pred_expansion: predicted string
    :return:
    """
    return int(true_expansion == pred_expansion)
