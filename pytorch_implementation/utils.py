import torch, math, time, argparse      # type: ignore

class Tokenizer:
    def __init__(self):
        self.sos_token = '<s>'
        self.eos_token = '</s>'
        self.pad_token = '<pad>'
        self.sos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2
        self.current_token_idx = 3
        self.vocab_dict = dict()
        self.vocab_dict[self.sos_token] = self.sos_token_id
        self.vocab_dict[self.eos_token] = self.eos_token_id
        self.vocab_dict[self.pad_token] = self.pad_token_id
        self.vocab_size = len(self.vocab_dict)
        self.id_dict = dict((v, k) for k, v in self.vocab_dict.items())

    def expand_vocabulary(self, expressions):
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
        expansion_label_ids.extend([self.pad_token_id] * expansion_padding_length)
        return torch.tensor(factor_input_ids, dtype = torch.long), \
            torch.tensor(expansion_label_ids, dtype = torch.long)

    def encode_expression(self, expression, max_seq_length):
        """Encode a single expression into its corresponding numeric ids"""
        input_ids = self.convert_tokens_to_ids(expression)
        input_ids.insert(self.sos_token_id, 0)
        input_ids.append(self.eos_token_id)
        padding_length = max_seq_length - len(input_ids)
        input_ids.extend([self.pad_token_id] * padding_length)
        return torch.tensor(input_ids, dtype = torch.long)

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


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def get_training_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_filepath',
                        type=str, help='Path to Input File')
    parser.add_argument('--hidden_size',
                        type=int,
                        help='Number of Neurons in Hidden Layers',
                        default=320)
    parser.add_argument('--accelerator',
                        type=str, help='Device to Accelerate Training',
                        default = 'cpu')
    parser.add_argument('--learning_rate',
                        type=int,
                        help='Learning Rate at which the model is to be '
                        'trained', default=2e-4)
    parser.add_argument('--model_path',
                        type=str,
                        help='Path at which the model is to be saved',
                        default = './models/encoder_decoder_model.pt')
    parser.add_argument('--epochs',
                        type=int,
                        help='Number of Epochs to train the model',
                        default = 20)
    parser.add_argument('--tokenizer_filepath',
                        type=str,
                        help='Path to tokenizer which is to be used',
                        default='./tokenizers/tokenizer.pickle')
    return parser.parse_args()


def get_inference_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_filepath',
                        type=str, help='Path to Input File')
    parser.add_argument('--tokenizer_filepath',
                        type=str,
                        help='Path to load tokenizer file',
                        default='./tokenizers/tokenizer.pickle')
    parser.add_argument('--model_path',
                        type=str,
                        help='Path to saved model state dictionary',
                        default = './models/new_encoder_decoder_model.pt')
    parser.add_argument('--hidden_size',
                        type=str, help='Number of neurons in hidden layer',
                        default = 320)
    parser.add_argument('--accelerator',
                        type=str, help='Device to speed up inference',
                        default = 'cpu')
    return parser.parse_args()
