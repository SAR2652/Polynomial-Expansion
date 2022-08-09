import re, torch, math, time, argparse

class Tokenizer:
    def __init__(self):
        self.pattern = re.compile(r'\w+|\d|\*{2}|\W')
        self.sos_token = '<s>'
        self.eos_token = '</s>'
        self.pad_token = '<pad>'
        self.sos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2
        self.current_token_idx = self.eos_token_id
        self.vocab_dict = dict()

    def expand_vocabulary(self, expressions):
        """Create Vocabulary, i.e. mapping for each unique token and its corresponding index"""
        self.current_token_idx = self.eos_token_id
        for expression in expressions:
            tokens = self.pattern.findall(expression)
            for token in tokens:
                if token not in self.vocab_dict.keys():
                    self.vocab_dict[token] = self.current_token_idx
                    self.current_token_idx += 1
        self.update_special_tokens()

    def update_special_tokens(self):
        """Update the token IDs of special tokens namely the EOS and PAD tokens"""
        self.vocab_dict[self.eos_token] = self.current_token_idx
        self.eos_token_id = self.current_token_idx
        self.current_token_idx += 1
        self.vocab_dict[self.pad_token] = self.current_token_idx
        self.pad_token_id = self.current_token_idx
        self.current_token_idx += 1
        
    def convert_tokens_to_ids(self, expression):
        """Convert Tokens into their corresponding Integer mappings"""
        tokens = self.pattern.findall(expression)
        input_ids = [self.vocab_dict[token] for token in tokens]
        return input_ids

    def encode(self, factor, expansion, max_seq_length):
        """Tokenize a single factor and its corresponding expansion and append EOS token"""
        factor_input_ids = self.convert_tokens_to_ids(factor)
        expansion_label_ids = self.convert_tokens_to_ids(expansion)
        factor_input_ids.append(self.eos_token_id)
        expansion_label_ids.append(self.eos_token_id)
        factor_padding_length = max_seq_length - len(factor_input_ids)
        expansion_padding_length = max_seq_length - len(expansion_label_ids)
        factor_input_ids.extend([self.pad_token_id] * factor_padding_length)
        expansion_label_ids.extend([self.pad_token_id] * expansion_padding_length)
        return torch.tensor(factor_input_ids, dtype = torch.long), \
            torch.tensor(expansion_label_ids, dtype = torch.long)

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
    parser.add_argument('input_filepath', type=str, help = 'Path to Input File')
    parser.add_argument('--hidden_size', type=int, help = 'Number of Neurons in Hidden Layers', default = 352)
    parser.add_argument('--accelerator', type=str, help = 'Device to Accelerate Training', default = 'cpu')
    parser.add_argument('--learning_rate', type=int, help = 'Learning Rate at which the model is to be trained', default = 2e-4)
    parser.add_argument('--model_path', type=str, help = 'Path at which the model is to be saved', default = './models/encoder_decoder_model.pt')
    parser.add_argument('--epochs', type=int, help = 'Number of Epochs to train the model', default = 20)
    parser.add_argument('--tokenizer_filepath', type=str, help = 'Path to tokenizer which is to be used', default = './tokenizers/tokenizer.pickle')
    return parser.parse_args()

def get_vocabulary_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_filepath', type=str, help = 'Path to Input File')
    parser.add_argument('tokenizer_filepath', type=str, help = 'Path to save tokenizer file', default = './tokenizers/tokenizer.pickle')
    return parser.parse_args()
