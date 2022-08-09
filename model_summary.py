import sys, torch
sys.path.append('..')
from polynomial.main import load_file
from polynomial.model import Encoder, AttentionDecoder
from polynomial.utils import Tokenizer
from pytorch_model_summary import summary

factors, expressions = load_file('./data/train.txt')
tokenizer = Tokenizer()
tokenizer.expand_vocabulary(factors)
tokenizer.expand_vocabulary(expressions)

hidden_size = 352
f = open('network.txt', 'w')
encoder = Encoder(tokenizer.current_token_idx, hidden_size)
print(summary(encoder, torch.zeros(1, dtype = torch.long)))
f.write(summary(encoder, torch.zeros(1, dtype = torch.long)))
decoder = AttentionDecoder(hidden_size, tokenizer.current_token_idx)
print(summary(decoder, torch.zeros((1, 1), dtype = torch.long), torch.zeros((1, 1, hidden_size)), torch.zeros((30, hidden_size), dtype = torch.long)))
f.write(summary(decoder, torch.zeros((1, 1), dtype = torch.long), torch.zeros((1, 1, hidden_size)), torch.zeros((30, hidden_size), dtype = torch.long)))
f.close()