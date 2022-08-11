import sys, torch
sys.path.append('..')
from new_polynomial.main import load_file
from model import Encoder, Decoder, Seq2Seq
from new_polynomial.utils import Tokenizer
from pytorch_model_summary import summary

factors, expressions = load_file('./data/train.txt')
tokenizer = Tokenizer()
tokenizer.expand_vocabulary(factors)
tokenizer.expand_vocabulary(expressions)

hidden_size = 320

device = torch.device('cpu')

max_seq_length = 30
batch_size = 32
encoder = Encoder(tokenizer.current_token_idx, hidden_size)
decoder = Decoder(hidden_size, tokenizer.current_token_idx)
model = Seq2Seq(encoder, decoder, tokenizer.vocab_dict, device)
print(summary(encoder, torch.zeros((max_seq_length, batch_size), dtype = torch.long)))
f = open('network.txt', 'w')
f.write(summary(encoder, torch.zeros((max_seq_length, batch_size), dtype = torch.long)))
# decoder = AttentionDecoder(hidden_size, tokenizer.current_token_idx)
print(summary(decoder, torch.zeros((1, 1), dtype = torch.long), torch.zeros((max_seq_length, 1, hidden_size * 2)), torch.zeros((1, 1, hidden_size)), torch.zeros((1, 1, hidden_size), dtype = torch.long)))
f.write(summary(decoder, torch.zeros((1, 1), dtype = torch.long), torch.zeros((max_seq_length, 1, hidden_size * 2)), torch.zeros((1, 1, hidden_size)), torch.zeros((1, 1, hidden_size), dtype = torch.long)))
# print(summary(model, torch.zeros((30, 1), dtype = torch.long), torch.zeros((30, 1), dtype = torch.long)))
f.close()