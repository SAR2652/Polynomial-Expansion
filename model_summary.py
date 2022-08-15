import sys, torch
sys.path.append('..')
from main import load_file
from model import Encoder, Decoder, Seq2Seq
from utils import Tokenizer
from pytorch_model_summary import summary

factors, expressions = load_file('./data/train.txt')
tokenizer = Tokenizer()
tokenizer.expand_vocabulary(factors)
tokenizer.expand_vocabulary(expressions)

hidden_size = 320

device = torch.device('cpu')

max_seq_length = 31
batch_size = 1
encoder = Encoder(tokenizer.current_token_idx, hidden_size)
decoder = Decoder(hidden_size, tokenizer.current_token_idx)
model = Seq2Seq(encoder, decoder, tokenizer.vocab_dict, device)
print(summary(encoder, torch.zeros((max_seq_length, batch_size), dtype = torch.long)))
f = open('network.txt', 'w')
f.write(summary(encoder, torch.zeros((max_seq_length, batch_size), dtype = torch.long)))
# decoder = AttentionDecoder(hidden_size, tokenizer.current_token_idx)
print(summary(decoder, torch.zeros((16), dtype = torch.long), torch.zeros((max_seq_length, 16, hidden_size * 2)), torch.zeros((1, 16, hidden_size)), torch.zeros((1, 16, hidden_size), dtype = torch.long)))
f.write(summary(decoder, torch.zeros((16), dtype = torch.long), torch.zeros((max_seq_length, 16, hidden_size * 2)), torch.zeros((1, 16, hidden_size)), torch.zeros((1, 16, hidden_size), dtype = torch.long)))
f.close()