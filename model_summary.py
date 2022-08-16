import sys, torch, pickle
sys.path.append('..')
from main import load_file
from model import Encoder, Decoder
from utils import Tokenizer
from pytorch_model_summary import summary

with open('./tokenizers/tokenizer.pickle', 'rb') as tok_binary:
    tokenizer = pickle.load(tok_binary)

hidden_size = 320

device = torch.device('cpu')

max_seq_length = 31
batch_size = 1
encoder = Encoder(tokenizer.vocab_size, hidden_size)
decoder = Decoder(hidden_size, tokenizer.vocab_size)
f = open('network.txt', 'w')
# Encoder
print(summary(encoder, torch.zeros((max_seq_length, batch_size), dtype = torch.long)))
f.write(summary(encoder, torch.zeros((max_seq_length, batch_size), dtype = torch.long)))
# Decoder
print(summary(decoder, torch.zeros((16), dtype = torch.long), torch.zeros((max_seq_length, 16, hidden_size * 2)), torch.zeros((1, 16, hidden_size)), torch.zeros((1, 16, hidden_size), dtype = torch.long)))
f.write(summary(decoder, torch.zeros((16), dtype = torch.long), torch.zeros((max_seq_length, 16, hidden_size * 2)), torch.zeros((1, 16, hidden_size)), torch.zeros((1, 16, hidden_size), dtype = torch.long)))
f.close()