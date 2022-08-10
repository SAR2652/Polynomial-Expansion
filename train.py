
import math, time, torch, random, pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from dataset import PolynomialDataset
from main import load_file
import main, pickle
from model import Encoder, AttentionDecoder
from utils import *

args = get_training_arguments()

input_file = args.input_filepath
hidden_size = args.hidden_size
accelerator = args.accelerator
learning_rate = args.learning_rate
PATH = args.model_path
epochs = args.epochs
teacher_forcing_ratio = 0.5
tokenizer_filepath = args.tokenizer_filepath

tokenizer_binary = open(tokenizer_filepath, 'rb')
tokenizer = pickle.load(tokenizer_binary)

factors, expansions = load_file(input_file)

X_train, X_val, y_train, y_val = train_test_split(factors, expansions, test_size = 0.225, random_state = 42)

if accelerator == 'cuda' and not torch.cuda.is_available():
    accelerator = 'cpu'

if accelerator == 'mps' and not torch.backends.mps.is_available() and not torch.backends.mps.is_built():
    accelerator = 'cpu'

device = torch.device(accelerator)
print('Training Accelerator: {}'.format(device))
train_dataset = PolynomialDataset(X_train, y_train, tokenizer, main.MAX_SEQUENCE_LENGTH)

train_dataloader = DataLoader(train_dataset, shuffle = True, batch_size = 1)

encoder = Encoder(tokenizer.current_token_idx, hidden_size)
decoder = AttentionDecoder(hidden_size, tokenizer.current_token_idx, 0.1, main.MAX_SEQUENCE_LENGTH)

encoder_optimizer = optim.Adam(encoder.parameters(), lr = learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr = learning_rate)

encoder = encoder.to(device)
decoder = decoder.to(device)

def train(encoder, decoder, encoder_optimizer, decoder_optimizer, dataloader, epochs, device):
    encoder.train()
    decoder.train()
    epoch_losses = []
    criterion = nn.NLLLoss()
    start = time.time()

    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        running_loss = 0.0
        prev_running_loss = 0.0

        for i, batch in enumerate(dataloader):
            encoder_hidden = encoder.initHidden(device)
        
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            input_ids = batch['input_ids'].contiguous()[0, :, :].to(device)
            # print('Input IDs shape = ', input_ids.shape)
            labels = batch['labels'].contiguous()[0, :, :].to(device)
            # print('Labels Shape = ', labels.shape)
            input_length = input_ids.size(0)
            target_length = labels.size(0)

            encoder_outputs = torch.zeros(main.MAX_SEQUENCE_LENGTH, encoder.hidden_size, device=device)

            loss = 0

            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_ids[ei], encoder_hidden)

            encoder_outputs[ei] = encoder_output[0, 0]

            decoder_input = torch.tensor([[tokenizer.sos_token_id]], device=device)

            decoder_hidden = encoder_hidden

            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                for di in range(target_length):

                    # print('Decoder Input Shape = ', decoder_input.shape)
                    # print('Decoder Hidden Shape = ', decoder_hidden.shape)
                    # print('Encoder Outputs Shape = ', encoder_outputs.shape)
                    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                    # decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input.to(torch.long), decoder_hidden, encoder_outputs.to(torch.long))
                    # print('Decoder Output Shape = ', decoder_output.shape)
                    # print('target_tensor[di] shape = ', labels[di].shape)
                    loss += criterion(decoder_output, labels[di])
                    decoder_input = labels[di]  # Teacher forcing

            else:
                # Without teacher forcing: use its own predictions as the next input
                for di in range(target_length):
                    # print('Decoder Input Shape = ', decoder_input.shape)
                    # print('Decoder Hidden Shape = ', decoder_hidden.shape)
                    # print('Encoder Outputs Shape = ', encoder_outputs.shape)
                    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                    # decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input.to(torch.long), decoder_hidden, encoder_outputs.to(torch.long))
                    # print('Decoder Output Shape = ', decoder_output.shape)
                    # print('labels[di] shape = ', labels[di].shape)
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze().detach()  # detach from history as input

                    loss += criterion(decoder_output, labels[di])
                    if decoder_input.item() == tokenizer.eos_token_id:
                        break
            
            
            current_loss = loss.item() / target_length 
            # print('Current Item Loss = {}'.format(current_loss))
            epoch_loss += current_loss
            running_loss += current_loss

            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()
            
            # if i > 0 and (i + 1) % 1000 == 0:
                # print('Total Epoch Loss uptil now = {}'.format(epoch_loss))

            if i > 0 and (i + 1) % 5000 == 0:
                now = time.time()
                hours, rem = divmod(now-start, 3600)
                minutes, seconds = divmod(rem, 60)
                print("Time Elapsed = {:0>2}:{:0>2}:{:05.2f}, Running Loss = {}".format(int(hours),int(minutes),seconds, running_loss - prev_running_loss))
                prev_running_loss = running_loss

        print('Training Loss for Epoch {} = {}'.format(epoch_loss))
        epoch_losses.append(epoch_loss)
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
            'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
            'loss': epoch_loss,
            }, PATH)


train(encoder, decoder, encoder_optimizer, decoder_optimizer, train_dataloader, epochs, device)