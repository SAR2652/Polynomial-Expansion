import time, torch, pickle      # type: ignore
from torch.utils.data import DataLoader # type: ignore
import torch.nn as nn       # type: ignore
from torch import optim     # type: ignore
import pandas as pd         # type: ignore
from dataset import PolynomialDataset
import main, pickle
from model import create_model
from pytorch_implementation.utils import *

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

df = pd.read_csv(input_file)

factors = df['factor'].tolist()
expansions = df['expansion'].tolist()

if accelerator == 'cuda' and not torch.cuda.is_available():
    accelerator = 'cpu'

if accelerator == 'mps' and not torch.backends.mps.is_available() and not torch.backends.mps.is_built():
    accelerator = 'cpu'

device = torch.device(accelerator)
print('Training Accelerator: {}'.format(device))
print('Model will run for {} epochs.'.format(epochs))

train_dataset = PolynomialDataset(factors, expansions, tokenizer, main.MAX_SEQUENCE_LENGTH + 2)
batch_size = 16

train_dataloader = DataLoader(train_dataset, shuffle = True, batch_size = batch_size)

model = create_model(tokenizer.vocab_dict, tokenizer.vocab_size, hidden_size, device)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)   

steps_per_epoch = train_dataset.__len__() // batch_size

def train(model, optimizer, dataloader, epochs, device, print_every):
    model.train()
    epoch_losses = []
    epochwise_running_losses = []
    criterion = nn.CrossEntropyLoss(ignore_index = tokenizer.pad_token_id)
    start = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        running_loss = 0.0
        running_losses = []
        prev_running_loss = 0.0

        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].squeeze(2).to(device)
            # print('Input IDs shape = ', input_ids.shape)
            labels = batch['labels'].squeeze(2).to(device)
            output = model(torch.t(input_ids), torch.t(labels))

            # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
            # doesn't take input in that form. For example if we have MNIST we want to have
            # output to be: (N, 10) and targets just (N). Here we can view it in a similar
            # way that we have output_words * batch_size that we want to send in into
            # our cost function, so we need to do some reshapin. While we're at it
            # Let's also remove the start token while we're at it

            # print('Output Shape = ', output.shape)
            # print('Labels Shape = ', labels.shape)
            
            output = output[1:].reshape(-1, output.shape[2])
            labels = torch.t(labels)[1:].reshape(-1)
            # print('Output Shape = ', output.shape)
            # print('Labels Shape = ', labels.shape)
            loss = criterion(output, labels)
            
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            loss = loss.detach().cpu()

            # print('Current Item Loss = {}'.format(current_loss))
            epoch_loss += loss.item()
            running_loss += loss.item()

            optimizer.step()

            if i > 0 and ((i + 1) * batch_size) % print_every == 0:
                now = time.time()
                hours, rem = divmod(now-start, 3600)
                minutes, seconds = divmod(rem, 60)
                current_running_loss = running_loss - prev_running_loss
                print("Samples Processed = {}, Time Elapsed = {:0>2}:{:0>2}:{:05.2f}, Running Loss = {}".format((i + 1) * 32, int(hours),int(minutes),seconds, current_running_loss))
                running_losses.append(current_running_loss)
                prev_running_loss = running_loss

        print('Training Loss for Epoch {} = {}'.format(epoch, epoch_loss))
        epoch_losses.append(epoch_loss)
        epochwise_running_losses.append(running_losses)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            'epoch_losses': epoch_losses,
            'epochwise_running_losses': epochwise_running_losses
            }, PATH)


train(model, optimizer, train_dataloader, epochs, device, steps_per_epoch)