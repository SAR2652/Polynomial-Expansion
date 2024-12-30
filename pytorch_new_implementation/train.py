import os
import torch
import argparse
import pandas as pd     # type: ignore
import torch.nn as nn
from torch.optim import Adam
from dataset import PolynomialDataset
from torch.utils.data import DataLoader
from common_utils import collate_fn, load_tokenizer
from pytorch_new_implementation.model import Seq2SeqModel


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filepath',
                        help='CSV file containing training data',
                        type=str, default='./output/training.csv')
    parser.add_argument('--embed_dim',
                        help='Dimension of Embeddings',
                        type=int, default=128)
    parser.add_argument('--hidden_dim',
                        help='Hidden layer dimensions',
                        type=int, default=128)
    parser.add_argument('--learning_rate',
                        help='Learning rate for model training',
                        type=float, default=1e-3)
    parser.add_argument('--epochs',
                        help='Number of training epochs',
                        type=int, default=100)
    parser.add_argument('--output_dir',
                        type=str,
                        help='Directory to save output',
                        default='./output')
    parser.add_argument('--batch_size',
                        help='Batch size for model training',
                        type=int, default=1536)
    parser.add_argument('--tokenizer_filepath',
                        type=str,
                        help='Path to tokenizer which is to be used',
                        default='./output/tokenizer.joblib')
    parser.add_argument('--teacher_force_ratio',
                        type=float, default=0.5)
    parser.add_argument('--random_state',
                        help='Random state for weights initialization',
                        type=int, default=42)
    parser.add_argument('--fca',
                        help='Force CPU Acceleration',
                        action='store_true')
    parser.add_argument('--continue_from_ckpt',
                        help='Continue model training from a previous '
                        'checkpoint',
                        action='store_true')
    parser.add_argument('--ckpt_file',
                        help='Checkpoint to continue model training from')
    return parser.parse_args()


def train_model(args):

    input_filepath = args.input_filepath
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    embed_dim = args.embed_dim
    hidden_dim = args.hidden_dim
    learning_rate = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size
    tokenizer_filepath = args.tokenizer_filepath
    teacher_force_ratio = args.teacher_force_ratio
    random_state = args.random_state
    fca = args.fca
    tokenizer = load_tokenizer(tokenizer_filepath)

    torch.manual_seed(random_state)

    accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        accelerator = 'mps'
    if fca and accelerator != 'cpu':
        accelerator = 'cpu'
    device = torch.device(accelerator)

    df = pd.read_csv(input_filepath)
    # df = df.iloc[:25600, :]

    factors = df['factor'].tolist()
    expansions = df['expansion'].tolist()

    train_dataset = PolynomialDataset(factors, expansions, tokenizer,
                                      tokenizer.MAX_SEQUENCE_LENGTH, 'pytorch')
    train_dataloader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=batch_size, collate_fn=collate_fn,
                                  pin_memory=True)

    model = Seq2SeqModel(tokenizer.vocab_size, embed_dim, hidden_dim,
                         hidden_dim, tokenizer.sos_token_id,
                         tokenizer.MAX_SEQUENCE_LENGTH, device)

    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    min_avg_loss = float('inf')
    best_model_path = os.path.join(output_dir, 'best_model.pth')

    for epoch in range(epochs):

        running_loss = 0
        batch_losses = list()

        for i, batch in enumerate(train_dataloader):
            print(f'Batch {i + 1}')

            optimizer.zero_grad()

            inputs, targets, _, _ = batch
            inputs = torch.from_numpy(inputs).type(torch.LongTensor) \
                .to(device, non_blocking=True)
            targets = torch.from_numpy(targets).type(torch.LongTensor) \
                .to(device, non_blocking=True)
            outputs = model(inputs, teacher_force_ratio, targets)

            # get rid of SOS token
            outputs = outputs[:, :, 1:]
            targets = targets[:, 1:]
            # print(outputs.shape)
            # print(targets.shape)

            # Loss values
            loss = criterion(outputs, targets)

            # Backprop
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_losses.append(loss.item())

            if (i + 1) % 10 == 0:
                print(f'Running Loss after {i + 1} batches = '
                      f'{running_loss:.4f}')

        epoch_loss = running_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}: Loss = {epoch_loss:.4f}')

        if epoch_loss < min_avg_loss:
            min_avg_loss = epoch_loss
            state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(state, best_model_path)


def main():
    args = get_arguments()
    train_model(args)


if __name__ == '__main__':
    main()
