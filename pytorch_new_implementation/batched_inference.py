import torch
import argparse
import pandas as pd     # type: ignore
from dataset import PolynomialDataset
from torch.utils.data import DataLoader
from common_utils import load_tokenizer, collate_fn
from pytorch_new_implementation.model import Seq2SeqModel


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filepath',
                        help='CSV file containing training data',
                        type=str, default='./output/training.csv')
    parser.add_argument('--ckpt_filepath',
                        help='Model checkpoint filepath',
                        type=str, default='./output/best_model_272.pth')
    parser.add_argument('--embed_dim',
                        help='Dimension of Embeddings',
                        type=int, default=64)
    parser.add_argument('--hidden_dim',
                        help='Hidden layer dimensions',
                        type=int, default=64)
    parser.add_argument('--batch_size',
                        help='Batch size for model training',
                        type=int, default=5)
    parser.add_argument('--tokenizer_filepath',
                        type=str,
                        help='Path to tokenizer which is to be used',
                        default='./output/tokenizer.joblib')
    parser.add_argument('--random_state',
                        help='Random state for weights initialization',
                        type=int, default=42)
    parser.add_argument('--fca',
                        help='Force CPU Acceleration',
                        action='store_true')
    return parser.parse_args()


def batched_inference(args):
    input_filepath = args.input_filepath
    ckpt_filepath = args.ckpt_filepath
    # output_dir = args.output_dir
    # os.makedirs(output_dir, exist_ok=True)
    embed_dim = args.embed_dim
    hidden_dim = args.hidden_dim
    batch_size = args.batch_size
    tokenizer_filepath = args.tokenizer_filepath

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
    df = df.iloc[:10, :]

    factors = df['factor'].tolist()
    expansions = df['expansion'].tolist()

    train_dataset = PolynomialDataset(factors, expansions, tokenizer,
                                      tokenizer.MAX_SEQUENCE_LENGTH, 'pytorch')
    val_dataloader = DataLoader(train_dataset, shuffle=False,
                                batch_size=batch_size, collate_fn=collate_fn,
                                pin_memory=True)

    model = Seq2SeqModel(tokenizer.vocab_size, embed_dim, hidden_dim,
                         hidden_dim, tokenizer.sos_token_id,
                         tokenizer.MAX_SEQUENCE_LENGTH, device)

    model = model.to(device)

    ckpt = torch.load(ckpt_filepath, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])

    model.eval()

    expressions = list()

    for i, batch in enumerate(val_dataloader):

        inputs, targets, _, _ = batch
        inputs = torch.from_numpy(inputs).type(torch.LongTensor) \
            .to(device, non_blocking=True)
        targets = torch.from_numpy(targets).type(torch.LongTensor) \
            .to(device, non_blocking=True)
        _, best_guesses = model(inputs, eval=True)

        curr_expressions = tokenizer.batch_decode_expressions(best_guesses)
        expressions.extend(curr_expressions)

    factors = df['factor'].tolist()
    for i in range(len(expressions)):
        print(f'{factors[i]}={expressions[i]}')


def main():
    args = get_arguments()
    batched_inference(args)


if __name__ == '__main__':
    main()
