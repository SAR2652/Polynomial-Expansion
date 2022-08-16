import sys
import numpy as np
from typing import Tuple
import torch, pickle
from model import load_model


MAX_SEQUENCE_LENGTH = 29
TRAIN_URL = "https://scale-static-assets.s3-us-west-2.amazonaws.com/ml-interview/expand/train.txt"

hidden_size = 320
accelerator = 'cpu'

if torch.cuda.is_available():
    accelerator = 'cuda'
   
device = torch.device(accelerator)

with open('./tokenizers/tokenizer.pickle', 'rb') as tok_binary:
    tokenizer = pickle.load(tok_binary)

model_path = './models/new_encoder_decoder_model.pt'

model = load_model(tokenizer.vocab_dict, tokenizer.vocab_size, hidden_size, device, model_path)
model.eval()
model = model.to(device)
# count = 0

# print('Accelerator = {}'.format(accelerator))

def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """ A helper functions that loads the file into a tuple of strings

    :param file_path: path to the data file
    :return factors: (LHS) inputs to the model
            expansions: (RHS) group truth
    """
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])
    return factors, expansions


def score(true_expansion: str, pred_expansion: str) -> int:
    """ the scoring function - this is how the model will be evaluated

    :param true_expansion: group truth string
    :param pred_expansion: predicted string
    :return:
    """
    return int(true_expansion == pred_expansion)


# --------- START OF IMPLEMENT THIS --------- #
def predict(factor: str):
    max_length = MAX_SEQUENCE_LENGTH + 2
    input_ids = tokenizer.encode_expression(factor, max_length).view(-1, 1)
    input_ids = input_ids.to(device)
    
    with torch.no_grad():
        outputs_encoder, hiddens, cells = model.encoder(input_ids)
        
    outputs = [tokenizer.sos_token_id]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)
            
        with torch.no_grad():
            output, hiddens, cells = model.decoder(previous_word, outputs_encoder, hiddens, cells)
            
        best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == tokenizer.eos_token_id:
            break
    
    # global count
    # if count > 0 and (count + 1) % 100000 == 0:
    #     print('{} samples processed'.format(count + 1))

    # count += 1
    expansion = tokenizer.decode_expression(outputs)
    return expansion


# --------- END OF IMPLEMENT THIS --------- #


def main(filepath: str):
    factors, expansions = load_file(filepath)
    pred = [predict(f) for f in factors]
    scores = [score(te, pe) for te, pe in zip(expansions, pred)]
    print(np.mean(scores))


if __name__ == "__main__":
    main("test.txt" if "-t" in sys.argv else "train.txt")