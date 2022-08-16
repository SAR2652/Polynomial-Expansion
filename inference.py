from model import load_model
import torch, pickle, main
import pandas as pd
from utils import get_inference_arguments

args = get_inference_arguments()
input_file = args.input_filepath
tok_file = args.tokenizer_filepath
model_path = args.model_path
accelerator = args.accelerator

f = open(tok_file, 'rb')
tokenizer = pickle.load(f)

df = pd.read_csv(input_file)

factors = df['factor'].tolist()
print('Number of Factors = {}'.format(df.shape[0]))

if torch.cuda.is_available():
    accelerator = 'cuda'

hidden_size = 320
    
device = torch.device(accelerator)

model = load_model(tokenizer.vocab_dict, tokenizer.vocab_size, hidden_size, device, model_path)
model.eval()
model = model.to(device)

def expand_polynomial(model, factors, tokenizer, device, max_length=31):
    expansions = []
    for factor in factors:
        input_ids = tokenizer.encode_expression(factor, max_length).view(-1, 1)
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

        expansion = tokenizer.decode_expression(outputs)
        expansions.append(expansion)

    return expansions

expansions = expand_polynomial(model, factors[:20], tokenizer, device)

with open('expansions.txt', 'w') as f:
    for factor, expansion in list(zip(factors[:20], expansions)):
        f.write(f"{factor}={expansion}\n")
