
import torch, pickle, main      # type: ignore
import numpy as np      # type: ignore
import pandas as pd     # type: ignore
from model import load_model
from pytorch_implementation.utils import get_inference_arguments

args = get_inference_arguments()
input_file = args.input_filepath
tok_file = args.tokenizer_filepath
model_path = args.model_path
accelerator = args.accelerator

f = open(tok_file, 'rb')
tokenizer = pickle.load(f)

df = pd.read_csv(input_file)

factors = df['factor'].tolist()
expansions = df['expansion'].values
print('Number of Factors = {}'.format(df.shape[0]))

hidden_size = 320

if accelerator == 'cuda' and not torch.cuda.is_available():
    print('CUDA enabled GPU not available. Switching Inference Acceleration to CPU')
    accelerator = 'cpu'

print('Accelerator = {}'.format(accelerator))
device = torch.device(accelerator)

model = load_model(tokenizer.vocab_dict, tokenizer.vocab_size, hidden_size, device, model_path)
model.eval()
model = model.to(device)

def expand_polynomial(model, factors, tokenizer, device, max_length):
    expansions = []
    for i, factor in enumerate(factors):
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

        expansion = tokenizer.decode_expression(outputs)
        if (i + 1) % 20000 == 0:
            print('{} samples processed.'.format(i + 1))
        expansions.append(expansion)

    return expansions

predictions = expand_polynomial(model, factors, tokenizer, device, main.MAX_SEQUENCE_LENGTH + 2)
predictions = np.array(predictions)
print('Average Score on Validation Data of {} samples = {}'.format(df.shape[0], np.mean((predictions == expansions))))


# with open('expansions.txt', 'w') as f:
#     for factor, expansion in list(zip(factors, predictions)):
#         f.write(f"{factor}={expansion}\n")
