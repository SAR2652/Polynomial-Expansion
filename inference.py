from model import load_model
import torch, pickle, main
import pandas as pd
from utils import get_inference_arguments

args = get_inference_arguments()
input_file = args.input_filepath
tok_file = args.tokenizer_filepath
model_path = args.model_path

f = open(tok_file, 'rb')
tokenizer = pickle.load(f)

df = pd.read_csv(input_file)

factors = df['factor'].tolist()

accelerator = 'cpu'
if torch.cuda.is_available():
    accelerator = 'cuda'

hidden_size = 320
    
device = torch.device(accelerator)

model = load_model(tokenizer.vocab_dict, tokenizer.vocab_size, hidden_size, device, model_path)
model.eval()
model = model.to(device)

def expand_polynomial(model, factors, tokenizer, device, max_length=30):
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

        
            
        

# def translate_sentence(model, sentence, german, english, device, max_length=50):
#     # Load german tokenizer
#     spacy_ger = spacy.load("de")

#     # Create tokens using spacy and everything in lower case (which is what our vocab is)
#     if type(sentence) == str:
#         tokens = [token.text.lower() for token in spacy_ger(sentence)]
#     else:
#         tokens = [token.lower() for token in sentence]

#     # Add <SOS> and <EOS> in beginning and end respectively
#     tokens.insert(0, german.init_token)
#     tokens.append(german.eos_token)

#     # Go through each german token and convert to an index
#     text_to_indices = [german.vocab.stoi[token] for token in tokens]

#     # Convert to Tensor
#     sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

#     # Build encoder hidden, cell state
#     with torch.no_grad():
#         outputs_encoder, hiddens, cells = model.encoder(sentence_tensor)

#     outputs = [english.vocab.stoi["<sos>"]]

#     for _ in range(max_length):
#         previous_word = torch.LongTensor([outputs[-1]]).to(device)

#         with torch.no_grad():
#             output, hiddens, cells = model.decoder(
#                 previous_word, outputs_encoder, hiddens, cells
#             )
#             best_guess = output.argmax(1).item()

#         outputs.append(best_guess)

#         # Model predicts it's the end of the sentence
#         if output.argmax(1).item() == english.vocab.stoi["<eos>"]:
#             break

#     translated_sentence = [english.vocab.itos[idx] for idx in outputs]

#     # remove start token
#     return translated_sentence[1:]
