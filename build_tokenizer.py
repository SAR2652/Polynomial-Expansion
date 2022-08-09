import pickle
from utils import *
from main import load_file

args = get_vocabulary_arguments()
input_file = args.input_filepath
tokenizer_filepath = args.tokenizer_filepath

factors, expressions = load_file(input_file)

tokenizer = Tokenizer()
tokenizer.expand_vocabulary(factors)
tokenizer.expand_vocabulary(expressions)

with open(tokenizer_filepath, 'wb') as f:
    pickle.dump(tokenizer, f)

print('Successfully built Tokenizer!')
