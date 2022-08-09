import re, torch
from torch.utils.data import Dataset

class PolynomialDataset(Dataset):
    def __init__(self, factors, expansions, tokenizer, max_seq_length):
        self.factors = factors
        self.expansions = expansions
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
    def __len__(self):
        return min(len(self.factors), len(self.expansions))

    def __getitem__(self, idx):
        """Obtain a single tuple comprising of a tokenized factor and its corresponding tokenized expansion"""
        factor = self.factors[idx]
        expansion = self.expansions[idx]
        factor_input_ids, expansion_label_ids = self.tokenizer.encode(factor, expansion, self.max_seq_length)
        item = dict()
        item['factor'] = factor
        item['expansion'] = expansion
        item['input_ids'] = factor_input_ids.view(-1, 1)
        item['labels'] = expansion_label_ids.view(-1, 1)
        item['vocabulary'] = self.tokenizer.vocab_dict
        return item
    
    