from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from ..tokenizer import SMILESTokenizer


# Load the dataset from HuggingFace
dataset = load_dataset("liupf/chEBI-20-MM")

# Initialize tokenizer (you can replace with your preferred model)
texttokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
smilestokenizer = SMILESTokenizer()

class ChEBIDataset(Dataset):
    def __init__(self, hf_dataset, text_tokenizer, smiles_tokenizer=None, split='train'):
        self.dataset = hf_dataset[split]
        self.text_tokenizer = text_tokenizer
        if smiles_tokenizer == None: 
            self.smiles_tokenizer=text_tokenizer
        else: 
            self.smiles_tokenizer = smiles_tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Get text fields
        smiles = str(item['SMILES'])
        description = str(item['description'])
        inchi = str(item['inchi'])
        iupac = str(item['iupacname'])
        selfies = str(item['SELFIES'])
        
        # Get numerical fields
        cid = item['CID']
        polar_area = float(item['polararea'])
        xlogp = float(item['xlogp'])
        
        smiles_tokens = self.smiles_tokenizer.encode(
            smiles,
            max_length=128
        )
        smiles_encoding = {
            'input_ids': torch.tensor(smiles_tokens, dtype=torch.long),
            'attention_mask': torch.tensor([1 if t != self.smiles_tokenizer.vocab['[PAD]'] else 0 
                                         for t in smiles_tokens], dtype=torch.long)
        }
        
        description_encoding = self.text_tokenizer(
            description,
            padding='max_length',
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )
        
        return {
            'cid': torch.tensor(cid, dtype=torch.long),
            'smiles_ids': smiles_encoding['input_ids'].squeeze(),
            'smiles_mask': smiles_encoding['attention_mask'].squeeze(),
            'description_ids': description_encoding['input_ids'].squeeze(),
            'description_mask': description_encoding['attention_mask'].squeeze(),
            # 'polar_area': torch.tensor(polar_area, dtype=torch.float),
            # 'xlogp': torch.tensor(xlogp, dtype=torch.float),
            # 'inchi': inchi,
            # 'iupac': iupac,
            # 'selfies': selfies
        }

def get_dataloader(batch_size=32, split='train'):
    ds = ChEBIDataset(dataset, tokenizer, split=split)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=4
    )



