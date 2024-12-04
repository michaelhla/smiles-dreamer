import re
from typing import List

class SMILESTokenizer:
    def __init__(self):
        # Define token types
        self.atoms = {'B', 'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I', 'H'}
        self.special_atoms = {'c', 'n', 'o', 's', 'p'}  # aromatic atoms
        
        # Define bond symbols
        self.bonds = {'-', '=', '#', ':'}
        
        # Special tokens
        self.special_tokens = {
            '[SOS]': 0,  # Start of sequence
            '[EOS]': 1,  # End of sequence
            '[PAD]': 2,  # Padding token
            '(': 3,      # Opening parenthesis
            ')': 4,      # Closing parenthesis
            '[': 5,      # Opening bracket
            ']': 6,      # Closing bracket
        }
        
        # Build vocabulary
        self.vocab = {}
        self.vocab.update(self.special_tokens)
        current_idx = len(self.special_tokens)
        
        # Add atoms
        for atom in self.atoms:
            self.vocab[atom] = current_idx
            current_idx += 1
        
        # Add aromatic atoms
        for atom in self.special_atoms:
            self.vocab[atom] = current_idx
            current_idx += 1
            
        # Add bonds
        for bond in self.bonds:
            self.vocab[bond] = current_idx
            current_idx += 1
            
        # Create reverse vocabulary
        self.idx2token = {v: k for k, v in self.vocab.items()}
        
    def _find_atom_groups(self, smiles: str) -> List[str]:
        """Find all atom groups in brackets like [NH3+], [OH-], etc."""
        return re.findall(r'\[[^\]]+\]', smiles)
    
    def _add_atom_group_to_vocab(self, atom_group: str):
        """Add new atom group to vocabulary if not present"""
        if atom_group not in self.vocab:
            self.vocab[atom_group] = len(self.vocab)
            self.idx2token[self.vocab[atom_group]] = atom_group
    
    def tokenize(self, smiles: str, max_length: int = 128) -> List[str]:
        """Tokenize SMILES string into list of tokens"""
        # Add SOS token
        tokens = ['[SOS]']
        
        # Find and temporarily replace atom groups
        atom_groups = self._find_atom_groups(smiles)
        placeholder_map = {}
        for i, group in enumerate(atom_groups):
            placeholder = f"PLACEHOLDER{i}"
            placeholder_map[placeholder] = group
            self._add_atom_group_to_vocab(group)
            smiles = smiles.replace(group, placeholder)
        
        # Tokenize the string
        i = 0
        while i < len(smiles):
            # Check for two-character atoms
            if i + 1 < len(smiles) and smiles[i:i+2] in self.atoms:
                tokens.append(smiles[i:i+2])
                i += 2
            # Check for placeholders (atom groups)
            elif any(smiles[i:].startswith(p) for p in placeholder_map):
                matching_placeholder = next(p for p in placeholder_map if smiles[i:].startswith(p))
                tokens.append(placeholder_map[matching_placeholder])
                i += len(matching_placeholder)
            # Single character tokens
            else:
                if smiles[i] in self.atoms or smiles[i] in self.special_atoms or \
                   smiles[i] in self.bonds or smiles[i] in {'(', ')', '[', ']'}:
                    tokens.append(smiles[i])
                i += 1
        
        # Add EOS token
        tokens.append('[EOS]')
        
        # Pad sequence
        tokens.extend(['[PAD]'] * (max_length - len(tokens)))
        
        return tokens[:max_length]
    
    def encode(self, smiles: str, max_length: int = 128) -> List[int]:
        """Convert SMILES string to token indices"""
        tokens = self.tokenize(smiles, max_length)
        return [self.vocab.get(token, self.vocab['[PAD]']) for token in tokens]
    
    def decode(self, indices: List[int]) -> str:
        """Convert token indices back to SMILES string"""
        tokens = [self.idx2token[idx] for idx in indices if idx in self.idx2token]
        # Remove special tokens and join
        valid_tokens = [t for t in tokens if t not in {'[SOS]', '[EOS]', '[PAD]'}]
        return ''.join(valid_tokens)

# Example usage:
if __name__ == "__main__":
    tokenizer = SMILESTokenizer()
    smiles = "C(C(=O)O)[NH3+]"
    tokens = tokenizer.tokenize(smiles)
    indices = tokenizer.encode(smiles)
    reconstructed = tokenizer.decode(indices)
    
    print(f"Original SMILES: {smiles}")
    print(f"Tokens: {tokens}")
    print(f"Indices: {indices}")
    print(f"Reconstructed: {reconstructed}")