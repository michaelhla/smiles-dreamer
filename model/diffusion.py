import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class SUBSParamaterization(nn.Module):
    def __init__(self, vocab_size: int, mask_token_id: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
    
    def forward(self, logits: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """Apply SUBS parameterization to network outputs
        
        Args:
            logits: Raw logits from network (batch_size, seq_len, vocab_size)
            z_t: Current tokens (batch_size, seq_len)
            
        Returns:
            Modified logits following SUBS constraints
        """
        # Zero masking probabilities - set mask token logit to -inf
        logits[:, :, self.mask_token_id] = float('-inf')
        
        # Carry-over unmasking - copy forward unmasked tokens
        is_masked = (z_t == self.mask_token_id)
        not_masked = ~is_masked
        
        # Create one-hot versions of input tokens
        z_t_one_hot = F.one_hot(z_t, num_classes=self.vocab_size).float()
        
        # Where tokens aren't masked, replace logits with one-hot of input
        logits = torch.where(
            not_masked.unsqueeze(-1).expand_as(logits),
            z_t_one_hot * 1e9,  # Large value to create near-one probability
            logits
        )
        
        return logits

class MDLM(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,  # DiT or similar backbone
        vocab_size: int,
        mask_token_id: int,
        max_seq_len: int
    ):
        super().__init__()
        self.backbone = backbone
        self.subs = SUBSParamaterization(vocab_size, mask_token_id)
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.max_seq_len = max_seq_len
        
    def get_alphas(self, t: torch.Tensor) -> torch.Tensor:
        """Get alpha schedule values for timesteps"""
        # Log-linear schedule as used in paper
        return torch.exp(-t)
    
    def get_alpha_primes(self, t: torch.Tensor) -> torch.Tensor:
        """Get alpha prime (derivative) values for timesteps"""
        return -torch.exp(-t)  # Derivative of log-linear schedule

    def forward(
        self, 
        x: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass computing loss
        
        Args:
            x: Input tokens (batch_size, seq_len)
            t: Timesteps (batch_size,)
            
        Returns:
            Loss value and predicted logits
        """
        batch_size, seq_len = x.shape
        
        # Get diffusion parameters for timestep
        alphas = self.get_alphas(t)
        alpha_primes = self.get_alpha_primes(t)
        
        # Sample z_t by randomly masking tokens
        mask_prob = 1 - alphas
        mask = torch.bernoulli(
            mask_prob.unsqueeze(-1).expand(batch_size, seq_len)
        ).bool()
        z_t = torch.where(mask, self.mask_token_id, x)
        
        # Get network predictions
        logits = self.backbone(z_t, t)
        
        # Apply SUBS parameterization
        logits = self.subs(logits, z_t)
        
        # Compute loss only for masked tokens
        log_probs = F.log_softmax(logits, dim=-1)
        token_losses = F.nll_loss(
            log_probs.view(-1, self.vocab_size),
            x.view(-1),
            reduction='none'
        ).view(batch_size, seq_len)
        
        # Mask loss for unmasked tokens
        token_losses = token_losses * mask
        
        # Weight losses by schedule derivative
        weighted_losses = (alpha_primes / (1 - alphas)).unsqueeze(-1) * token_losses
        
        loss = weighted_losses.mean()
        
        return loss, logits

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        num_steps: int,
        temperature: float = 1.0,
        progress_callback: Optional[callable] = None
    ) -> torch.Tensor:
        """Generate samples using MDLM
        
        Args:
            batch_size: Number of sequences to generate
            num_steps: Number of diffusion steps
            temperature: Sampling temperature
            progress_callback: Optional callback for progress updates
            
        Returns:
            Generated token sequences
        """
        device = next(self.parameters()).device
        
        # Start with all masked tokens
        z = torch.full(
            (batch_size, self.max_seq_len),
            self.mask_token_id,
            device=device
        )
        
        # Sample from t=1 to t=0
        for i in range(num_steps):
            t = 1.0 - (i / num_steps)
            t_batch = torch.full((batch_size,), t, device=device)
            
            # Get network predictions
            logits = self.backbone(z, t_batch)
            logits = self.subs(logits, z)
            
            # Sample new tokens for masked positions
            probs = F.softmax(logits / temperature, dim=-1)
            z_new = torch.multinomial(
                probs.view(-1, self.vocab_size),
                num_samples=1
            ).view(batch_size, self.max_seq_len)
            
            # Only update masked positions
            is_masked = (z == self.mask_token_id)
            z = torch.where(is_masked, z_new, z)
            
            if progress_callback is not None:
                progress_callback(i, num_steps)
                
        return z


class ConditionalMDLM(nn.Module):
    def forward(self, text_ids, smiles_ids, timesteps):
        # Encode text description (no diffusion)
        text_encoding = self.text_encoder(text_ids)
        
        # Apply masking diffusion to SMILES
        alphas = self.get_alphas(timesteps)
        mask_prob = 1 - alphas
        masked_smiles = self.apply_masking(smiles_ids, mask_prob)
        
        # Predict SMILES tokens conditioned on text
        logits = self.dit(
            smiles_sequence=masked_smiles,
            cross_attention_context=text_encoding,
            timesteps=timesteps
        )
        
        # Apply SUBS parameterization
        logits = self.subs(logits, masked_smiles)
        
        # Compute loss only on masked SMILES tokens
        loss = self.compute_masked_loss(logits, smiles_ids, mask_prob)
        
        return loss

    def sample(self, text_ids, num_steps):
        text_encoding = self.text_encoder(text_ids)
        # Start with all masked SMILES
        smiles = torch.full((...), MASK_TOKEN)
        
        for t in reversed(range(num_steps)):
            # Gradually unmask SMILES conditioned on text
            logits = self.dit(smiles, text_encoding, t)
            smiles = self.update_masked_tokens(logits, smiles)
            
        return smiles