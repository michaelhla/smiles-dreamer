import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Optional, Dict, Any

class TextGuidedMDLM(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,  # DiT backbone
        vocab_size: int,
        mask_token_id: int,
        max_seq_len: int,
        guidance_scale: float = 7.5,
        scibert_name: str = "allenai/scibert_scivocab_uncased"
    ):
        super().__init__()
        # MDLM components
        self.mdlm = MDLM(backbone, vocab_size, mask_token_id, max_seq_len)
        
        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(scibert_name)
        self.text_proj = nn.Linear(768, backbone.hidden_size)  # Project SciBERT hidden size to DiT size
        
        self.guidance_scale = guidance_scale
        
    def forward(
        self,
        smiles_ids: torch.Tensor,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        t: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass computing conditional and unconditional losses
        
        Args:
            smiles_ids: Input SMILES tokens (batch_size, seq_len)
            text_ids: Text description tokens (batch_size, text_seq_len)
            text_mask: Attention mask for text (batch_size, text_seq_len)
            t: Timesteps (batch_size,)
        """
        # Get text embeddings
        text_outputs = self.text_encoder(
            input_ids=text_ids,
            attention_mask=text_mask,
            return_dict=True
        )
        text_emb = self.text_proj(text_outputs.last_hidden_state)
        
        # Compute conditional loss
        self.mdlm.backbone.set_cross_attention_context(text_emb)
        cond_loss, cond_logits = self.mdlm(smiles_ids, t)
        
        # Compute unconditional loss
        self.mdlm.backbone.set_cross_attention_context(None)  # Clear context for unconditional
        uncond_loss, uncond_logits = self.mdlm(smiles_ids, t)
        
        return {
            "loss": cond_loss + uncond_loss,
            "cond_logits": cond_logits,
            "uncond_logits": uncond_logits
        }
    
    @torch.no_grad()
    def sample(
        self,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        batch_size: int,
        num_steps: int,
        temperature: float = 1.0,
        progress_callback: Optional[callable] = None
    ) -> torch.Tensor:
        """Generate SMILES sequences conditioned on text descriptions
        
        Args:
            text_ids: Text description tokens (batch_size, text_seq_len)
            text_mask: Attention mask for text (batch_size, text_seq_len)
            batch_size: Number of sequences to generate
            num_steps: Number of diffusion steps
            temperature: Sampling temperature
            progress_callback: Optional callback for progress updates
        """
        device = next(self.parameters()).device
        
        # Get text embeddings
        text_outputs = self.text_encoder(
            input_ids=text_ids,
            attention_mask=text_mask,
            return_dict=True
        )
        text_emb = self.text_proj(text_outputs.last_hidden_state)
        
        # Start with all masked tokens
        z = torch.full(
            (batch_size, self.mdlm.max_seq_len),
            self.mdlm.mask_token_id,
            device=device
        )
        
        # Sample from t=1 to t=0
        for i in range(num_steps):
            t = 1.0 - (i / num_steps)
            t_batch = torch.full((batch_size,), t, device=device)
            
            # Get conditional predictions
            self.mdlm.backbone.set_cross_attention_context(text_emb)
            cond_logits = self.mdlm.backbone(z, t_batch)
            cond_logits = self.mdlm.subs(cond_logits, z)
            
            # Get unconditional predictions
            self.mdlm.backbone.set_cross_attention_context(None)
            uncond_logits = self.mdlm.backbone(z, t_batch)
            uncond_logits = self.mdlm.subs(uncond_logits, z)
            
            # Apply classifier-free guidance
            guided_logits = uncond_logits + self.guidance_scale * (cond_logits - uncond_logits)
            
            # Sample new tokens for masked positions
            probs = torch.softmax(guided_logits / temperature, dim=-1)
            z_new = torch.multinomial(
                probs.view(-1, self.mdlm.vocab_size),
                num_samples=1
            ).view(batch_size, self.mdlm.max_seq_len)
            
            # Only update masked positions
            is_masked = (z == self.mdlm.mask_token_id)
            z = torch.where(is_masked, z_new, z)
            
            if progress_callback is not None:
                progress_callback(i, num_steps)
                
        return z