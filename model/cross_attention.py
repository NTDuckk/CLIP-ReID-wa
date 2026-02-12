import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionBlock(nn.Module):
    """Self-attention block với residual connections"""
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # Self-attention layer
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=nhead, 
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        # x: (L, B, D)
        
        # Self-attention với residual
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # FFN với residual
        ff_output = self.linear2(self.dropout2(self.activation(self.linear1(x))))
        x = x + ff_output
        x = self.norm2(x)
        
        return x

class CrossAttentionModule(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_self_blocks=2, dropout=0.1):
        super().__init__()
        # Cross-attention layer
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        self.cross_norm = nn.LayerNorm(d_model)
        self.cross_dropout = nn.Dropout(dropout)
        
        # Multiple self-attention blocks (giống PromptSG)
        self.self_blocks = nn.ModuleList([
            SelfAttentionBlock(d_model, nhead, d_model * 4, dropout)
            for _ in range(num_self_blocks)
        ])

    def forward(self, query, key, value, cls_token=None):
        # query: (B, num_patches, D)  - patch tokens
        # key:   (B, num_text_tokens, D) - text tokens
        # value: (B, num_text_tokens, D) - text tokens
        # cls_token: (B, D) - CLS token

        # Cross attention: query (patches) attends to key/value (text)
        attn_output, attn_weights = self.cross_attention(
            query=query,
            key=key,
            value=value
        )
        
        # Residual connection and layer norm cho cross-attention
        x = self.cross_norm(query + self.cross_dropout(attn_output))

        # Reweight patches using attn_weights
        weights = attn_weights.mean(dim=[1, 3])  # (B, num_patches)
        reweighted_patches = weights.unsqueeze(-1) * query  # (B, num_patches, D)

        # Concat CLS with reweighted patches
        if cls_token is not None:
            concat_tokens = torch.cat([cls_token.unsqueeze(1), reweighted_patches], dim=1)  # (B, num_patches+1, D)
        else:
            concat_tokens = reweighted_patches

        # Apply multiple self-attention blocks
        for self_block in self.self_blocks:
            concat_tokens = self_block(concat_tokens)

        return concat_tokens, attn_weights