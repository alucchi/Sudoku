import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.ln1 = nn.LayerNorm(embed_dim)
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.Wproj = nn.Linear(embed_dim, embed_dim)  # output projection after attention
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        # x: [batch_size, grid_size, embed_dim] (grid_size = 81 for Sudoku)
        
        normalized = self.ln1(x)
        Q = self.Wq(normalized)  # [batch_size, grid_size, embed_dim]
        K = self.Wk(normalized)  # [batch_size, grid_size, embed_dim]
        V = self.Wv(normalized)  # [batch_size, grid_size, embed_dim]
        
        d = Q.size(-1)  # scalar: embed_dim
        
        # K.transpose(-2, -1): [batch_size, embed_dim, grid_size]
        scores = Q @ K.transpose(-2, -1) / d**0.5
        # scores: [batch_size, grid_size, grid_size] (query cells, key cells)
        
        A = torch.softmax(scores, dim = -1)  # [batch_size, grid_size, grid_size]
                
        attn = A @ V  # [batch_size, grid_size, embed_dim]
        
        attn = self.Wproj(attn)  # [batch_size, grid_size, embed_dim]
    
        out = x + attn  # residual: [batch_size, grid_size, embed_dim]
        # LayerNorm preserves shape; MLP expands to hidden_dim, then returns to embed_dim.
        out = out + self.mlp(self.ln2(out))  # residual: [batch_size, grid_size, embed_dim]
        
        return out

class NeuralNetwork(torch.nn.Module):
    """Attention model with one block per feed-forward width in layer_sizes."""

    def __init__(self, layer_sizes, grid_size=81, n_classes=9, embed_dim=256):
                 
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.n_classes = n_classes
        self.input_projection = nn.Linear(n_classes + 1, embed_dim)
        self.position_embedding = nn.Embedding(grid_size, embed_dim)

        self.blocks = nn.ModuleList([
            AttentionBlock(embed_dim, layer_size) for layer_size in layer_sizes
        ])
        self.Wo = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        
        
        x = x.long().reshape(-1, self.grid_size)
        # One-hot encoding turns each cell value into a binary vector where
        # only the index matching that value is 1; this avoids treating digits
        # as ordered magnitudes.
        X = torch.nn.functional.one_hot(x, num_classes=self.n_classes + 1)
        X = self.input_projection(X.float())
        positions = torch.arange(self.grid_size, device=x.device)
        X = X + self.position_embedding(positions)
        
        for block in self.blocks:
            X = block(X)
                
        return self.Wo(X).transpose(1, 2)  # shape (batch_size, n_classes, grid_size)

    def loss(self, x, y):
        # Targets are digits 1-9, shifted to class indices 0-8.
        logits = self.forward(x)
        return torch.nn.functional.cross_entropy(logits, y - 1)
