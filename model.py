# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from data import load_pcqm4m_lsc, preprocess_batch

class GraphormerAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super(GraphormerAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sp_enc, edge_enc, mask=None):
        batch_size, seq_len, _ = x.size()
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores with spatial and edge encodings
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores + sp_enc.unsqueeze(1) + edge_enc.unsqueeze(1)  # Add b

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        out = self.out_linear(out)
        return out

class GraphormerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super(GraphormerLayer, self).__init__()
        self.attn = GraphormerAttention(hidden_dim, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sp_enc, edge_enc, mask=None):
        x = self.norm1(x)
        attn_out = self.attn(x, sp_enc, edge_enc, mask)
        x = x + self.dropout(attn_out)
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        return x

class Graphormer(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=128, num_layers=6, num_heads=8, max_degree=100, max_sp=20, edge_dim=3):
        super(Graphormer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        ## Node feature embedding
        self.node_embed = nn.Linear(input_dim, hidden_dim)  # input_dim=9 for PCQM4Mv2

        ## Centrality Encoding
        self.deg_embed = nn.Embedding(max_degree + 1, hidden_dim)

        ## Spatial Encoding
        self.sp_embed = nn.Embedding(max_sp + 2, 1)  # Scalar bias per distance

        ## Edge Encoding
        self.edge_weight = nn.Parameter(torch.randn(edge_dim, hidden_dim))  # edge_dim=3 for PCQM4Mv2
        self.edge_proj = nn.Linear(hidden_dim, 1)  # Project to scalar bias

        ## Transformer layers
        self.layers = nn.ModuleList([
            GraphormerLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])

        ## Virtual Node and Output
        self.vnode = nn.Parameter(torch.randn(1, hidden_dim))
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, data):

        if data["num_nodes"] == 0: # Predict 0 for empty graphs
            batch_size = data["y"].size(0)
            return torch.zeros(batch_size, 1, device=data["y"].device)  
        
        x, edge_index, edge_attr, deg, sp, batch, num_nodes = (
            data["x"], data["edge_index"], data["edge_attr"], data["deg"],
            data["sp"], data["batch"], data["num_nodes"]
        )

        ## Node embedding
        h = self.node_embed(x)

        # Centrality Encoding
        deg_enc = self.deg_embed(deg)
        h = h + deg_enc

        # Spatial Encoding
        sp_enc = self.sp_embed(sp + 1).squeeze(-1)  # [batch, num_nodes, num_nodes]

        # Edge Encoding: Compute dot-products and project
        edge_enc = torch.einsum('bijd,de->bije', edge_attr, self.edge_weight)
        edge_enc = self.edge_proj(edge_enc).squeeze(-1)  # [batch, num_nodes, num_nodes]

        # Add virtual node
        batch_size = batch.max().item() + 1
        vnode = self.vnode.expand(batch_size, -1)
        h = torch.cat([h, vnode], dim=0)

        # Prepare for Transformer
        mask = torch.ones(batch_size, num_nodes + 1, dtype=torch.bool, device=h.device)
        for i in range(batch_size):
            mask[i, :torch.sum(batch == i) + 1] = False

        h = h.view(batch_size, -1, self.hidden_dim)

        # Transformer forward with encodings
        for layer in self.layers:
            h = layer(h, sp_enc, edge_enc, mask)

        # Extract virtual node representation
        h = h[:, -1, :]  # Virtual node is last
        out = self.fc(h)
        return out

if __name__ == "__main__":
    train_loader, _ = load_pcqm4m_lsc(batch_size=2)
    model = Graphormer(input_dim=9, edge_dim=3)
    for batch in train_loader:
        data = preprocess_batch(batch)
        out = model(data)
        print(out.shape)  # [batch_size, 1]
        break