"""
SAPPHIRE fedproto.py 
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# -----------------------------------------------------------------------
# GROUP 1 — Linear family
# -----------------------------------------------------------------------
class Group1LinearModel(nn.Module):
    def __init__(self, model_type, num_features, em_d, num_classes):
        super().__init__()
        self.model_type = model_type
        if model_type == "logistic":
            self.encoder = nn.Sequential(
                nn.Linear(num_features, em_d),
                nn.ReLU() 
            )
        elif model_type == "logistic_bn":
            self.encoder = nn.Sequential(
                nn.Linear(num_features, em_d),
                nn.LayerNorm(em_d)
            )
        else:
            self.encoder = nn.Linear(num_features, em_d)
        self.classifier = nn.Linear(em_d, num_classes)

    def forward(self, x):
        v = self.encoder(x)
        return v, self.classifier(v)

# -----------------------------------------------------------------------
# GROUP 2 — Neural family
# -----------------------------------------------------------------------
class SimpleAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)

class Group2NeuralNetwork(nn.Module):
    def __init__(self, model_type, num_features, em_d, num_classes):
        super().__init__()
        self.model_type = model_type
        if model_type == "mlp":
            self.encoder = nn.Sequential(
                nn.Linear(num_features, 64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, em_d)
            )
        elif model_type == "transformer_lite":
            self.token_proj   = nn.Linear(1, 8)
            self.attention    = SimpleAttention(embed_dim=8, num_heads=2)
            self.flatten_proj = nn.Linear(num_features * 8, em_d)
            self.encoder      = None
        else:
            self.encoder = nn.Linear(num_features, em_d)
        self.classifier = nn.Linear(em_d, num_classes)

    def forward(self, x):
        if self.model_type == "transformer_lite":
            x_tok       = x.unsqueeze(-1)
            tokens      = self.token_proj(x_tok)
            attn_out    = self.attention(tokens)
            flat        = attn_out.reshape(attn_out.size(0), -1)
            v           = self.flatten_proj(flat)
        else:
            v = self.encoder(x)
        return v, self.classifier(v)

# -----------------------------------------------------------------------
# GROUP 3 — Soft tree family
# -----------------------------------------------------------------------
class DepthTwoSoftTree(nn.Module):
    def __init__(self, num_features, em_d):
        super().__init__()
        self.node_root  = nn.Linear(num_features, 1)
        self.node_left  = nn.Linear(num_features, 1)
        self.node_right = nn.Linear(num_features, 1)
        self.leaf_params = nn.ParameterList(
            [nn.Parameter(torch.randn(1, em_d)) for _ in range(4)]
        )
        self.temperature = nn.Parameter(torch.tensor(1.0)) 

    def forward(self, x):
        temp = torch.clamp(self.temperature, min=1e-3)
        p_r  = torch.sigmoid(self.node_root(x) / temp)
        p_l  = torch.sigmoid(self.node_left(x) / temp)
        p_rr = torch.sigmoid(self.node_right(x) / temp)
        
        weights = [p_r * p_l, p_r * (1-p_l), (1-p_r) * p_rr, (1-p_r) * (1-p_rr)]
        return sum(w * lp for w, lp in zip(weights, self.leaf_params))

class Group3Ensemble(nn.Module):
    def __init__(self, model_type, num_features, em_d, num_classes):
        super().__init__()
        self.model_type = model_type
        if model_type == "rf":
            self.num_trees    = 5
            self.feature_drop = nn.Dropout(p=0.3)
        else:  
            self.num_trees    = 1
            self.feature_drop = None
        self.forest     = nn.ModuleList([DepthTwoSoftTree(num_features, em_d) for _ in range(self.num_trees)])
        self.classifier = nn.Linear(em_d, num_classes)

    def forward(self, x):
        if self.model_type == "rf":
            tree_outs = [tree(self.feature_drop(x)) for tree in self.forest]
            v = torch.mean(torch.stack(tree_outs), dim=0)
        else:
            v = self.forest[0](x)
        return v, self.classifier(v)

# -----------------------------------------------------------------------
# Builder
# -----------------------------------------------------------------------
GROUP_MAP = {
    "logistic":           "grp1_linear",
    "logistic_bn":        "grp1_linear",
    "mlp":                "grp2_neural",
    "transformer_lite":   "grp2_neural",
    "decision_tree_soft": "grp3_tree",
    "rf":                 "grp3_tree",
}

def build_local_model(group, model_name, X, y, em_d, num_classes):
    num_features = X.shape[1]
    if group   == "grp1_linear":  model = Group1LinearModel(model_name, num_features, em_d, num_classes)
    elif group == "grp2_neural":  model = Group2NeuralNetwork(model_name, num_features, em_d, num_classes)
    elif group == "grp3_tree":    model = Group3Ensemble(model_name, num_features, em_d, num_classes)
    else: raise ValueError(f"Group {group} not supported.")
    return model, X

# -----------------------------------------------------------------------
# Training + prototype extraction (Statistical Heuristic)
# -----------------------------------------------------------------------
def train_and_extract_prototypes(model, optimizer, parameters, X, y,
                                 em_d, num_classes, current_round=1):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    has_valid_protos = len(parameters) > 0 and np.any(parameters[0] != 0)
    if has_valid_protos:
        global_protos      = torch.tensor(parameters[0].reshape((num_classes, em_d)), dtype=torch.float32).to(device)
        global_protos_norm = F.normalize(global_protos, dim=1)
    else:
        global_protos_norm = None

    # LOSS SETUP
    criterion_cls = nn.CrossEntropyLoss()

    # STATISTICAL HEURISTIC RULE: 1 / sqrt(n) decay
    k = 3.0
    lambda_min, lambda_max = 0.1, 0.9
    lambda_weights = torch.zeros(num_classes, device=device)
    class_counts = np.bincount(y, minlength=num_classes)
    
    for c in range(num_classes):
        count = max(class_counts[c], 1)  # Prevent division by zero
        val = k / np.sqrt(count)
        lambda_weights[c] = np.clip(val, lambda_min, lambda_max)

    # Train on all local data
    train_loader = DataLoader(
        TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long)), 
        batch_size=20, shuffle=True
    )

    # STANDARD TRAINING LOOP
    model.train()
    for _ in range(10):  #  local epochs
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            protos, logits = model(X_batch)
            loss_local = criterion_cls(logits, y_batch)

            if global_protos_norm is not None:
                batch_global_protos = global_protos_norm[y_batch]
                protos_norm = F.normalize(protos, dim=1)

                # Fetch the smooth heuristic weight for the patients in this batch
                batch_lambdas = lambda_weights[y_batch].unsqueeze(1)
                
                mse_per_sample = F.mse_loss(protos_norm, batch_global_protos, reduction='none').mean(dim=1, keepdim=True)
                loss_align = (batch_lambdas * mse_per_sample).mean()
                loss = loss_local + loss_align
            else:
                loss = loss_local

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

    # EXTRACT CENTROIDS
    final_prototypes = np.zeros((num_classes, em_d), dtype=np.float32)
    model.eval()
    X_t = torch.as_tensor(X, dtype=torch.float32).to(device)
    y_t = torch.as_tensor(y, dtype=torch.long).to(device)
    with torch.no_grad():
        embeddings, _ = model(X_t)
        embeddings_norm = F.normalize(embeddings, dim=1)
        for c in np.unique(y):
            if c < num_classes:
                mask = (y_t == c)
                if mask.sum() > 0:
                    centroid = embeddings_norm[mask].mean(dim=0)
                    final_prototypes[c] = F.normalize(centroid, dim=0).cpu().numpy()

    return final_prototypes.flatten()