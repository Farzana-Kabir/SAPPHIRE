import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# -----------------------------------------------------------------------
# GROUP 1 — Dynamic CNN Family
# -----------------------------------------------------------------------
class Group1CNN(nn.Module):
    def __init__(self, model_type, em_d, num_classes, in_channels=1):
        super().__init__()
        self.model_type = model_type
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32) if model_type == "cnn_bn" else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Dynamically adapts to any input resolution
        )
        self.flatten_dim = 64 * 4 * 4
        self.encoder = nn.Linear(self.flatten_dim, em_d)
        self.classifier = nn.Linear(em_d, num_classes)

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        v = self.encoder(f)
        return v, self.classifier(v)

# -----------------------------------------------------------------------
# GROUP 2 — Residual Vision Family
# -----------------------------------------------------------------------
class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        res = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + res)

class Group2ResNet(nn.Module):
    def __init__(self, model_type, em_d, num_classes, in_channels=1):
        super().__init__()
        self.model_type = model_type
        self.prep = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer1 = SimpleResBlock(32)
        self.downsample = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64)
        )
        self.layer2 = SimpleResBlock(64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.encoder = nn.Linear(64, em_d)
        self.classifier = nn.Linear(em_d, num_classes)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.downsample(x)
        x = self.layer2(x)
        x = self.pool(x)
        f = x.view(x.size(0), -1)
        v = self.encoder(f)
        return v, self.classifier(v)

# -----------------------------------------------------------------------
# GROUP 3 — Tabular MLP Family (for non-image inputs)
# -----------------------------------------------------------------------
class Group3TabularMLP(nn.Module):
    def __init__(self, input_dim, em_d, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.encoder = nn.Linear(32, em_d)
        self.classifier = nn.Linear(em_d, num_classes)

    def forward(self, x):
        f = self.net(x)
        v = self.encoder(f)
        return v, self.classifier(v)

# -----------------------------------------------------------------------
# Model Builder
# -----------------------------------------------------------------------
GROUP_MAP = {
    "cnn_lite":       "grp1_vision",
    "cnn_bn":         "grp1_vision",
    "resnet_lite":    "grp2_vision",
    "mlp":            "grp3_tabular"
}

def build_local_model(group, model_name, X, y, em_d, num_classes):
    is_vision = len(X.shape) >= 3
    if is_vision:
        in_channels = X.shape[1] if len(X.shape) == 4 else 1
        if group == "grp1_vision":  
            model = Group1CNN(model_name, em_d, num_classes, in_channels=in_channels)
        elif group == "grp2_vision":  
            model = Group2ResNet(model_name, em_d, num_classes, in_channels=in_channels)
        else:
            model = Group1CNN("cnn_lite", em_d, num_classes, in_channels=in_channels)
    else:
        input_dim = X.shape[1]
        model = Group3TabularMLP(input_dim, em_d, num_classes)

    return model, X

# -----------------------------------------------------------------------
# Training + Prototype Extraction
# -----------------------------------------------------------------------
def train_and_extract_prototypes(model, optimizer, parameters, X, y,
                                 em_d, num_classes, current_round=1):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    has_valid_protos = len(parameters) > 0 and np.any(parameters[0] != 0)
    if has_valid_protos:
        global_protos = torch.tensor(parameters[0].reshape((num_classes, em_d)), dtype=torch.float32).to(device)
        global_protos_norm = F.normalize(global_protos, dim=1)
    else:
        global_protos_norm = None

    criterion_cls = nn.CrossEntropyLoss()

    k = 3.0
    lambda_min, lambda_max = 0.1, 0.9
    lambda_weights = torch.zeros(num_classes, device=device)
    class_counts = np.bincount(y, minlength=num_classes)
    
    for c in range(num_classes):
        count = max(class_counts[c], 1)
        val = k / np.sqrt(count)
        lambda_weights[c] = np.clip(val, lambda_min, lambda_max)

    if len(X.shape) == 3:
        X = np.expand_dims(X, axis=1)

    train_loader = DataLoader(
        TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long)), 
        batch_size=16, shuffle=True
    )

    model.train()
    for _ in range(5):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            protos, logits = model(X_batch)
            loss_local = criterion_cls(logits, y_batch)

            if global_protos_norm is not None:
                batch_global_protos = global_protos_norm[y_batch]
                protos_norm = F.normalize(protos, dim=1)
                batch_lambdas = lambda_weights[y_batch].unsqueeze(1)
                mse_per_sample = F.mse_loss(protos_norm, batch_global_protos, reduction='none').mean(dim=1, keepdim=True)
                loss_align = (batch_lambdas * mse_per_sample).mean()
                loss = loss_local + loss_align
            else:
                loss = loss_local

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

    final_prototypes = np.zeros((num_classes, em_d), dtype=np.float32)
    model.eval()
    X_t = torch.as_tensor(X, dtype=torch.float32).to(device)
    y_t = torch.as_tensor(y, dtype=torch.long).to(device)
    
    with torch.no_grad():
        all_embeddings = []
        for i in range(0, len(X_t), 16):
            batch = X_t[i:i+16]
            emb, _ = model(batch)
            all_embeddings.append(emb)
        embeddings = torch.cat(all_embeddings, dim=0)
        
        embeddings_norm = F.normalize(embeddings, dim=1)
        for c in np.unique(y):
            if c < num_classes:
                mask = (y_t == c)
                if mask.sum() > 0:
                    centroid = embeddings_norm[mask].mean(dim=0)
                    final_prototypes[c] = F.normalize(centroid, dim=0).cpu().numpy()

    return final_prototypes.flatten()