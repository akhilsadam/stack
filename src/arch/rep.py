import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class OrthogonalContinuousSurrogate(nn.Module):
    def __init__(self, num_operators, input_dim, output_dim, embed_dim=16, hidden_dim=128):
        super().__init__()
        if embed_dim < num_operators:
            raise ValueError("embed_dim must be >= num_operators for a strict orthogonal basis.")
            
        # 1. Create and freeze an orthogonal embedding basis
        emb = torch.empty(num_operators, embed_dim)
        nn.init.orthogonal_(emb)
        # Ensure they are exact unit vectors
        emb = F.normalize(emb, p=2, dim=1)
        self.operator_embeddings = nn.Parameter(emb, requires_grad=False)
        
        # 2. The MLP Network
        self.net = nn.Sequential(
            nn.Linear(input_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, query_vector):
        """
        x: Tensor of shape [batch_size, input_dim]
        query_vector: Tensor of shape [batch_size, embed_dim] or [embed_dim]
        """
        # Calculate dot products between the query and our orthogonal basis
        # logits shape: [batch_size, num_operators]
        logits = torch.matmul(query_vector, self.operator_embeddings.t())
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Project probabilities back onto the embedding space
        # blended_emb shape: [batch_size, embed_dim]
        blended_emb = torch.matmul(probs, self.operator_embeddings)
        
        # Concatenate with input data and compute output
        combined = torch.cat([x, blended_emb], dim=-1)
        return self.net(combined), probs


class ContinuousSurrogateWrapper(nn.Module):
    def __init__(self, original_module_list: nn.ModuleList, input_dim: int, output_dim: int, embed_dim: int = 16, hidden_dim: int = 128):
        super().__init__()
        self.original_modules = original_module_list
        self.num_operators = len(original_module_list)
        
        self.surrogate = OrthogonalContinuousSurrogate(
            self.num_operators, input_dim, output_dim, embed_dim, hidden_dim
        )

    def train_and_freeze_surrogate(
        self, 
        input_range: tuple = (-10.0, 10.0), 
        samples_per_operator: int = 10000, 
        epochs: int = 50, 
        batch_size: int = 256, 
        lr: float = 1e-3,
        device: str = "cpu"
    ):
        self.surrogate.to(device)
        self.surrogate.train()
        
        low, high = input_range
        input_dim = self.surrogate.net[0].in_features - self.surrogate.operator_embeddings.size(1)
        
        all_X, all_Y, all_Queries = [], [], []
        
        # 1. Generate data. We scale the target query by a large number (e.g., 50.0) 
        # so the softmax yields a sharp one-hot distribution during base training.
        scale_factor = 50.0 
        
        for i, original_model in enumerate(self.original_modules):
            original_model = original_model.to(device)
            original_model.eval()
            
            X = (high - low) * torch.rand((samples_per_operator, input_dim), device=device) + low
            
            # The "perfect" query for this operator
            target_emb = self.surrogate.operator_embeddings[i] * scale_factor
            Queries = target_emb.unsqueeze(0).expand(samples_per_operator, -1)
            
            with torch.no_grad():
                Y = original_model(X)
                
            all_X.append(X)
            all_Y.append(Y)
            all_Queries.append(Queries)
            
        dataset = TensorDataset(
            torch.cat(all_X, dim=0), 
            torch.cat(all_Queries, dim=0), 
            torch.cat(all_Y, dim=0)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 2. Train the surrogate network
        optimizer = optim.Adam(self.surrogate.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            for batch_x, batch_q, batch_y in dataloader:
                optimizer.zero_grad()
                predictions, _ = self.surrogate(batch_x, batch_q)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                
        # 3. Freeze the surrogate completely
        self.surrogate.eval()
        for param in self.surrogate.parameters():
            param.requires_grad = False
            
        print(f"Surrogate successfully trained and frozen (Final MSE: {loss.item():.6f}).")