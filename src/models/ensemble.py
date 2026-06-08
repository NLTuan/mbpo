import torch
import torch.nn as nn
import torch.nn.functional as F

class EnsembleDynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, num_models=7, hidden_dim=200):
        super().__init__()
        self.num_models = num_models
        
        # The output needs to predict the NEXT state and the REWARD.
        # Since it's a probabilistic model, it predicts both the MEAN and the VARIANCE (log_var)
        self.out_dim = state_dim + 1 
        
        # We use a ModuleList to hold our ensemble of independent neural networks
        self.models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.SiLU(), # Swish/SiLU activation is standard for MBPO models
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.out_dim * 2) # * 2 because we need mean AND log_var
            ) for _ in range(self.num_models)
        ])

        # Bounds to prevent the log variance from collapsing to 0 or blowing up to infinity
        self.max_logvar = nn.Parameter(torch.ones(1, self.out_dim) * 0.5, requires_grad=True)
        self.min_logvar = nn.Parameter(torch.ones(1, self.out_dim) * -10.0, requires_grad=True)

    def forward(self, state, action):
        """
        Takes in a state and action, and returns the predicted mean and log_var 
        for all models in the ensemble.
        """
        # 1. Combine the inputs
        inputs = torch.cat([state, action], dim=-1)
        
        # 2. Pass the data through every model in the ensemble independently
        outputs = [model(inputs) for model in self.models]
        
        # 3. Stack the list of outputs into a single giant 3D tensor
        # Shape becomes: (num_models, batch_size, out_dim * 2)
        outputs = torch.stack(outputs, dim=0)
        
        # 4. Split the tensor perfectly in half: the first half is the mean, the second is log_var
        mean, log_var = torch.chunk(outputs, 2, dim=-1)
        
        # 5. Clamp the log_var between our trainable bounds!
        # Instead of torch.clamp (which kills gradients), it's standard MBPO practice 
        # to use the Softplus trick. This acts like a clamp but has a smooth, differentiable curve!
        log_var = self.max_logvar - F.softplus(self.max_logvar - log_var)
        log_var = self.min_logvar + F.softplus(log_var - self.min_logvar)
        
        return mean, log_var

    def compute_loss(self, state, action, next_state, reward):
        """
        Calculates the Gaussian Negative Log-Likelihood (NLL) loss to train the ensemble.
        """
        # 1. The network learns to predict the change in state, which is much easier than absolute state
        target = torch.cat([next_state - state, reward], dim=-1)
        
        # 2. Expand target to broadcast across all models in the ensemble
        # Target shape becomes: (1, batch_size, out_dim)
        target = target.unsqueeze(0)
        
        # 3. Get predictions from the ensemble
        # Shapes are: (num_models, batch_size, out_dim)
        mean, log_var = self.forward(state, action)
        
        # 4. Calculate inverse variance to safely divide without ZeroDivisionError
        inv_var = torch.exp(-log_var)
        
        # 5. Gaussian NLL Loss Formula
        mse_loss = (mean - target) ** 2
        loss = (mse_loss * inv_var) + log_var
        
        # 6. Average over the batch and feature dimensions, then sum across all models
        return loss.mean(dim=(1, 2)).sum()
