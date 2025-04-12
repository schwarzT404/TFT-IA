"""
Module définissant le réseau de dynamique pour MuZero.
Ce réseau prédit l'état suivant et la récompense à partir de l'état actuel et de l'action.
"""
import torch
import torch.nn as nn
from typing import Tuple

class DynamicsNetwork(nn.Module):
    """
    Réseau de dynamique pour MuZero.
    Prédit l'état suivant et la récompense à partir de l'état actuel et de l'action.
    """
    
    def __init__(self, hidden_dim: int, action_dim: int):
        """
        Initialise le réseau de dynamique.
        
        Args:
            hidden_dim: Dimension de l'état latent
            action_dim: Dimension de l'espace d'action
        """
        super().__init__()
        
        # Réseau pour prédire l'état suivant
        self.dynamics_network = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim)
        )
        
        # Réseau pour prédire la récompense
        self.reward_network = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prédit l'état suivant et la récompense.
        
        Args:
            state: État latent (batch_size, hidden_dim)
            action: Action encodée en one-hot (batch_size, action_dim)
            
        Returns:
            Tuple (état suivant, récompense)
        """
        # Concaténer l'état et l'action
        x = torch.cat([state, action], dim=1)
        
        # Prédire l'état suivant
        next_state = self.dynamics_network(x)
        
        # Prédire la récompense
        reward = self.reward_network(next_state)
        
        return next_state, reward 