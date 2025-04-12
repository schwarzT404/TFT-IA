"""
Module définissant le réseau de prédiction pour MuZero.
Ce réseau prédit la politique et la valeur à partir d'un état latent.
"""
import torch
import torch.nn as nn
from typing import Tuple

class PredictionNetwork(nn.Module):
    """
    Réseau de prédiction pour MuZero.
    Prédit la politique (probabilités d'action) et la valeur d'un état.
    """
    
    def __init__(self, hidden_dim: int, action_dim: int):
        """
        Initialise le réseau de prédiction.
        
        Args:
            hidden_dim: Dimension de l'état latent
            action_dim: Dimension de l'espace d'action
        """
        super().__init__()
        
        # Couches partagées
        self.shared_layers = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU()
        )
        
        # Réseau de politique
        self.policy_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # Réseau de valeur
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()  # Normaliser la valeur entre -1 et 1
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prédit la politique et la valeur.
        
        Args:
            state: État latent (batch_size, hidden_dim)
            
        Returns:
            Tuple (logits de politique, valeur)
        """
        # Passer par les couches partagées
        x = self.shared_layers(state)
        
        # Prédire la politique (logits)
        policy_logits = self.policy_head(x)
        
        # Prédire la valeur
        value = self.value_head(x)
        
        return policy_logits, value 