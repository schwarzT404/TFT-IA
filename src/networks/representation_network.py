"""
Module définissant le réseau de représentation pour MuZero.
Ce réseau encode les observations en une représentation d'état latente.
"""
import torch
import torch.nn as nn

class RepresentationNetwork(nn.Module):
    """
    Réseau de représentation pour MuZero.
    Transforme l'observation en un état latent que les autres réseaux peuvent traiter.
    """
    
    def __init__(self, observation_dim: int, hidden_dim: int):
        """
        Initialise le réseau de représentation.
        
        Args:
            observation_dim: Dimension de l'observation
            hidden_dim: Dimension de l'état latent
        """
        super().__init__()
        
        # Architecture du réseau
        self.network = nn.Sequential(
            nn.Linear(observation_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim)
        )
    
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Transforme l'observation en état latent.
        
        Args:
            observation: Observation (batch_size, observation_dim)
            
        Returns:
            État latent (batch_size, hidden_dim)
        """
        return self.network(observation) 