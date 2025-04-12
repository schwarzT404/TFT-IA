#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour créer un modèle de test simple pour l'agent MuZero
"""

import os
import torch
from src.environment.tft_game import TFTGame
from src.agents.muzero_agent import MuZeroAgent
from src.config import config

def create_test_model(output_dir="./models"):
    """
    Crée un modèle de test simple pour l'agent MuZero.
    
    Args:
        output_dir: Répertoire où sauvegarder le modèle
    """
    # Créer le répertoire si nécessaire
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialiser l'environnement
    print("Initialisation de l'environnement...")
    env = TFTGame(config)
    
    # Initialiser l'agent
    print("Initialisation de l'agent MuZero...")
    agent = MuZeroAgent(config)
    
    # Obtenir une observation pour initialiser les réseaux
    init_obs = env.reset()
    agent.init_networks(init_obs)
    
    # Sauvegarder le modèle avec des poids initiaux
    model_path = os.path.join(output_dir, "muzero_test_model.pt")
    agent.save(model_path)
    
    print(f"Modèle de test créé et sauvegardé à: {model_path}")
    
    return model_path

if __name__ == "__main__":
    model_path = create_test_model()
    print("\nVous pouvez maintenant tester l'agent avec la commande:")
    print(f"python test_agent.py --model {model_path} --mode visualize --episodes 1") 