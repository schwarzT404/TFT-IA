"""
Script de démonstration pour tester l'environnement TFT et l'agent MuZero.
"""
import numpy as np
from src.config import config
from src.environment.tft_game import TFTGame
from src.agents.muzero_agent import MuZeroAgent
from src.utils.visualizer import Visualizer

def run_demo():
    """Exécute une démonstration simple de l'environnement et de l'agent."""
    print("Initialisation de l'environnement TFT...")
    env = TFTGame(config)
    
    print("Initialisation de l'agent MuZero...")
    agent = MuZeroAgent(config)
    
    print("Exécution d'un épisode de démonstration...")
    
    # Réinitialiser l'environnement
    observation = env.reset()
    
    # Variables pour suivre l'épisode
    done = False
    step = 0
    total_reward = 0
    actions = []
    
    # Créer un visualiseur
    visualizer = Visualizer(save_dir='data/demo')
    
    # Afficher l'état initial
    env.render()
    
    # Boucle d'épisode
    while not done and step < 100:  # Limiter à 100 étapes pour la démo
        # Obtenir les actions légales (simplifiées pour la démo)
        legal_actions = list(range(agent.action_dim))
        
        # Sélectionner une action
        action = agent.select_action(observation, legal_actions)
        actions.append(action)
        
        # Convertir l'action pour l'environnement
        env_action = agent._convert_action(action)
        
        # Afficher l'action
        action_names = ['Achat', 'Vente', 'Position', 'Niveau', 'Refresh']
        print(f"Étape {step}: Action = {action_names[action]} ({env_action})")
        
        # Exécuter l'action dans l'environnement
        next_observation, reward, done, info = env.step(env_action)
        
        # Mettre à jour les variables
        observation = next_observation
        total_reward += reward
        step += 1
        
        # Afficher l'état actuel
        if step % 5 == 0:
            env.render()
    
    # Afficher les résultats
    print(f"\nÉpisode terminé après {step} étapes.")
    print(f"Récompense totale: {total_reward:.2f}")
    
    # Visualiser la distribution des actions
    visualizer.plot_action_distribution(actions)
    print(f"Distribution des actions sauvegardée dans {visualizer.save_dir}")
    
    return env, agent

if __name__ == "__main__":
    print("Démarrage de la démonstration TFT-IA...")
    env, agent = run_demo()
    print("Démonstration terminée.") 