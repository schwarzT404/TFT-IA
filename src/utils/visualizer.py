"""
Module de visualisation pour afficher les résultats et statistiques de l'agent TFT.
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
import os

class Visualizer:
    """Classe pour visualiser les résultats et statistiques de l'agent TFT."""
    
    def __init__(self, save_dir: str = 'data/visualizations'):
        """
        Initialise le visualiseur.
        
        Args:
            save_dir: Répertoire où sauvegarder les visualisations
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_rewards(self, rewards: List[float], title: str = 'Récompenses par épisode'):
        """
        Trace la courbe des récompenses par épisode.
        
        Args:
            rewards: Liste des récompenses par épisode
            title: Titre du graphique
        """
        plt.figure(figsize=(10, 6))
        plt.plot(rewards)
        plt.title(title)
        plt.xlabel('Épisode')
        plt.ylabel('Récompense')
        plt.grid(True)
        
        # Sauvegarder l'image
        plt.savefig(os.path.join(self.save_dir, 'rewards.png'))
        plt.close()
    
    def plot_placements(self, placements: List[int], title: str = 'Placements par partie'):
        """
        Trace la courbe des placements par partie.
        
        Args:
            placements: Liste des placements par partie
            title: Titre du graphique
        """
        plt.figure(figsize=(10, 6))
        plt.plot(placements)
        plt.title(title)
        plt.xlabel('Partie')
        plt.ylabel('Placement')
        plt.grid(True)
        
        # Inverser l'axe y (le placement 1 est le meilleur)
        plt.gca().invert_yaxis()
        
        # Sauvegarder l'image
        plt.savefig(os.path.join(self.save_dir, 'placements.png'))
        plt.close()
    
    def plot_win_rate(self, win_rates: List[float], window_size: int = 100, title: str = 'Taux de victoire glissant'):
        """
        Trace la courbe du taux de victoire glissant.
        
        Args:
            win_rates: Liste des indicateurs de victoire (1 pour victoire, 0 pour défaite)
            window_size: Taille de la fenêtre glissante
            title: Titre du graphique
        """
        plt.figure(figsize=(10, 6))
        
        # Calculer le taux de victoire glissant
        sliding_wins = np.convolve(win_rates, np.ones(window_size) / window_size, mode='valid')
        
        plt.plot(sliding_wins)
        plt.title(title)
        plt.xlabel('Partie')
        plt.ylabel('Taux de victoire')
        plt.grid(True)
        
        # Sauvegarder l'image
        plt.savefig(os.path.join(self.save_dir, 'win_rate.png'))
        plt.close()
    
    def plot_action_distribution(self, actions: List[int], num_actions: int = 5, title: str = 'Distribution des actions'):
        """
        Trace la distribution des actions choisies par l'agent.
        
        Args:
            actions: Liste des actions choisies
            num_actions: Nombre d'actions possibles
            title: Titre du graphique
        """
        plt.figure(figsize=(10, 6))
        
        # Compter les occurrences de chaque action
        action_counts = np.bincount(actions, minlength=num_actions)
        
        # Noms des actions
        action_names = ['Achat', 'Vente', 'Position', 'Niveau', 'Refresh']
        
        plt.bar(action_names, action_counts)
        plt.title(title)
        plt.xlabel('Action')
        plt.ylabel('Nombre d\'occurrences')
        
        # Sauvegarder l'image
        plt.savefig(os.path.join(self.save_dir, 'action_distribution.png'))
        plt.close()
    
    def plot_training_metrics(self, metrics: Dict[str, List[float]]):
        """
        Trace plusieurs métriques d'entraînement.
        
        Args:
            metrics: Dictionnaire des métriques {nom: valeurs}
        """
        plt.figure(figsize=(15, 10))
        
        # Nombre de métriques
        num_metrics = len(metrics)
        
        # Créer un subplot pour chaque métrique
        for i, (metric_name, values) in enumerate(metrics.items()):
            plt.subplot(num_metrics, 1, i + 1)
            plt.plot(values)
            plt.title(metric_name)
            plt.grid(True)
        
        plt.tight_layout()
        
        # Sauvegarder l'image
        plt.savefig(os.path.join(self.save_dir, 'training_metrics.png'))
        plt.close()
    
    def visualize_board(self, board: List[Dict[str, Any]], title: str = 'Plateau de jeu'):
        """
        Visualise le plateau de jeu.
        
        Args:
            board: Liste des unités sur le plateau
            title: Titre de la visualisation
        """
        plt.figure(figsize=(12, 8))
        
        # Coordonnées des hexagones
        hex_coords = [
            # Rangée 1 (front)
            (1, 0), (3, 0), (5, 0), (7, 0), (9, 0), (11, 0), (13, 0),
            # Rangée 2
            (0, 2), (2, 2), (4, 2), (6, 2), (8, 2), (10, 2), (12, 2),
            # Rangée 3
            (1, 4), (3, 4), (5, 4), (7, 4), (9, 4), (11, 4), (13, 4),
            # Rangée 4 (arrière)
            (0, 6), (2, 6), (4, 6), (6, 6), (8, 6), (10, 6), (12, 6)
        ]
        
        # Dessiner les hexagones
        for i, (x, y) in enumerate(hex_coords):
            # Hexagone vide par défaut
            color = 'lightgray'
            label = ''
            
            # Si une unité est présente à cette position
            if i < len(board) and board[i] is not None:
                unit = board[i]
                # Couleur basée sur le coût
                if unit['cost'] == 1:
                    color = 'gray'
                elif unit['cost'] == 2:
                    color = 'green'
                elif unit['cost'] == 3:
                    color = 'blue'
                elif unit['cost'] == 4:
                    color = 'purple'
                elif unit['cost'] == 5:
                    color = 'orange'
                
                # Nom de l'unité
                label = unit['name']
            
            # Dessiner l'hexagone
            self._draw_hexagon(x, y, color, label)
        
        plt.title(title)
        plt.axis('equal')
        plt.axis('off')
        
        # Sauvegarder l'image
        plt.savefig(os.path.join(self.save_dir, 'board.png'))
        plt.close()
    
    def _draw_hexagon(self, x, y, color: str, label: str):
        """
        Dessine un hexagone à la position spécifiée.
        
        Args:
            x: Coordonnée x du centre
            y: Coordonnée y du centre
            color: Couleur de l'hexagone
            label: Texte à afficher dans l'hexagone
        """
        # Paramètres de l'hexagone
        r = 1  # Rayon
        angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 sommets
        
        # Coordonnées des sommets
        hex_x = x + r * np.cos(angles)
        hex_y = y + r * np.sin(angles)
        
        # Dessiner l'hexagone
        plt.fill(hex_x, hex_y, color=color, alpha=0.7)
        plt.plot(hex_x, hex_y, 'k-')
        
        # Ajouter le texte
        plt.text(x, y, label, ha='center', va='center', fontsize=8)

def visualize_episode(env, agent, episode_num: int, save_dir: str = 'data/visualizations'):
    """
    Visualise un épisode complet.
    
    Args:
        env: Environnement de jeu
        agent: Agent MuZero
        episode_num: Numéro de l'épisode
        save_dir: Répertoire où sauvegarder les visualisations
    """
    visualizer = Visualizer(save_dir=os.path.join(save_dir, f'episode_{episode_num}'))
    
    observation = env.reset()
    done = False
    step = 0
    actions = []
    
    while not done:
        # Choisir une action
        legal_actions = list(range(agent.action_dim))
        action = agent.select_action(observation, legal_actions, training=False)
        actions.append(action)
        
        # Convertir l'action
        env_action = agent._convert_action(action)
        
        # Effectuer l'action
        observation, reward, done, info = env.step(env_action)
        
        # Visualiser le plateau tous les 5 pas de temps
        if step % 5 == 0:
            player = info['player']
            visualizer.visualize_board(player['board'], title=f'Plateau - Étape {step}')
        
        step += 1
    
    # Visualiser la distribution des actions
    visualizer.plot_action_distribution(actions)
    
    print(f"Visualisation de l'épisode {episode_num} terminée. Résultats sauvegardés dans {visualizer.save_dir}") 