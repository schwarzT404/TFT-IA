#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de benchmark pour comparer différents agents dans l'environnement TFT
"""

import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

from src.environment.tft_game import TFTGame
from src.agents.muzero_agent import MuZeroAgent
from src.config import config

def load_agent(agent_path, env):
    """
    Charge un agent à partir d'un fichier modèle.
    
    Args:
        agent_path: Chemin vers le fichier modèle
        env: Environnement pour initialiser l'agent
        
    Returns:
        Agent chargé
    """
    # Initialiser l'agent
    agent = MuZeroAgent(config)
    
    # Initialiser les réseaux
    init_obs = env.reset()
    agent.init_networks(init_obs)
    
    # Charger les poids
    agent.load(agent_path)
    
    return agent

def evaluate_agent(agent, env, num_episodes=50, name="Agent"):
    """
    Évalue un agent sur plusieurs épisodes.
    
    Args:
        agent: Agent à évaluer
        env: Environnement de test
        num_episodes: Nombre d'épisodes pour l'évaluation
        name: Nom de l'agent pour l'affichage
        
    Returns:
        Dictionnaire contenant les statistiques de performance
    """
    rewards = []
    placements = []
    wins = 0
    top4s = 0
    steps_per_episode = []
    
    print(f"Évaluation de {name} sur {num_episodes} épisodes...")
    
    for episode in tqdm(range(num_episodes)):
        obs = env.reset()
        done = False
        episode_reward = 0
        step = 0
        
        while not done:
            # Sélectionner une action
            legal_actions = list(range(agent.action_dim))
            action = agent.select_action(obs, legal_actions, training=False)
            
            # Convertir l'action
            env_action = agent._convert_action(action)
            
            # Exécuter l'action
            next_obs, reward, done, info = env.step(env_action)
            
            # Mise à jour pour la prochaine étape
            obs = next_obs
            episode_reward += reward
            step += 1
            
            # Limiter le nombre d'étapes
            if step > 1000:
                done = True
        
        # Collecter les statistiques
        rewards.append(episode_reward)
        
        player = info['player']
        if not player['eliminated']:
            placement = 1
            wins += 1
            top4s += 1
        else:
            eliminated_players = sum(1 for p in env.players if p['eliminated'])
            placement = env.num_players - eliminated_players + 1
            if placement <= 4:
                top4s += 1
        
        placements.append(placement)
        steps_per_episode.append(step)
    
    # Calculer les statistiques
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    avg_placement = np.mean(placements)
    win_rate = (wins / num_episodes) * 100
    top4_rate = (top4s / num_episodes) * 100
    avg_steps = np.mean(steps_per_episode)
    
    # Afficher les résultats
    print(f"\nRésultats pour {name}:")
    print(f"Récompense moyenne: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Placement moyen: {avg_placement:.2f}")
    print(f"Taux de victoire: {win_rate:.2f}%")
    print(f"Taux de Top 4: {top4_rate:.2f}%")
    print(f"Nombre moyen d'étapes par épisode: {avg_steps:.2f}")
    
    return {
        'name': name,
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_placement': avg_placement,
        'win_rate': win_rate,
        'top4_rate': top4_rate,
        'avg_steps': avg_steps,
        'rewards': rewards,
        'placements': placements
    }

def compare_agents(agent_files, num_episodes=50, output_dir="benchmark_results"):
    """
    Compare plusieurs agents sur le même environnement.
    
    Args:
        agent_files: Liste de tuples (nom, chemin) pour chaque agent
        num_episodes: Nombre d'épisodes pour chaque agent
        output_dir: Répertoire pour sauvegarder les résultats
    """
    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialiser l'environnement
    env = TFTGame(config)
    
    results = []
    
    # Évaluer chaque agent
    for name, agent_path in agent_files:
        try:
            agent = load_agent(agent_path, env)
            stats = evaluate_agent(agent, env, num_episodes, name)
            results.append(stats)
        except Exception as e:
            print(f"Erreur lors de l'évaluation de {name}: {e}")
    
    # Comparer les résultats
    if len(results) > 0:
        # Afficher le tableau comparatif
        print("\nComparaison des agents:")
        print("-" * 100)
        print(f"{'Agent':<20} {'Récompense':<20} {'Placement':<15} {'Win Rate':<15} {'Top4 Rate':<15}")
        print("-" * 100)
        
        for stats in results:
            print(f"{stats['name']:<20} {stats['avg_reward']:.2f} ± {stats['std_reward']:.2f} "
                  f"{stats['avg_placement']:<15.2f} {stats['win_rate']:<15.2f}% {stats['top4_rate']:<15.2f}%")
        
        print("-" * 100)
        
        # Tracer les graphiques comparatifs
        plt.figure(figsize=(15, 10))
        
        # Récompenses moyennes
        plt.subplot(2, 2, 1)
        names = [stats['name'] for stats in results]
        rewards = [stats['avg_reward'] for stats in results]
        std_rewards = [stats['std_reward'] for stats in results]
        plt.bar(names, rewards, yerr=std_rewards)
        plt.title('Récompense moyenne par agent')
        plt.ylabel('Récompense')
        plt.xticks(rotation=45)
        
        # Placements moyens
        plt.subplot(2, 2, 2)
        placements = [stats['avg_placement'] for stats in results]
        plt.bar(names, placements)
        plt.title('Placement moyen par agent')
        plt.ylabel('Placement')
        plt.xticks(rotation=45)
        
        # Taux de victoire
        plt.subplot(2, 2, 3)
        win_rates = [stats['win_rate'] for stats in results]
        plt.bar(names, win_rates)
        plt.title('Taux de victoire par agent')
        plt.ylabel('Win Rate (%)')
        plt.xticks(rotation=45)
        
        # Taux de Top 4
        plt.subplot(2, 2, 4)
        top4_rates = [stats['top4_rate'] for stats in results]
        plt.bar(names, top4_rates)
        plt.title('Taux de Top 4 par agent')
        plt.ylabel('Top4 Rate (%)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(output_dir, f"agent_comparison_{timestamp}.png"))
        plt.close()
        
        # Sauvegarder les résultats au format JSON
        results_data = {
            'timestamp': timestamp,
            'num_episodes': num_episodes,
            'agents': [
                {k: v for k, v in stats.items() if k not in ['rewards', 'placements']}
                for stats in results
            ]
        }
        
        with open(os.path.join(output_dir, f"benchmark_results_{timestamp}.json"), 'w') as f:
            json.dump(results_data, f, indent=4)
        
        print(f"Résultats sauvegardés dans {output_dir}")
    else:
        print("Aucun agent n'a été évalué avec succès.")

def main():
    """Point d'entrée principal du script."""
    parser = argparse.ArgumentParser(description="Benchmark de différents agents MuZero pour TFT")
    parser.add_argument('--models', type=str, nargs='+', required=True,
                        help='Chemins vers les modèles à comparer (format: nom:chemin)')
    parser.add_argument('--episodes', type=int, default=50,
                        help='Nombre d\'épisodes pour chaque agent')
    parser.add_argument('--output', type=str, default="benchmark_results",
                        help='Répertoire pour sauvegarder les résultats')
    
    args = parser.parse_args()
    
    # Préparer la liste des agents
    agent_files = []
    for model_arg in args.models:
        parts = model_arg.split(':', 1)
        if len(parts) == 2:
            name, path = parts
        else:
            # Utiliser le nom du fichier comme nom
            path = parts[0]
            name = os.path.basename(path).split('.')[0]
        
        agent_files.append((name, path))
    
    # Comparer les agents
    compare_agents(agent_files, args.episodes, args.output)

if __name__ == '__main__':
    main() 