#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour tester l'agent MuZero dans différentes configurations
"""

import os
import argparse
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.environment.tft_game import TFTGame
from src.agents.muzero_agent import MuZeroAgent
from src.config import config

def visualize_test(agent, env, num_episodes=1, delay=0.5):
    """
    Visualise le comportement de l'agent en temps réel.
    
    Args:
        agent: Agent entraîné à tester
        env: Environnement de test
        num_episodes: Nombre d'épisodes à visualiser
        delay: Délai entre les actions (secondes)
    """
    for episode in range(num_episodes):
        print(f"\n=== Épisode de test {episode+1}/{num_episodes} ===")
        obs = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        while not done:
            # Afficher l'état actuel
            env.render()
            
            # Sélectionner une action
            legal_actions = list(range(agent.action_dim))
            action = agent.select_action(obs, legal_actions, training=False)
            
            # Convertir l'action
            env_action = agent._convert_action(action)
            
            # Exécuter l'action
            next_obs, reward, done, info = env.step(env_action)
            
            # Afficher l'action et la récompense
            print(f"Étape {step}: Action = {env_action['type']} ({env_action})")
            print(f"Récompense: {reward}")
            
            # Mise à jour pour la prochaine étape
            obs = next_obs
            total_reward += reward
            step += 1
            
            # Pause pour mieux visualiser
            time.sleep(delay)
        
        # Afficher le résultat final
        print(f"\nÉpisode terminé après {step} étapes.")
        print(f"Récompense totale: {total_reward}")
        
        player = info['player']
        if not player['eliminated']:
            print("Victoire!")
        else:
            eliminated_players = sum(1 for p in env.players if p['eliminated'])
            placement = env.num_players - eliminated_players + 1
            print(f"Placement final: {placement}")

def benchmark_performance(agent, env, num_episodes=100):
    """
    Évalue les performances de l'agent sur un grand nombre d'épisodes.
    
    Args:
        agent: Agent entraîné à tester
        env: Environnement de test
        num_episodes: Nombre d'épisodes pour l'évaluation
        
    Returns:
        Dictionnaire contenant les statistiques de performance
    """
    rewards = []
    placements = []
    wins = 0
    top4s = 0
    steps_per_episode = []
    
    print(f"Évaluation sur {num_episodes} épisodes...")
    
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
            
            # Limiter le nombre d'étapes (éviter les boucles infinies)
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
    print("\nRésultats de l'évaluation:")
    print(f"Récompense moyenne: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Placement moyen: {avg_placement:.2f}")
    print(f"Taux de victoire: {win_rate:.2f}%")
    print(f"Taux de Top 4: {top4_rate:.2f}%")
    print(f"Nombre moyen d'étapes par épisode: {avg_steps:.2f}")
    
    # Tracer les graphiques de performance
    plt.figure(figsize=(15, 10))
    
    # Récompenses
    plt.subplot(2, 2, 1)
    plt.plot(rewards)
    plt.title('Récompenses par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Récompense')
    
    # Placements
    plt.subplot(2, 2, 2)
    plt.plot(placements)
    plt.title('Placements par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Placement')
    plt.gca().invert_yaxis()  # Inverser l'axe Y pour que 1 soit en haut
    
    # Distribution des placements
    plt.subplot(2, 2, 3)
    plt.hist(placements, bins=range(1, env.num_players + 2), align='left')
    plt.title('Distribution des placements')
    plt.xlabel('Placement')
    plt.ylabel('Fréquence')
    plt.xticks(range(1, env.num_players + 1))
    
    # Nombre d'étapes par épisode
    plt.subplot(2, 2, 4)
    plt.plot(steps_per_episode)
    plt.title('Étapes par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Nombre d\'étapes')
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png')
    plt.close()
    
    print(f"Graphiques de performance sauvegardés dans 'evaluation_results.png'")
    
    return {
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_placement': avg_placement,
        'win_rate': win_rate,
        'top4_rate': top4_rate,
        'avg_steps': avg_steps
    }

def compare_strategies(agent, env, num_episodes=50):
    """
    Compare différentes stratégies ou configurations de l'agent.
    
    Args:
        agent: Agent entraîné à tester
        env: Environnement de test
        num_episodes: Nombre d'épisodes pour chaque configuration
    """
    # Définir différentes configurations à tester
    configurations = [
        {"name": "Greedy", "temperature": 0.0},
        {"name": "Balanced", "temperature": 1.0},
        {"name": "Exploratory", "temperature": 2.0}
    ]
    
    results = {}
    
    for config in configurations:
        print(f"\nTest de la configuration: {config['name']}")
        
        # Appliquer la configuration
        original_temp = agent.temperature if hasattr(agent, 'temperature') else 1.0
        if hasattr(agent, 'temperature'):
            agent.temperature = config["temperature"]
        
        # Évaluer la configuration
        stats = benchmark_performance(agent, env, num_episodes)
        results[config["name"]] = stats
        
        # Restaurer la configuration originale
        if hasattr(agent, 'temperature'):
            agent.temperature = original_temp
    
    # Afficher la comparaison
    print("\nComparaison des configurations:")
    print("-" * 80)
    print(f"{'Configuration':<15} {'Récompense':<15} {'Placement':<15} {'Win Rate':<15} {'Top4 Rate':<15}")
    print("-" * 80)
    
    for name, stats in results.items():
        print(f"{name:<15} {stats['avg_reward']:<15.2f} {stats['avg_placement']:<15.2f} "
              f"{stats['win_rate']:<15.2f}% {stats['top4_rate']:<15.2f}%")
    
    print("-" * 80)
    
    # Tracer des graphiques comparatifs
    plt.figure(figsize=(15, 10))
    
    # Récompenses moyennes
    plt.subplot(2, 2, 1)
    names = list(results.keys())
    rewards = [results[name]['avg_reward'] for name in names]
    plt.bar(names, rewards)
    plt.title('Récompense moyenne par configuration')
    plt.ylabel('Récompense')
    
    # Placements moyens
    plt.subplot(2, 2, 2)
    placements = [results[name]['avg_placement'] for name in names]
    plt.bar(names, placements)
    plt.title('Placement moyen par configuration')
    plt.ylabel('Placement')
    
    # Taux de victoire
    plt.subplot(2, 2, 3)
    win_rates = [results[name]['win_rate'] for name in names]
    plt.bar(names, win_rates)
    plt.title('Taux de victoire par configuration')
    plt.ylabel('Win Rate (%)')
    
    # Taux de Top 4
    plt.subplot(2, 2, 4)
    top4_rates = [results[name]['top4_rate'] for name in names]
    plt.bar(names, top4_rates)
    plt.title('Taux de Top 4 par configuration')
    plt.ylabel('Top4 Rate (%)')
    
    plt.tight_layout()
    plt.savefig('strategy_comparison.png')
    plt.close()
    
    print(f"Graphiques de comparaison sauvegardés dans 'strategy_comparison.png'")

def action_distribution_analysis(agent, env, num_episodes=20):
    """
    Analyse la distribution des actions prises par l'agent.
    
    Args:
        agent: Agent entraîné à tester
        env: Environnement de test
        num_episodes: Nombre d'épisodes pour l'analyse
    """
    action_counts = {}
    action_rewards = {}
    
    print(f"Analyse de la distribution des actions sur {num_episodes} épisodes...")
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        
        while not done:
            # Sélectionner une action
            legal_actions = list(range(agent.action_dim))
            action = agent.select_action(obs, legal_actions, training=False)
            
            # Convertir l'action
            env_action = agent._convert_action(action)
            action_type = env_action['type']
            
            # Compter les actions par type
            if action_type not in action_counts:
                action_counts[action_type] = 0
                action_rewards[action_type] = []
            
            action_counts[action_type] += 1
            
            # Exécuter l'action
            next_obs, reward, done, info = env.step(env_action)
            
            # Enregistrer la récompense associée à cette action
            action_rewards[action_type].append(reward)
            
            # Mise à jour pour la prochaine étape
            obs = next_obs
    
    # Calculer les récompenses moyennes par type d'action
    avg_rewards = {action: np.mean(rewards) for action, rewards in action_rewards.items()}
    
    # Afficher les résultats
    total_actions = sum(action_counts.values())
    
    print("\nDistribution des actions:")
    print("-" * 60)
    print(f"{'Type d\'action':<20} {'Nombre':<10} {'Pourcentage':<15} {'Récompense moyenne':<20}")
    print("-" * 60)
    
    for action_type, count in sorted(action_counts.items()):
        percentage = (count / total_actions) * 100
        avg_reward = avg_rewards[action_type]
        print(f"{action_type:<20} {count:<10} {percentage:<15.2f}% {avg_reward:<20.4f}")
    
    print("-" * 60)
    
    # Tracer les graphiques
    plt.figure(figsize=(12, 10))
    
    # Distribution des actions
    plt.subplot(2, 1, 1)
    actions = list(action_counts.keys())
    counts = list(action_counts.values())
    plt.bar(actions, counts)
    plt.title('Distribution des actions')
    plt.ylabel('Nombre d\'actions')
    plt.xticks(rotation=45)
    
    # Récompense moyenne par action
    plt.subplot(2, 1, 2)
    rewards = [avg_rewards[action] for action in actions]
    plt.bar(actions, rewards)
    plt.title('Récompense moyenne par type d\'action')
    plt.ylabel('Récompense moyenne')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('action_distribution.png')
    plt.close()
    
    print(f"Graphiques de distribution d'actions sauvegardés dans 'action_distribution.png'")

def main():
    """Point d'entrée principal du script."""
    parser = argparse.ArgumentParser(description="Test de l'agent MuZero pour TFT")
    parser.add_argument('--model', type=str, required=True, help='Chemin vers le modèle à tester')
    parser.add_argument('--mode', type=str, default='visualize', 
                        choices=['visualize', 'benchmark', 'compare', 'action_analysis'],
                        help='Mode de test à exécuter')
    parser.add_argument('--episodes', type=int, default=10, help='Nombre d\'épisodes pour le test')
    parser.add_argument('--delay', type=float, default=0.5, help='Délai entre les actions (pour le mode visualize)')
    
    args = parser.parse_args()
    
    # Initialiser l'environnement
    env = TFTGame(config)
    
    # Initialiser l'agent
    agent = MuZeroAgent(config)
    
    # Obtenir une observation pour initialiser les réseaux
    init_obs = env.reset()
    agent.init_networks(init_obs)
    
    # Charger le modèle
    try:
        agent.load(args.model)
        print(f"Modèle chargé depuis {args.model}")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return
    
    # Exécuter le mode de test spécifié
    if args.mode == 'visualize':
        visualize_test(agent, env, num_episodes=args.episodes, delay=args.delay)
    elif args.mode == 'benchmark':
        benchmark_performance(agent, env, num_episodes=args.episodes)
    elif args.mode == 'compare':
        compare_strategies(agent, env, num_episodes=args.episodes)
    elif args.mode == 'action_analysis':
        action_distribution_analysis(agent, env, num_episodes=args.episodes)

if __name__ == '__main__':
    main() 