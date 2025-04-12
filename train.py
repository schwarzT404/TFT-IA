#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script d'entraînement pour l'agent MuZero dans l'environnement TFT.
Ce script permet de :
1. Entraîner l'agent sur plusieurs épisodes
2. Suivre les performances via TensorBoard
3. Sauvegarder les modèles régulièrement
4. Évaluer l'agent après l'entraînement
"""

import os
import argparse
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import deque

from src.env.tft_env import TFTGame
from src.agents.muzero_agent import MuZeroAgent
from src.config import muzero_config

def train(args):
    """
    Entraîne l'agent MuZero sur plusieurs épisodes.
    
    Args:
        args: Arguments de ligne de commande
    """
    # Création des répertoires nécessaires
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Initialisation de TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Initialisation de l'environnement et de l'agent
    env = TFTGame()
    agent = MuZeroAgent(muzero_config)
    
    # Chargement d'un modèle pré-entraîné si spécifié
    if args.load_model and os.path.exists(args.load_model):
        agent.load(args.load_model)
        print(f"Modèle chargé depuis: {args.load_model}")
    
    # Variables pour le suivi des performances
    episode_rewards = []
    episode_placements = []
    win_rate_window = deque(maxlen=100)
    top4_rate_window = deque(maxlen=100)
    
    # Temps de début d'entraînement
    start_time = time.time()
    
    print(f"Démarrage de l'entraînement pour {args.num_episodes} épisodes")
    
    for episode in range(args.num_episodes):
        # Réinitialisation de l'environnement
        obs = env.reset()
        done = False
        episode_reward = 0
        step = 0
        
        # Si c'est le premier épisode, initialiser les réseaux
        if episode == 0 and not args.load_model:
            agent.init_networks(obs)
        
        # Boucle d'épisode
        while not done:
            # Sélection d'action
            action = agent.select_action(obs)
            
            # Exécution de l'action
            next_obs, reward, done, info = env.step(action)
            
            # Stockage de l'expérience
            agent.store_experience(obs, action, reward, next_obs, done)
            
            # Mise à jour de l'agent
            if len(agent.replay_buffer) >= agent.min_replay_buffer_size:
                agent._train_networks()
                
                # Journalisation des pertes dans TensorBoard
                if hasattr(agent, 'last_losses') and agent.last_losses:
                    for loss_name, loss_value in agent.last_losses.items():
                        writer.add_scalar(f'Loss/{loss_name}', loss_value, agent.training_step)
            
            # Mise à jour pour la prochaine étape
            obs = next_obs
            episode_reward += reward
            step += 1
            
            # Journalisation de l'étape
            if agent.training_step > 0 and agent.training_step % 100 == 0:
                writer.add_scalar('Training/Steps_per_second', 
                                 100 / (time.time() - start_time), 
                                 agent.training_step)
                start_time = time.time()
        
        # Fin de l'épisode
        episode_rewards.append(episode_reward)
        placement = info.get('placement', 8)  # Position par défaut 8 (dernier)
        episode_placements.append(placement)
        
        # Mise à jour des fenêtres glissantes
        win_rate_window.append(1 if placement == 1 else 0)
        top4_rate_window.append(1 if placement <= 4 else 0)
        
        # Journalisation dans TensorBoard
        writer.add_scalar('Episode/Reward', episode_reward, episode)
        writer.add_scalar('Episode/Placement', placement, episode)
        writer.add_scalar('Episode/Win_Rate', np.mean(win_rate_window), episode)
        writer.add_scalar('Episode/Top4_Rate', np.mean(top4_rate_window), episode)
        
        # Affichage de progression
        if (episode + 1) % 10 == 0:
            print(f"Épisode {episode+1}/{args.num_episodes}, "
                  f"Récompense: {episode_reward:.2f}, "
                  f"Placement: {placement}, "
                  f"Win Rate: {np.mean(win_rate_window):.2f}, "
                  f"Top4 Rate: {np.mean(top4_rate_window):.2f}")
        
        # Sauvegarde périodique du modèle
        if (episode + 1) % args.save_interval == 0:
            save_path = os.path.join(args.model_dir, f"muzero_episode_{episode+1}.pt")
            agent.save(save_path)
            print(f"Modèle sauvegardé: {save_path}")
        
        # Visualisation périodique d'un épisode
        if args.visualize_interval > 0 and (episode + 1) % args.visualize_interval == 0:
            evaluate(agent, 1, writer, episode, visualize=True)
    
    # Sauvegarde finale du modèle
    final_model_path = os.path.join(args.model_dir, "muzero_final.pt")
    agent.save(final_model_path)
    print(f"Entraînement terminé. Modèle final sauvegardé: {final_model_path}")
    
    # Fermeture de TensorBoard
    writer.close()
    
    # Retourner l'agent entraîné
    return agent

def evaluate(agent, num_games=10, writer=None, episode=None, visualize=False):
    """
    Évalue l'agent sur plusieurs parties.
    
    Args:
        agent: Agent MuZero à évaluer
        num_games: Nombre de parties pour l'évaluation
        writer: Writer TensorBoard (optionnel)
        episode: Numéro de l'épisode actuel (optionnel)
        visualize: Afficher l'environnement pendant l'évaluation
        
    Returns:
        Statistiques d'évaluation
    """
    env = TFTGame()
    rewards = []
    placements = []
    
    for game in range(num_games):
        obs = env.reset()
        done = False
        game_reward = 0
        
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            game_reward += reward
            
            if visualize:
                env.render()
                time.sleep(0.1)
        
        rewards.append(game_reward)
        placements.append(info.get('placement', 8))
    
    # Calcul des statistiques
    avg_reward = np.mean(rewards)
    avg_placement = np.mean(placements)
    win_rate = np.mean([1 if p == 1 else 0 for p in placements])
    top4_rate = np.mean([1 if p <= 4 else 0 for p in placements])
    
    # Journalisation si TensorBoard est disponible
    if writer and episode is not None:
        writer.add_scalar('Eval/Avg_Reward', avg_reward, episode)
        writer.add_scalar('Eval/Avg_Placement', avg_placement, episode)
        writer.add_scalar('Eval/Win_Rate', win_rate, episode)
        writer.add_scalar('Eval/Top4_Rate', top4_rate, episode)
    
    print(f"Évaluation sur {num_games} parties:")
    print(f"  Récompense moyenne: {avg_reward:.2f}")
    print(f"  Placement moyen: {avg_placement:.2f}")
    print(f"  Win Rate: {win_rate:.2f}")
    print(f"  Top4 Rate: {top4_rate:.2f}")
    
    return {
        'avg_reward': avg_reward,
        'avg_placement': avg_placement,
        'win_rate': win_rate,
        'top4_rate': top4_rate
    }

def imitation_learning(args):
    """
    Apprendre à partir de données de joueurs humains.
    
    Args:
        args: Arguments de ligne de commande
    """
    # Cette fonction serait implémentée ultérieurement
    print("L'apprentissage par imitation n'est pas encore implémenté.")
    pass

def main():
    """Point d'entrée principal du script."""
    parser = argparse.ArgumentParser(description="Entraînement et évaluation de l'agent MuZero pour TFT")
    
    # Arguments communs
    parser.add_argument('--model_dir', type=str, default='./models',
                        help='Répertoire pour sauvegarder les modèles')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Répertoire pour les logs TensorBoard')
    
    # Sous-parsers pour différentes commandes
    subparsers = parser.add_subparsers(dest='command', help='Commande à exécuter')
    
    # Parser pour l'entraînement
    train_parser = subparsers.add_parser('train', help='Entraîner l\'agent MuZero')
    train_parser.add_argument('--num_episodes', type=int, default=1000,
                            help='Nombre d\'épisodes d\'entraînement')
    train_parser.add_argument('--load_model', type=str, default=None,
                            help='Chemin vers un modèle à charger')
    train_parser.add_argument('--save_interval', type=int, default=100,
                            help='Intervalle pour sauvegarder le modèle')
    train_parser.add_argument('--visualize_interval', type=int, default=0,
                            help='Intervalle pour visualiser un épisode (0 pour désactiver)')
    
    # Parser pour l'évaluation
    eval_parser = subparsers.add_parser('eval', help='Évaluer l\'agent MuZero')
    eval_parser.add_argument('--load_model', type=str, required=True,
                           help='Chemin vers le modèle à évaluer')
    eval_parser.add_argument('--num_games', type=int, default=10,
                           help='Nombre de parties pour l\'évaluation')
    eval_parser.add_argument('--visualize', action='store_true',
                           help='Visualiser les parties pendant l\'évaluation')
    
    # Parser pour l'apprentissage par imitation
    il_parser = subparsers.add_parser('imitation', help='Apprentissage par imitation à partir de données humaines')
    il_parser.add_argument('--data_dir', type=str, required=True,
                         help='Répertoire contenant les données de parties humaines')
    il_parser.add_argument('--output_model', type=str, required=True,
                         help='Chemin pour sauvegarder le modèle entraîné')
    
    args = parser.parse_args()
    
    # Exécuter la commande appropriée
    if args.command == 'train':
        train(args)
    elif args.command == 'eval':
        agent = MuZeroAgent(muzero_config)
        agent.load(args.load_model)
        evaluate(agent, args.num_games, visualize=args.visualize)
    elif args.command == 'imitation':
        imitation_learning(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 