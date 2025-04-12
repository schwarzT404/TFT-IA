"""
Script principal pour l'entraînement et l'évaluation de l'agent TFT.
"""
import argparse
import os
from src.config import config
from src.environment.tft_game import TFTGame
from src.agents.muzero_agent import MuZeroAgent

def train(args):
    """
    Entraîne l'agent MuZero.
    
    Args:
        args: Arguments de ligne de commande
    """
    # Créer le répertoire de sauvegarde si nécessaire
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Initialiser l'environnement
    env = TFTGame(config)
    
    # Initialiser l'agent
    agent = MuZeroAgent(config)
    
    # Entraîner l'agent
    agent.train(env, num_episodes=args.num_episodes, save_interval=args.save_interval)
    
    # Sauvegarder le modèle
    agent.save(os.path.join(args.model_dir, 'model.pt'))
    
    print(f"Entraînement terminé. Modèle sauvegardé dans {args.model_dir}")

def evaluate(args):
    """
    Évalue l'agent MuZero.
    
    Args:
        args: Arguments de ligne de commande
    """
    # Initialiser l'environnement
    env = TFTGame(config)
    
    # Initialiser l'agent
    agent = MuZeroAgent(config)
    
    # Charger le modèle
    try:
        agent_state = env.reset()  # Pour initialiser l'agent avec les bonnes dimensions
        agent.init_networks(agent_state)
        agent.load(args.load_model)
        print("Modèle chargé avec succès.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return
    
    # Évaluer l'agent
    results = agent.evaluate(env, num_games=args.num_games, visualize=args.visualize)
    
    # Afficher les résultats
    print(f"Résultats d'évaluation:")
    print(f"  Taux de victoire: {results['win_rate']:.2f}%")
    print(f"  Placement moyen: {results['avg_placement']:.2f}")

def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(description='Entraînement et évaluation de l\'agent TFT')
    subparsers = parser.add_subparsers(dest='command', help='Commande à exécuter')
    
    # Sous-commande pour l'entraînement
    train_parser = subparsers.add_parser('train', help='Entraîner l\'agent')
    train_parser.add_argument('--num_episodes', type=int, default=1000, help='Nombre d\'épisodes d\'entraînement')
    train_parser.add_argument('--model_dir', type=str, default='data/models', help='Répertoire pour sauvegarder le modèle')
    train_parser.add_argument('--save_interval', type=int, default=100, help='Intervalle de sauvegarde des modèles')
    
    # Sous-commande pour l'évaluation
    eval_parser = subparsers.add_parser('evaluate', help='Évaluer l\'agent')
    eval_parser.add_argument('--load_model', type=str, required=True, help='Chemin du modèle à charger')
    eval_parser.add_argument('--num_games', type=int, default=10, help='Nombre de parties pour l\'évaluation')
    eval_parser.add_argument('--visualize', action='store_true', help='Activer la visualisation des résultats')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args)
    elif args.command == 'evaluate':
        evaluate(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 