"""
Module implémentant l'agent MuZero pour TFT
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import logging
import csv
import datetime
from typing import Dict, List, Any, Tuple
from collections import deque

# Importation des réseaux de neurones pour MuZero
from src.networks.representation_network import RepresentationNetwork
from src.networks.dynamics_network import DynamicsNetwork
from src.networks.prediction_network import PredictionNetwork
from src.agents.mcts import MCTS

# Logger
logger = logging.getLogger("tft_muzero")

class ReplayBuffer:
    """Tampon de rejeu pour stocker et échantillonner des expériences."""
    
    def __init__(self, capacity: int):
        """
        Initialise le tampon de rejeu.
        
        Args:
            capacity: Capacité maximale du tampon
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Tuple):
        """
        Ajoute une expérience au tampon.
        
        Args:
            experience: Tuple (état, action, récompense, état suivant, terminé, info)
        """
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """
        Échantillonne des expériences aléatoirement dans le tampon.
        
        Args:
            batch_size: Nombre d'expériences à échantillonner
            
        Returns:
            Liste d'expériences
        """
        return random.sample(self.buffer, min(len(self.buffer), batch_size))
    
    def __len__(self) -> int:
        """Renvoie la taille actuelle du tampon."""
        return len(self.buffer)

class MuZeroAgent:
    """Agent utilisant l'algorithme MuZero pour jouer à TFT."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise l'agent MuZero.
        
        Args:
            config: Dictionnaire de configuration
        """
        self.config = config
        
        # Dimensions de l'espace d'observation et d'action
        self.observation_dim = None  # Sera déterminé lors de la première observation
        self.action_dim = config["environment"]["action_space"]
        
        # Hyperparamètres
        self.batch_size = config["training"]["batch_size"]
        self.learning_rate = config["training"]["learning_rate"]
        self.discount_factor = config["training"]["discount_factor"]
        self.min_replay_buffer_size = config["training"]["min_replay_buffer_size"]
        self.replay_buffer_size = config["training"]["replay_buffer_size"]
        self.weight_decay = config["training"]["weight_decay"]
        
        # Paramètres du réseau
        self.latent_dim = config["network"]["latent_dim"]
        self.hidden_dim = config["network"]["hidden_dim"]
        self.num_layers = config["network"]["num_layers"]
        self.dropout = config["network"]["dropout"]
        
        # Tampon de rejeu
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        
        # Réseaux de neurones
        self.representation_network = None
        self.dynamics_network = None
        self.prediction_network = None
        
        # Optimiseur
        self.optimizer = None
        
        # Compteur d'étapes d'entraînement
        self.training_step = 0
        
        # Historique des pertes pour le suivi TensorBoard
        self.last_losses = {
            'value': 0.0,
            'policy': 0.0,
            'reward': 0.0,
            'total': 0.0
        }
        
        # Suivi des statistiques MCTS dans le temps
        self.mcts_stats_history = {
            'visit_entropy': deque(maxlen=100),
            'confidence': deque(maxlen=100),
            'value_std': deque(maxlen=100)
        }
        
        # Dispositif d'exécution (CPU/GPU)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # MCTS
        self.mcts = None
        
        logger.info(f"Agent MuZero initialisé avec dispositif: {self.device}")
    
    def init_networks(self, observation: np.ndarray):
        """
        Initialise les réseaux de neurones.
        
        Args:
            observation: Exemple d'observation pour déterminer les dimensions
        """
        if isinstance(observation, np.ndarray):
            observation_tensor = torch.FloatTensor(observation)
        else:
            observation_tensor = observation
            
        self.observation_dim = observation_tensor.shape[-1]
        
        # Initialiser les réseaux avec les paramètres de configuration
        self.representation_network = RepresentationNetwork(
            observation_dim=self.observation_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            num_residual_blocks=self.num_layers,
            dropout=self.dropout,
            use_spatial=True
        ).to(self.device)
        
        self.dynamics_network = DynamicsNetwork(
            hidden_dim=self.hidden_dim,
            action_space=self.action_dim,
            latent_dim=self.latent_dim,
            num_residual_blocks=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        self.prediction_network = PredictionNetwork(
            hidden_dim=self.hidden_dim,
            action_space=self.action_dim,
            latent_dim=self.latent_dim,
            num_residual_blocks=self.num_layers,
            dropout=self.dropout,
            value_support_size=self.config["network"]["value_support"]
        ).to(self.device)
        
        # Initialiser l'optimiseur
        parameters = list(self.representation_network.parameters()) + \
                     list(self.dynamics_network.parameters()) + \
                     list(self.prediction_network.parameters())
        
        self.optimizer = optim.Adam(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Initialiser MCTS avec les réseaux
        self.mcts = MCTS(
            config=self.config,
            networks={
                "representation": self.representation_network,
                "dynamics": self.dynamics_network,
                "prediction": self.prediction_network
            },
            device=self.device
        )
        
        logger.info(f"Réseaux initialisés - Dimension d'observation: {self.observation_dim}, "
                   f"Dimension latente: {self.latent_dim}, "
                   f"Espace d'action: {self.action_dim}")
    
    def collect_mcts_statistics(self, root=None):
        """
        Collecte les statistiques utiles sur la dernière recherche MCTS.
        
        Args:
            root: La racine de la recherche MCTS (utilise la racine actuelle si None)
            
        Returns:
            Un dictionnaire de statistiques
        """
        import math
        
        if root is None:
            if not hasattr(self.mcts, 'root') or self.mcts.root is None:
                return {}
            root = self.mcts.root
            
        stats = {}
        
        # Si la racine n'a pas d'enfants, retourner un dictionnaire vide
        if not hasattr(root, 'children') or not root.children:
            return {}
            
        # Calculer la profondeur moyenne des simulations
        depths = [node.depth for node in root.children.values() if hasattr(node, 'depth')]
        if depths:
            stats['avg_depth'] = sum(depths) / len(depths)
            stats['max_depth'] = max(depths)
        
        # Calculer la valeur moyenne et l'incertitude
        values = [child.value() for child in root.children.values() if hasattr(child, 'value')]
        if values:
            stats['value_mean'] = sum(values) / len(values)
            stats['value_std'] = (sum((v - stats['value_mean'])**2 for v in values) / len(values))**0.5
            
            # Ajout de statistiques sur la distribution des valeurs
            stats['value_max'] = max(values)
            stats['value_min'] = min(values)
            stats['value_range'] = stats['value_max'] - stats['value_min']
            
            # Calcul du coefficient de variation (écart-type relatif) pour mesurer la dispersion
            if stats['value_mean'] != 0:
                stats['value_cv'] = stats['value_std'] / abs(stats['value_mean'])
            
            # Nombre de nœuds avec des valeurs positives vs négatives
            stats['value_positive_ratio'] = sum(1 for v in values if v > 0) / len(values)
        
        # Statistiques sur les visites
        visits = [child.visit_count for child in root.children.values() if hasattr(child, 'visit_count')]
        if visits:
            total_visits = sum(visits)
            stats['visit_mean'] = total_visits / len(visits)
            stats['visit_std'] = (sum((v - stats['visit_mean'])**2 for v in visits) / len(visits))**0.5
            stats['visit_max'] = max(visits)
            
            # Entropie des visites (mesure de l'exploration)
            if total_visits > 0:
                stats['visit_entropy'] = -sum((v/total_visits)*math.log2(v/total_visits) 
                                             for v in visits if v > 0)
        
        # Nombre d'actions explorées
        stats['actions_explored'] = len(root.children)
        
        # Ajout de métriques supplémentaires
        if visits and total_visits > 0:
            # Obtenir l'action avec le plus de visites (meilleure action)
            best_action = max(root.children.items(), key=lambda x: x[1].visit_count if hasattr(x[1], 'visit_count') else 0)
            if hasattr(best_action[1], 'visit_count'):
                best_action_visits = best_action[1].visit_count
                # Confiance: ratio entre les visites de la meilleure action et le total
                stats['confidence'] = best_action_visits / total_visits
                
                # Dominance: ratio entre les visites de la meilleure action et la moyenne
                stats['dominance'] = best_action_visits / stats['visit_mean'] if stats['visit_mean'] > 0 else 1.0
                
                # Concentration: Indice de Gini sur la distribution des visites (mesure d'inégalité)
                # Un indice de Gini de 0 signifie une exploration uniforme, 1 signifie tout sur une action
                sorted_visits = sorted(visits)
                cum_visits = np.cumsum(sorted_visits)
                gini = 1 - 2 * np.sum((cum_visits - sorted_visits/2) / cum_visits[-1]) / len(visits)
                stats['gini_concentration'] = gini
        
        # Mémoriser ces statistiques
        self.mcts_stats = stats
        
        # Mettre à jour l'historique des statistiques clés
        for key in self.mcts_stats_history.keys():
            if key in stats:
                self.mcts_stats_history[key].append(stats[key])
        
        return stats
    
    def select_action(self, observation: np.ndarray, legal_actions: List[int] = None, deterministic: bool = False) -> int:
        """
        Sélectionne une action à partir de l'observation.
        
        Args:
            observation: État du jeu
            legal_actions: Liste des actions légales (optionnel)
            deterministic: Si True, utilise la politique déterministe (exploitation)
            
        Returns:
            Action sélectionnée
        """
        # Initialiser les réseaux si ce n'est pas déjà fait
        if self.representation_network is None:
            self.init_networks(observation)
        
        # Convertir l'observation en tensor sur le bon dispositif
        if isinstance(observation, np.ndarray):
            observation_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        else:
            observation_tensor = observation.to(self.device)
        
        # Si aucune action légale n'est fournie, considérer toutes les actions comme légales
        if legal_actions is None:
            legal_actions = list(range(self.action_dim))
        
        # Obtenir l'état latent avec le réseau de représentation
        with torch.no_grad():
            hidden_state = self.representation_network(observation_tensor)
        
        # Température pour l'exploration
        temperature = 0.0 if deterministic else self.mcts.temperature
        
        # Exécuter MCTS avec l'état latent
        root = self.mcts.run(hidden_state, legal_actions)
        
        # Collecter des statistiques sur la recherche MCTS
        self.collect_mcts_statistics(root)
        
        # Sélectionner une action basée sur les visites
        action, probs = self.mcts.get_action_probabilities(root, temperature)
        
        # Diminuer la température au fil du temps
        if not deterministic:
            self.mcts.decay_temperature()
        
        return action
    
    def store_transition(self, observation: np.ndarray, action: int, reward: float, 
                        next_observation: np.ndarray, done: bool, info: Dict = None):
        """
        Stocke une transition dans le tampon de rejeu.
        
        Args:
            observation: État actuel
            action: Action effectuée
            reward: Récompense reçue
            next_observation: État suivant
            done: Indicateur de fin d'épisode
            info: Informations supplémentaires (optionnel)
        """
        self.replay_buffer.push((observation, action, reward, next_observation, done, info))
    
    def update(self) -> Dict[str, float]:
        """
        Met à jour les réseaux à partir du tampon de rejeu.
        
        Returns:
            Dictionnaire des pertes (policy, value, reward, total)
        """
        # Vérifier si le tampon est suffisamment rempli
        if len(self.replay_buffer) < self.min_replay_buffer_size:
            return None
        
        # Échantillonner des transitions
        transitions = self.replay_buffer.sample(self.batch_size)
        
        # Entraîner les réseaux
        losses = self._train_networks(transitions)
        
        # Mettre à jour le compteur d'étapes
        self.training_step += 1
        
        return losses
    
    def _train_networks(self, transitions):
        """
        Entraîne les réseaux à partir des transitions.
        
        Args:
            transitions: Liste de transitions
            
        Returns:
            Dictionnaire des pertes
        """
        # Décomposer les transitions
        observations, actions, rewards, next_observations, dones, infos = zip(*transitions)
        
        # Convertir en tensors
        observations = torch.FloatTensor(np.array(observations)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        
        # Représentation des états
        hidden_states = self.representation_network(observations)
        
        # Prédictions pour les états actuels
        policy_logits, predicted_values = self.prediction_network(hidden_states)
        
        # Dynamique et récompense
        next_hidden_states, predicted_rewards = self.dynamics_network(hidden_states, actions)
        
        # Prédictions pour les états suivants
        _, bootstrap_values = self.prediction_network(next_hidden_states)
        
        # TD-Target = r + gamma * V(s') * (1 - done)
        target_values = rewards + self.discount_factor * bootstrap_values * (1 - dones)
        
        # Perte de valeur
        value_loss = nn.MSELoss()(predicted_values, target_values.detach())
        
        # Perte de récompense
        reward_loss = nn.MSELoss()(predicted_rewards, rewards)
        
        # Perte de politique (pour simplifier, utiliser une politique uniforme comme cible)
        # Dans une implémentation complète, utilisez des résultats de MCTS ou des données d'expert
        uniform_policy = torch.ones_like(policy_logits) / self.action_dim
        policy_loss = nn.CrossEntropyLoss()(policy_logits, uniform_policy)
        
        # Perte totale avec pondération
        total_loss = value_loss + reward_loss + policy_loss
        
        # Optimisation
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping pour éviter l'explosion de gradient
        torch.nn.utils.clip_grad_norm_(
            self.representation_network.parameters(), 
            self.config["training"]["max_grad_norm"]
        )
        torch.nn.utils.clip_grad_norm_(
            self.dynamics_network.parameters(), 
            self.config["training"]["max_grad_norm"]
        )
        torch.nn.utils.clip_grad_norm_(
            self.prediction_network.parameters(), 
            self.config["training"]["max_grad_norm"]
        )
        
        self.optimizer.step()
        
        # Enregistrer les pertes pour TensorBoard
        self.last_losses = {
            'value': value_loss.item(),
            'policy': policy_loss.item(),
            'reward': reward_loss.item(),
            'total': total_loss.item()
        }
        
        return self.last_losses
    
    def log_metrics_to_writer(self, writer, prefix=''):
        """
        Journalise des métriques supplémentaires dans TensorBoard.
        
        Args:
            writer: Le SummaryWriter de TensorBoard
            prefix: Préfixe pour les noms des métriques (optionnel)
        """
        if not hasattr(self, 'last_losses') or not self.last_losses:
            return
            
        # Journaliser les pertes
        for loss_name, loss_value in self.last_losses.items():
            writer.add_scalar(f'{prefix}Loss/{loss_name}', loss_value, self.training_step)
        
        # Journaliser les statistiques du tampon de relecture
        writer.add_scalar(f'{prefix}Buffer/size', len(self.replay_buffer), self.training_step)
        
        # Journaliser la température de MCTS
        writer.add_scalar(f'{prefix}MCTS/temperature', self.mcts.temperature, self.training_step)
        
        # Journaliser le taux d'apprentissage actuel
        for i, param_group in enumerate(self.optimizer.param_groups):
            writer.add_scalar(f'{prefix}Training/learning_rate_{i}', param_group['lr'], self.training_step)
        
        # Si nous avons des statistiques de MCTS récentes, les journaliser
        if hasattr(self, 'mcts_stats'):
            for stat_name, stat_value in self.mcts_stats.items():
                writer.add_scalar(f'{prefix}MCTS/{stat_name}', stat_value, self.training_step)
        
        # Journaliser les moyennes mobiles des statistiques MCTS
        if hasattr(self, 'mcts_stats_history'):
            for stat_name, stat_values in self.mcts_stats_history.items():
                if len(stat_values) > 0:
                    avg_value = sum(stat_values) / len(stat_values)
                    writer.add_scalar(f'{prefix}MCTS/avg_{stat_name}_100', avg_value, self.training_step)

    def save(self, path: str):
        """
        Sauvegarde les poids des réseaux.
        
        Args:
            path: Chemin où sauvegarder les poids
        """
        # Vérifier que les réseaux sont initialisés
        if self.representation_network is None:
            logger.warning("Tentative de sauvegarde sans réseaux initialisés")
            return
        
        # Créer le dossier si nécessaire
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Construire le dictionnaire d'état
        state = {
            'representation': self.representation_network.state_dict(),
            'dynamics': self.dynamics_network.state_dict(),
            'prediction': self.prediction_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'temperature': self.mcts.temperature,
            'config': self.config
        }
        
        # Sauvegarder le modèle
        torch.save(state, path)
        logger.info(f"Modèle sauvegardé à {path}")
    
    def load(self, path: str):
        """
        Charge les poids des réseaux.
        
        Args:
            path: Chemin des poids à charger
        """
        # Vérifier que le fichier existe
        if not os.path.exists(path):
            logger.error(f"Le fichier de modèle {path} n'existe pas")
            return False
            
        # Charger l'état
        state = torch.load(path, map_location=self.device)
        
        # Vérifier si les réseaux sont déjà initialisés
        if self.representation_network is None:
            # Si les réseaux ne sont pas initialisés, mais que la configuration est disponible,
            # utiliser cette configuration
            if 'config' in state:
                self.config.update(state['config'])
                # Créer un tenseur d'observation factice pour initialiser les réseaux
                dummy_obs = torch.zeros(1, self.config["environment"]["observation_space"])
                self.init_networks(dummy_obs)
            else:
                logger.error("Impossible de charger le modèle: les réseaux ne sont pas initialisés")
                return False
        
        # Charger les poids des réseaux
        self.representation_network.load_state_dict(state['representation'])
        self.dynamics_network.load_state_dict(state['dynamics'])
        self.prediction_network.load_state_dict(state['prediction'])
        
        # Charger l'état de l'optimiseur
        self.optimizer.load_state_dict(state['optimizer'])
        
        # Charger les compteurs et paramètres
        self.training_step = state['training_step']
        if 'temperature' in state:
            self.mcts.temperature = state['temperature']
        
        logger.info(f"Modèle chargé depuis {path} (étape d'entraînement: {self.training_step})")
        return True 

    def export_mcts_statistics(self, file_path=None):
        """
        Exporte les statistiques MCTS accumulées vers un fichier CSV.
        
        Args:
            file_path: Chemin du fichier de sortie (optionnel, utilise un nom par défaut sinon)
        
        Returns:
            Chemin du fichier créé
        """
        if file_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"mcts_stats_{timestamp}.csv"
        
        # Créer le répertoire parent si nécessaire
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
        
        # Préparer les données
        data = []
        for stat_name, values in self.mcts_stats_history.items():
            for i, value in enumerate(values):
                while len(data) <= i:
                    data.append({})
                data[i][stat_name] = value
        
        # Écrire dans le fichier CSV
        if data:
            keys = self.mcts_stats_history.keys()
            with open(file_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for row in data:
                    writer.writerow(row)
            
            logger.info(f"Statistiques MCTS exportées vers {file_path}")
            return file_path
        else:
            logger.warning("Aucune statistique MCTS à exporter")
            return None 