"""
Module implémentant l'agent MuZero pour TFT
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from typing import Dict, List, Any, Tuple
from collections import deque

# Importation des réseaux de neurones pour MuZero
from src.networks.representation_network import RepresentationNetwork
from src.networks.dynamics_network import DynamicsNetwork
from src.networks.prediction_network import PredictionNetwork

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
            experience: Tuple (état, action, récompense, état suivant, terminé)
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

class MCTS:
    """Implémentation de Monte Carlo Tree Search pour MuZero."""
    
    def __init__(self, config: Dict[str, Any], networks: Dict[str, nn.Module]):
        """
        Initialise l'arbre de recherche Monte Carlo.
        
        Args:
            config: Dictionnaire de configuration
            networks: Dictionnaire contenant les réseaux de représentation, de dynamique et de prédiction
        """
        self.config = config
        self.representation_network = networks['representation']
        self.dynamics_network = networks['dynamics']
        self.prediction_network = networks['prediction']
        
        self.num_simulations = config['agent']['num_simulations']
        self.root_dirichlet_alpha = config['agent']['root_dirichlet_alpha']
        self.root_exploration_fraction = config['agent']['root_exploration_fraction']
        
        # Paramètres pour l'exploration UCB
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        
        self.discount = config['training']['discount_factor']
    
    def run(self, observation: np.ndarray, legal_actions: List[int]) -> Dict[str, Any]:
        """
        Exécute la recherche Monte Carlo et renvoie les statistiques et l'action choisie.
        
        Args:
            observation: État actuel du jeu
            legal_actions: Liste des actions légales
            
        Returns:
            Dictionnaire contenant les statistiques et l'action choisie
        """
        # Transformer l'observation en état latent
        with torch.no_grad():
            state = self.representation_network(torch.FloatTensor(observation).unsqueeze(0))
        
        # Initialiser la racine de l'arbre
        root = Node(0)
        root.state = state
        root.legal_actions = legal_actions
        
        # Ajouter du bruit de Dirichlet à la racine pour l'exploration
        self._add_exploration_noise(root)
        
        # Réaliser num_simulations simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            current_tree_depth = 0
            
            # Descente dans l'arbre jusqu'à atteindre un nœud non développé
            action = None
            while node.expanded() and current_tree_depth < self.config['training']['num_unroll_steps']:
                action, node = self._select_child(node)
                search_path.append(node)
                current_tree_depth += 1
            
            # Si le nœud n'est pas développé et n'est pas terminal, l'étendre
            if not node.expanded() and current_tree_depth < self.config['training']['num_unroll_steps'] and action is not None:
                self._expand_node(node, action)
            
            # Évaluer l'état
            value = self._evaluate(node.state)
            
            # Retropropagation des valeurs dans l'arbre
            self._backpropagate(search_path, value)
        
        # Vérifier si la racine a des enfants
        if not root.children:
            # Si la racine n'a pas d'enfant, choisir une action aléatoire
            action = np.random.choice(legal_actions)
            return {
                'action': action,
                'policy': {action: 1.0},
                'root': root
            }
        
        # Sélectionner l'action en fonction de la stratégie de visites
        visit_counts = np.array([child.visit_count for child in root.children.values()])
        actions = list(root.children.keys())
        action = actions[np.argmax(visit_counts)]
        
        # Calculer la politique basée sur les visites
        policy = visit_counts / np.sum(visit_counts)
        
        return {
            'action': action,
            'policy': dict(zip(actions, policy)),
            'root': root
        }
    
    def _add_exploration_noise(self, node: 'Node'):
        """
        Ajoute du bruit de Dirichlet à la racine pour favoriser l'exploration.
        
        Args:
            node: Nœud racine
        """
        # Génération du bruit de Dirichlet
        noise = np.random.dirichlet([self.root_dirichlet_alpha] * len(node.legal_actions))
        
        # Ajout du bruit aux probabilités à priori
        for i, action in enumerate(node.legal_actions):
            node.prior[action] = (1 - self.root_exploration_fraction) * node.prior.get(action, 1 / len(node.legal_actions)) + \
                                 self.root_exploration_fraction * noise[i]
    
    def _select_child(self, node: 'Node') -> Tuple[int, 'Node']:
        """
        Sélectionne un enfant selon la formule UCB.
        
        Args:
            node: Nœud parent
            
        Returns:
            Tuple (action, nœud enfant)
        """
        # Calcul du score UCB pour chaque enfant
        ucb_scores = {}
        for action in node.legal_actions:
            if action in node.children:
                child = node.children[action]
                # Formule UCB
                ucb_scores[action] = child.value() + self._ucb_score(node, child)
            else:
                # Pour les nœuds non visités, utiliser la probabilité à priori
                ucb_scores[action] = node.prior.get(action, 1 / len(node.legal_actions)) * \
                                     (self.pb_c_base + node.visit_count) / (1 + node.visit_count) * \
                                     self.pb_c_init
        
        # Sélectionner l'action avec le score UCB le plus élevé
        action = max(ucb_scores.items(), key=lambda x: x[1])[0]
        
        # Si le nœud enfant n'existe pas encore, le créer
        if action not in node.children:
            node.children[action] = Node(node.depth + 1)
            node.children[action].action = action
        
        return action, node.children[action]
    
    def _ucb_score(self, parent: 'Node', child: 'Node') -> float:
        """
        Calcule le score UCB (Upper Confidence Bound) pour un nœud enfant.
        
        Args:
            parent: Nœud parent
            child: Nœud enfant
            
        Returns:
            Score UCB
        """
        pb_c = np.log((parent.visit_count + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
        pb_c *= np.sqrt(parent.visit_count) / (child.visit_count + 1)
        
        prior_score = pb_c * parent.prior.get(child.action, 1 / len(parent.legal_actions))
        value_score = child.value()
        
        return prior_score + value_score
    
    def _expand_node(self, node: 'Node', action: int):
        """
        Développe un nœud en utilisant le réseau de dynamique pour prédire l'état suivant.
        
        Args:
            node: Nœud à développer
            action: Action qui a mené à ce nœud
        """
        with torch.no_grad():
            # Convertir action en one-hot
            action_one_hot = torch.zeros(1, len(node.legal_actions))
            action_index = node.legal_actions.index(action)
            action_one_hot[0, action_index] = 1
            
            # Prédire l'état suivant et la récompense
            next_state, reward = self.dynamics_network(node.state, action_one_hot)
            
            # Prédire la politique et la valeur
            policy, value = self.prediction_network(next_state)
        
        # Mettre à jour le nœud
        node.state = next_state
        node.reward = reward.item()
        node.action = action
        
        # Initialiser les probabilités à priori pour les actions légales
        priors = nn.functional.softmax(policy, dim=1).squeeze().numpy()
        for i, a in enumerate(node.legal_actions):
            node.prior[a] = priors[i]
    
    def _evaluate(self, state: torch.Tensor) -> float:
        """
        Évalue un état à l'aide du réseau de prédiction.
        
        Args:
            state: État latent
            
        Returns:
            Valeur estimée de l'état
        """
        with torch.no_grad():
            _, value = self.prediction_network(state)
        
        return value.item()
    
    def _backpropagate(self, search_path: List['Node'], value: float):
        """
        Rétropropage les valeurs dans l'arbre.
        
        Args:
            search_path: Chemin de recherche dans l'arbre
            value: Valeur à rétropropager
        """
        # Rétropropager depuis le dernier nœud
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            
            # Actualiser la valeur avec le facteur de remise
            if node.reward is not None:
                value = node.reward + self.discount * value

class Node:
    """Nœud dans l'arbre de recherche Monte Carlo."""
    
    def __init__(self, depth: int):
        """
        Initialise un nœud.
        
        Args:
            depth: Profondeur du nœud dans l'arbre
        """
        self.depth = depth
        self.state = None
        self.reward = None
        self.action = None
        self.visit_count = 0
        self.value_sum = 0
        self.children = {}
        self.prior = {}
        self.legal_actions = []
    
    def expanded(self) -> bool:
        """
        Vérifie si le nœud a été développé.
        
        Returns:
            True si le nœud a été développé, False sinon
        """
        return len(self.children) > 0
    
    def value(self) -> float:
        """
        Calcule la valeur moyenne du nœud.
        
        Returns:
            Valeur moyenne
        """
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

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
        self.action_dim = 5  # Exemple: Buy, Sell, Position, Level up, Refresh
        
        # Hyperparamètres
        self.batch_size = config['training']['batch_size']
        self.learning_rate = config['training']['learning_rate']
        self.discount_factor = config['training']['discount_factor']
        self.min_replay_buffer_size = config['training']['min_replay_buffer_size']
        self.replay_buffer_size = config['training']['replay_buffer_size']
        self.weight_decay = config['training']['weight_decay']
        
        # Tampon de rejeu
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        
        # Réseaux de neurones
        self.networks = {}
        
        # Optimiseur
        self.optimizer = None
        
        # Compteur d'étapes d'entraînement
        self.training_step = 0
        
        # Historique des pertes pour le suivi TensorBoard
        self.last_losses = {}
    
    def init_networks(self, observation: np.ndarray):
        """
        Initialise les réseaux de neurones.
        
        Args:
            observation: Exemple d'observation pour déterminer les dimensions
        """
        self.observation_dim = observation.shape[0]
        
        # Taille de la représentation latente
        hidden_dim = self.config['agent']['representation_channels']
        
        # Initialiser les réseaux
        self.networks['representation'] = RepresentationNetwork(
            observation_dim=self.observation_dim,
            hidden_dim=hidden_dim
        )
        
        self.networks['dynamics'] = DynamicsNetwork(
            hidden_dim=hidden_dim,
            action_dim=self.action_dim
        )
        
        self.networks['prediction'] = PredictionNetwork(
            hidden_dim=hidden_dim,
            action_dim=self.action_dim
        )
        
        # Initialiser l'optimiseur
        parameters = list(self.networks['representation'].parameters()) + \
                     list(self.networks['dynamics'].parameters()) + \
                     list(self.networks['prediction'].parameters())
        
        self.optimizer = optim.Adam(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Initialiser MCTS
        self.mcts = MCTS(self.config, self.networks)
    
    def select_action(self, observation: np.ndarray, legal_actions: List[int], training: bool = True) -> int:
        """
        Sélectionne une action à partir de l'observation.
        
        Args:
            observation: État du jeu
            legal_actions: Liste des actions légales
            training: Si True, utilise une politique d'exploration, sinon utilise la meilleure action
            
        Returns:
            Action sélectionnée
        """
        # Initialiser les réseaux si ce n'est pas déjà fait
        if self.observation_dim is None:
            self.init_networks(observation)
        
        # Utiliser MCTS pour sélectionner une action
        if training:
            # En mode entraînement, utiliser MCTS avec exploration
            mcts_output = self.mcts.run(observation, legal_actions)
            action = mcts_output['action']
        else:
            # En mode évaluation, prendre la meilleure action directement
            mcts_output = self.mcts.run(observation, legal_actions)
            action = mcts_output['action']
        
        return action
    
    def update(self, observation: np.ndarray, action: int, reward: float, next_observation: np.ndarray, done: bool):
        """
        Met à jour le tampon de rejeu avec une nouvelle expérience.
        
        Args:
            observation: État actuel
            action: Action effectuée
            reward: Récompense reçue
            next_observation: État suivant
            done: Indicateur de fin d'épisode
        """
        # Stocker l'expérience dans le tampon de rejeu
        self.replay_buffer.push((observation, action, reward, next_observation, done))
        
        # Mettre à jour les réseaux si le tampon de rejeu est suffisamment rempli
        if len(self.replay_buffer) >= self.min_replay_buffer_size:
            self._train_networks()
    
    def _train_networks(self):
        """Entraîne les réseaux à partir des expériences du tampon de rejeu."""
        # Échantillonner des expériences
        experiences = self.replay_buffer.sample(self.batch_size)
        
        # Décomposer les expériences
        observations, actions, rewards, next_observations, dones = zip(*experiences)
        
        # Convertir en tenseurs PyTorch
        observations = torch.FloatTensor(np.array(observations))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_observations = torch.FloatTensor(np.array(next_observations))
        dones = torch.FloatTensor(np.array(dones))
        
        # Forward pass
        states = self.networks['representation'](observations)
        
        # Convertir actions en one-hot
        action_one_hot = torch.zeros(self.batch_size, self.action_dim)
        for i, a in enumerate(actions):
            action_one_hot[i, a] = 1
        
        # Prédiction de l'état suivant et des récompenses
        next_states, predicted_rewards = self.networks['dynamics'](states, action_one_hot)
        
        # Prédiction des politiques et des valeurs
        predicted_policies, predicted_values = self.networks['prediction'](states)
        predicted_next_values = self.networks['prediction'](next_states)[1]
        
        # Targets pour l'entraînement
        # Target pour la valeur: récompense + facteur de remise * valeur suivante
        target_values = rewards + self.discount_factor * predicted_next_values * (1 - dones)
        
        # On utilise les valeurs cibles comme supervision
        value_loss = nn.functional.mse_loss(predicted_values, target_values)
        
        # Perte de politique (utilisation des visites MCTS comme cibles)
        # Pour simplifier, on utilise une distribution uniforme comme cible
        # Dans une implémentation complète, utiliser les visites MCTS
        target_policies = torch.ones_like(predicted_policies) / self.action_dim
        policy_loss = nn.functional.cross_entropy(predicted_policies, target_policies)
        
        # Perte de récompense
        reward_loss = nn.functional.mse_loss(predicted_rewards, rewards.unsqueeze(1))
        
        # Perte totale
        total_loss = value_loss + policy_loss + reward_loss
        
        # Backward pass et optimisation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Sauvegarder les valeurs de perte pour le suivi
        self.last_losses = {
            'value': value_loss.item(),
            'policy': policy_loss.item(),
            'reward': reward_loss.item(),
            'total': total_loss.item()
        }
        
        # Incrémenter le compteur d'étapes
        self.training_step += 1
    
    def train(self, env, num_episodes: int = None):
        """
        Entraîne l'agent dans l'environnement spécifié.
        
        Args:
            env: Environnement d'entraînement
            num_episodes: Nombre d'épisodes d'entraînement (utilise config si None)
        """
        if num_episodes is None:
            num_episodes = self.config['training']['num_episodes']
        
        for episode in range(num_episodes):
            observation = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Déterminer les actions légales (dans un vrai jeu, fournies par l'environnement)
                legal_actions = list(range(self.action_dim))
                
                # Sélectionner une action
                action = self.select_action(observation, legal_actions)
                
                # Convertir l'action en format attendu par l'environnement
                env_action = self._convert_action(action)
                
                # Effectuer l'action dans l'environnement
                next_observation, reward, done, _ = env.step(env_action)
                
                # Mettre à jour le tampon de rejeu et entraîner les réseaux
                self.update(observation, action, reward, next_observation, done)
                
                # Mettre à jour l'observation courante
                observation = next_observation
                episode_reward += reward
            
            print(f"Épisode {episode + 1}/{num_episodes}, Récompense: {episode_reward}")
    
    def _convert_action(self, action_index: int) -> Dict[str, Any]:
        """
        Convertit un index d'action en action compatible avec l'environnement.
        
        Args:
            action_index: Index de l'action
            
        Returns:
            Action au format attendu par l'environnement
        """
        # Mapping des indices d'action vers les actions de l'environnement
        if action_index == 0:
            # Achat (index de champion aléatoire)
            return {'type': 'buy', 'champion_index': random.randint(0, 4)}
        elif action_index == 1:
            # Vente (index de champion aléatoire)
            return {'type': 'sell', 'unit_index': random.randint(0, 8), 'location': 'bench'}
        elif action_index == 2:
            # Positionnement (indices aléatoires)
            return {
                'type': 'position',
                'from_idx': random.randint(0, 8),
                'from_loc': 'bench',
                'to_idx': random.randint(0, 27),
                'to_loc': 'board'
            }
        elif action_index == 3:
            # Monter de niveau
            return {'type': 'level_up'}
        elif action_index == 4:
            # Rafraîchir la boutique
            return {'type': 'refresh'}
        
        # Action par défaut (ne devrait pas arriver)
        return {'type': 'refresh'}
    
    def evaluate(self, env, num_games: int = 100) -> Dict[str, float]:
        """
        Évalue l'agent dans l'environnement spécifié.
        
        Args:
            env: Environnement d'évaluation
            num_games: Nombre de parties pour l'évaluation
            
        Returns:
            Dictionnaire contenant les statistiques d'évaluation
        """
        placements = []
        wins = 0
        
        for game in range(num_games):
            observation = env.reset()
            done = False
            
            while not done:
                # Déterminer les actions légales
                legal_actions = list(range(self.action_dim))
                
                # Sélectionner la meilleure action (sans exploration)
                action = self.select_action(observation, legal_actions, training=False)
                
                # Convertir l'action
                env_action = self._convert_action(action)
                
                # Effectuer l'action
                observation, _, done, info = env.step(env_action)
            
            # Déterminer le classement final
            player = info['player']
            if not player['eliminated']:
                placement = 1
                wins += 1
            else:
                eliminated_players = sum(1 for p in env.players if p['eliminated'])
                placement = env.num_players - eliminated_players + 1
            
            placements.append(placement)
        
        # Calculer les statistiques
        avg_placement = sum(placements) / len(placements)
        win_rate = (wins / num_games) * 100
        
        return {
            'avg_placement': avg_placement,
            'win_rate': win_rate
        }
    
    def save(self, path: str):
        """
        Sauvegarde les poids des réseaux.
        
        Args:
            path: Chemin où sauvegarder les poids
        """
        state = {
            'representation': self.networks['representation'].state_dict(),
            'dynamics': self.networks['dynamics'].state_dict(),
            'prediction': self.networks['prediction'].state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_step': self.training_step
        }
        torch.save(state, path)
    
    def load(self, path: str):
        """
        Charge les poids des réseaux.
        
        Args:
            path: Chemin des poids à charger
        """
        state = torch.load(path)
        
        # Vérifier si les réseaux sont déjà initialisés
        if self.observation_dim is None:
            # On ne peut pas charger les poids si les réseaux ne sont pas initialisés
            raise ValueError("Les réseaux doivent être initialisés avant de charger les poids")
        
        self.networks['representation'].load_state_dict(state['representation'])
        self.networks['dynamics'].load_state_dict(state['dynamics'])
        self.networks['prediction'].load_state_dict(state['prediction'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.training_step = state['training_step'] 