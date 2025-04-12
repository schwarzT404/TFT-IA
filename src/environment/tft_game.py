"""
Module définissant l'environnement de jeu TFT
"""
import numpy as np
from typing import List, Dict, Any, Tuple

class TFTGame:
    """
    Simulation de l'environnement Teamfight Tactics.
    Cette classe gère les mécaniques de jeu, les états, les transitions et les récompenses.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise l'environnement de jeu avec la configuration spécifiée.
        
        Args:
            config: Dictionnaire contenant les paramètres de configuration
        """
        self.config = config
        self.state_compression = config['environment'].get('state_compression', True)
        self.parallel_combat = config['environment'].get('parallel_combat', True)
        self.reward_shaping = config['environment'].get('reward_shaping', True)
        self.num_players = config['environment'].get('num_players', 4)
        self.max_rounds = config['environment'].get('max_rounds', 20)
        
        # État du jeu
        self.current_stage = 1  # Stage actuel (1-1, 1-2, etc.)
        self.players = []  # Liste des joueurs
        self.current_player_idx = 0  # Indice du joueur actuel
        self.shop = []  # Champions dans la boutique
        self.champion_pool = {}  # Pool de champions disponibles
        self.game_over = False  # Indicateur de fin de jeu
        
        # Initialisation
        self._initialize_game()
    
    def _initialize_game(self):
        """Initialise les composants du jeu, dont les joueurs et le pool de champions."""
        # Initialisation des joueurs
        self.players = [self._create_player(i) for i in range(self.num_players)]
        
        # Initialisation du pool de champions (simplifié)
        # Dans une implémentation complète, cela comprendrait tous les champions avec leurs traits et coûts
        self.champion_pool = self._initialize_champion_pool()
    
    def _create_player(self, player_id: int) -> Dict[str, Any]:
        """
        Crée un nouveau joueur avec les attributs par défaut.
        
        Args:
            player_id: Identifiant unique du joueur
            
        Returns:
            Dictionnaire contenant les attributs du joueur
        """
        return {
            'id': player_id,
            'health': 100,
            'gold': 0,
            'level': 1,
            'xp': 0,
            'bench': [],
            'board': [],
            'items': [],
            'streak': 0,
            'eliminated': False
        }
    
    def _initialize_champion_pool(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialise le pool de champions disponibles.
        
        Returns:
            Dictionnaire des champions avec leurs attributs
        """
        # Version simplifiée - dans une implémentation complète, cela inclurait
        # tous les champions avec leurs statistiques, traits, etc.
        return {
            # Format: 'nom': {'cost': coût, 'traits': [liste de traits], 'count': nombre disponible}
            'renekton': {'cost': 1, 'traits': ['divinicorp', 'bastion'], 'count': 29},
            'samira': {'cost': 4, 'traits': ['demon_urbain', 'amp'], 'count': 12},
            'urgot': {'cost': 4, 'traits': ['pegre', 'dynamo'], 'count': 12},
            'viego': {'cost': 5, 'traits': ['soul_killer', 'boeuf_dore', 'technophile'], 'count': 10}
            # Ajouter d'autres champions selon les besoins
        }
    
    def reset(self) -> np.ndarray:
        """
        Réinitialise l'environnement et renvoie l'état initial.
        
        Returns:
            Observation initiale (état du jeu)
        """
        self._initialize_game()
        self.game_over = False
        return self._get_observation()
    
    def step(self, action: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Effectue une action dans l'environnement et renvoie le nouvel état, la récompense et des informations supplémentaires.
        
        Args:
            action: Action à effectuer (achat, vente, positionnement, etc.)
            
        Returns:
            Tuple (observation, récompense, terminé, infos)
        """
        player = self.players[self.current_player_idx]
        
        # Traitement de l'action selon son type
        if action['type'] == 'buy':
            self._handle_buy_action(player, action)
        elif action['type'] == 'sell':
            self._handle_sell_action(player, action)
        elif action['type'] == 'position':
            self._handle_position_action(player, action)
        elif action['type'] == 'level_up':
            self._handle_level_up_action(player)
        elif action['type'] == 'refresh':
            self._handle_refresh_action(player)
        
        # Après les actions du joueur, on simule le combat s'il s'agit d'un tour de combat
        if self._is_combat_round():
            self._simulate_combat()
        
        # Vérifier la condition de fin de jeu
        if self._check_game_over():
            self.game_over = True
        
        # Calculer la récompense
        reward = self._calculate_reward(player)
        
        # Passer au joueur suivant ou au tour suivant
        self._next_turn()
        
        # Renvoyer l'observation, la récompense, l'indicateur de fin et les infos supplémentaires
        return self._get_observation(), reward, self.game_over, {'player': player}
    
    def _handle_buy_action(self, player: Dict[str, Any], action: Dict[str, Any]):
        """Gère l'action d'achat d'un champion."""
        champion_index = action.get('champion_index')
        if champion_index is not None and 0 <= champion_index < len(self.shop):
            champion = self.shop[champion_index]
            if champion is not None and player['gold'] >= champion['cost']:
                player['gold'] -= champion['cost']
                player['bench'].append(champion)
                self.shop[champion_index] = None  # Champion acheté
    
    def _handle_sell_action(self, player: Dict[str, Any], action: Dict[str, Any]):
        """Gère l'action de vente d'un champion."""
        unit_index = action.get('unit_index')
        location = action.get('location', 'bench')  # 'bench' ou 'board'
        
        if location == 'bench' and 0 <= unit_index < len(player['bench']):
            if player['bench'][unit_index] is not None:
                champion = player['bench'].pop(unit_index)
                player['gold'] += champion['cost']
        elif location == 'board' and 0 <= unit_index < len(player['board']):
            if player['board'][unit_index] is not None:
                champion = player['board'].pop(unit_index)
                player['gold'] += champion['cost']
    
    def _handle_position_action(self, player: Dict[str, Any], action: Dict[str, Any]):
        """Gère l'action de positionnement d'un champion sur le plateau."""
        from_idx = action.get('from_idx')
        from_loc = action.get('from_loc')  # 'bench' ou 'board'
        to_idx = action.get('to_idx')
        to_loc = action.get('to_loc')  # 'bench' ou 'board'
        
        # Vérifier que les indices sont valides
        if (from_loc == 'bench' and 0 <= from_idx < len(player['bench'])) and \
           (to_loc == 'board' and 0 <= to_idx < 28):  # 28 hexagones sur le plateau
            # Vérifier que le champion existe
            if player['bench'][from_idx] is not None:
                # Déplacer de bench à board
                champion = player['bench'].pop(from_idx)
                
                # S'assurer que la liste board a suffisamment d'éléments
                while len(player['board']) <= to_idx:
                    player['board'].append(None)
                    
                player['board'][to_idx] = champion
    
    def _handle_level_up_action(self, player: Dict[str, Any]):
        """Gère l'action d'achat d'XP pour monter de niveau."""
        if player['gold'] >= 4:  # Coût pour monter de niveau
            player['gold'] -= 4
            player['xp'] += 4
            
            # Vérifier si le joueur monte de niveau
            xp_required = [0, 2, 6, 10, 20, 36, 56, 80, 100]  # XP nécessaire pour chaque niveau
            if player['level'] < 9 and player['xp'] >= xp_required[player['level']]:
                player['level'] += 1
    
    def _handle_refresh_action(self, player: Dict[str, Any]):
        """Gère l'action de rafraîchissement de la boutique."""
        if player['gold'] >= 2:  # Coût pour rafraîchir
            player['gold'] -= 2
            self._refresh_shop(player)
    
    def _refresh_shop(self, player: Dict[str, Any]):
        """Rafraîchit la boutique avec de nouveaux champions."""
        self.shop = []
        
        # Nombre de champions dans la boutique basé sur le niveau du joueur
        shop_size = 5
        
        # Probabilités de tirage basées sur le niveau du joueur
        probabilities = self.config['game_mechanics']['champion_probabilities'].get(
            player['level'], {1: 1.0}  # Par défaut, 100% de champions de coût 1
        )
        
        # Générer de nouveaux champions pour la boutique
        for _ in range(shop_size):
            cost = self._select_champion_cost(probabilities)
            eligible_champions = [c for c, attrs in self.champion_pool.items() 
                                if attrs['cost'] == cost and attrs['count'] > 0]
            
            if eligible_champions:
                champion_name = np.random.choice(eligible_champions)
                champion = {
                    'name': champion_name,
                    'cost': self.champion_pool[champion_name]['cost'],
                    'traits': self.champion_pool[champion_name]['traits'].copy()
                }
                self.shop.append(champion)
                self.champion_pool[champion_name]['count'] -= 1
            else:
                # Si aucun champion de ce coût n'est disponible, ajouter None
                self.shop.append(None)
    
    def _select_champion_cost(self, probabilities: Dict[int, float]) -> int:
        """
        Sélectionne un coût de champion basé sur les probabilités.
        
        Args:
            probabilities: Dictionnaire {coût: probabilité}
            
        Returns:
            Coût du champion sélectionné
        """
        costs = list(probabilities.keys())
        probs = [probabilities[cost] for cost in costs]
        return np.random.choice(costs, p=probs)
    
    def _is_combat_round(self) -> bool:
        """
        Vérifie si le tour actuel est un tour de combat.
        
        Returns:
            True si c'est un tour de combat, False sinon
        """
        # Dans TFT, les tours alternent entre les tours de préparation et les tours de combat
        return (self.current_stage % 2) == 0
    
    def _simulate_combat(self):
        """Simule les combats entre les joueurs."""
        # Déterminer les matchups (qui affronte qui)
        matchups = self._determine_matchups()
        
        # Simuler chaque combat
        for player1_idx, player2_idx in matchups:
            player1 = self.players[player1_idx]
            player2 = self.players[player2_idx]
            
            # Si un joueur est éliminé, skip ce combat
            if player1['eliminated'] or player2['eliminated']:
                continue
            
            winner, damage = self._simulate_single_combat(player1, player2)
            
            # Appliquer les résultats du combat
            if winner == 1:
                player2['health'] -= damage
                player1['streak'] = max(1, player1['streak'] + 1)
                player2['streak'] = min(-1, player2['streak'] - 1)
            else:
                player1['health'] -= damage
                player2['streak'] = max(1, player2['streak'] + 1)
                player1['streak'] = min(-1, player1['streak'] - 1)
            
            # Vérifier si un joueur est éliminé
            if player1['health'] <= 0:
                player1['eliminated'] = True
            if player2['health'] <= 0:
                player2['eliminated'] = True
    
    def _determine_matchups(self) -> List[Tuple[int, int]]:
        """
        Détermine les matchups entre les joueurs pour la phase de combat.
        
        Returns:
            Liste de tuples (indice_joueur1, indice_joueur2)
        """
        # Sélectionner les joueurs non éliminés
        active_players = [i for i, p in enumerate(self.players) if not p['eliminated']]
        
        # Mélanger les joueurs pour les matchups
        np.random.shuffle(active_players)
        
        # Créer les paires
        matchups = []
        for i in range(0, len(active_players) - 1, 2):
            matchups.append((active_players[i], active_players[i + 1]))
        
        # S'il y a un nombre impair de joueurs, le dernier joue contre un "ghost" (simulation)
        if len(active_players) % 2 == 1:
            ghost_player_idx = len(self.players)  # Indice fictif pour le ghost
            matchups.append((active_players[-1], ghost_player_idx))
        
        return matchups
    
    def _simulate_single_combat(self, player1: Dict[str, Any], player2: Dict[str, Any]) -> Tuple[int, int]:
        """
        Simule un combat entre deux joueurs.
        
        Args:
            player1: Premier joueur
            player2: Deuxième joueur
            
        Returns:
            Tuple (gagnant, dégâts) où gagnant est 1 ou 2
        """
        # Calculer la force de chaque board (version simplifiée)
        board1_strength = self._calculate_board_strength(player1)
        board2_strength = self._calculate_board_strength(player2)
        
        # Déterminer le gagnant
        winner = 1 if board1_strength > board2_strength else 2
        
        # Calculer les dégâts (version simplifiée)
        # Dans une implémentation complète, les dégâts dépendent des unités restantes
        if winner == 1:
            damage = max(2, player1['level'] + len([u for u in player1['board'] if u is not None]))
        else:
            damage = max(2, player2['level'] + len([u for u in player2['board'] if u is not None]))
        
        return winner, damage
    
    def _calculate_board_strength(self, player: Dict[str, Any]) -> float:
        """
        Calcule la force du plateau d'un joueur.
        
        Args:
            player: Joueur dont on veut évaluer la force du plateau
            
        Returns:
            Score de force du plateau
        """
        # Sommer la force de chaque unité (version simplifiée)
        strength = 0
        
        for unit in player['board']:
            if unit is not None:
                # Dans une implémentation complète, on tiendrait compte des synergies,
                # des niveaux d'étoile, des objets, etc.
                strength += unit['cost'] * 2  # Force basée sur le coût
        
        # Bonus pour les synergies (version simplifiée)
        traits_count = {}
        for unit in player['board']:
            if unit is not None:
                for trait in unit['traits']:
                    traits_count[trait] = traits_count.get(trait, 0) + 1
        
        # Bonus pour les traits activés
        for trait, count in traits_count.items():
            if count >= 2:  # Seuil minimum pour activer un trait
                strength += count * 1.5
        
        return strength
    
    def _check_game_over(self) -> bool:
        """
        Vérifie si la partie est terminée.
        
        Returns:
            True si la partie est terminée, False sinon
        """
        # La partie est terminée s'il ne reste qu'un joueur ou si on a atteint le nombre maximum de tours
        active_players = sum(1 for p in self.players if not p['eliminated'])
        return active_players <= 1 or self.current_stage >= self.max_rounds
    
    def _next_turn(self):
        """Passe au joueur suivant ou au tour suivant."""
        # Passer au joueur suivant
        self.current_player_idx = (self.current_player_idx + 1) % self.num_players
        
        # Si on a fait le tour de tous les joueurs, passer au tour suivant
        if self.current_player_idx == 0:
            self.current_stage += 1
            
            # Augmenter l'or de chaque joueur au début du tour
            for player in self.players:
                if not player['eliminated']:
                    # Or de base par tour + intérêts
                    player['gold'] += 5
                    interest = min(5, player['gold'] // 10)
                    player['gold'] += interest
                    
                    # Bonus de streak
                    if abs(player['streak']) >= 2:
                        streak_bonus = min(3, abs(player['streak']) - 1)
                        player['gold'] += streak_bonus
    
    def _calculate_reward(self, player: Dict[str, Any]) -> float:
        """
        Calcule la récompense pour le joueur actuel.
        
        Args:
            player: Joueur pour lequel calculer la récompense
            
        Returns:
            Récompense
        """
        # Récompense de base selon le classement final
        if self.game_over:
            active_players = sum(1 for p in self.players if not p['eliminated'])
            if not player['eliminated']:
                # Le joueur a gagné
                return 1.0
            else:
                # Le joueur a perdu, récompense basée sur le classement
                eliminated_players = sum(1 for p in self.players if p['eliminated'])
                placement = eliminated_players
                # Normalisé entre -1 et 0, avec -1 pour la dernière place et 0 pour la deuxième place
                return -1.0 + (placement - 1) / (self.num_players - 1)
        
        # Récompenses intermédiaires si reward_shaping est activé
        if self.reward_shaping:
            reward = 0.0
            
            # Récompense pour maintenir une économie saine
            if player['gold'] >= 50:
                reward += 0.01
            
            # Récompense pour maintenir une bonne santé
            health_ratio = player['health'] / 100.0
            reward += 0.005 * health_ratio
            
            # Récompense pour les streaks
            if abs(player['streak']) >= 3:
                reward += 0.01
            
            return reward
        
        # Si reward_shaping est désactivé, pas de récompense intermédiaire
        return 0.0
    
    def _get_observation(self) -> np.ndarray:
        """
        Renvoie l'observation actuelle du jeu pour l'agent.
        
        Returns:
            Observation (état du jeu)
        """
        # Si la compression d'état est activée, renvoyer une version compacte de l'état
        if self.state_compression:
            return self._get_compressed_observation()
        
        # Sinon, renvoyer l'état complet
        return self._get_full_observation()
    
    def _get_compressed_observation(self) -> np.ndarray:
        """
        Renvoie une version compressée de l'observation.
        
        Returns:
            Observation compressée
        """
        # Exemple de compression d'état - dans une implémentation complète,
        # cela serait plus sophistiqué et optimisé pour l'apprentissage
        player = self.players[self.current_player_idx]
        
        # Caractéristiques du joueur
        player_features = np.array([
            player['health'] / 100.0,  # Santé normalisée
            player['gold'] / 50.0,     # Or normalisé
            player['level'] / 9.0,     # Niveau normalisé
            player['streak'] / 10.0    # Streak normalisée
        ])
        
        # Représentation du plateau (version simplifiée)
        board_size = 28  # Taille maximale du plateau
        board = np.zeros((board_size, 3))  # 3 caractéristiques par hexagone: présence, coût, force
        
        for i, unit in enumerate(player['board']):
            if i < board_size and unit is not None:
                board[i, 0] = 1.0  # Présence d'une unité
                board[i, 1] = unit['cost'] / 5.0  # Coût normalisé
                board[i, 2] = self._calculate_unit_strength(unit) / 10.0  # Force normalisée
        
        # Aplatir le tableau du plateau
        flat_board = board.flatten()
        
        # Boutique
        shop_size = 5
        shop = np.zeros((shop_size, 2))  # 2 caractéristiques par champion: présence, coût
        
        for i, champion in enumerate(self.shop):
            if i < shop_size and champion is not None:
                shop[i, 0] = 1.0  # Présence d'un champion
                shop[i, 1] = champion['cost'] / 5.0  # Coût normalisé
        
        # Aplatir le tableau de la boutique
        flat_shop = shop.flatten()
        
        # Concaténer toutes les caractéristiques
        observation = np.concatenate([
            player_features,
            flat_board,
            flat_shop,
            [self.current_stage / self.max_rounds]  # Stage normalisé
        ])
        
        return observation
    
    def _get_full_observation(self) -> np.ndarray:
        """
        Renvoie l'observation complète du jeu.
        
        Returns:
            Observation complète
        """
        # Dans une implémentation complète, cela renverrait une représentation
        # complète de l'état du jeu (tous les joueurs, toutes les unités, etc.)
        # Pour cet exemple, on renvoie une observation simplifiée
        return self._get_compressed_observation()
    
    def _calculate_unit_strength(self, unit: Dict[str, Any]) -> float:
        """
        Calcule la force d'une unité.
        
        Args:
            unit: Unité dont on veut calculer la force
            
        Returns:
            Force de l'unité
        """
        # Dans une implémentation complète, cela prendrait en compte les objets,
        # les niveaux d'étoile, les synergies, etc.
        return unit['cost'] * 2  # Force basée simplement sur le coût
    
    def render(self):
        """Affiche l'état actuel du jeu (pour le débogage)."""
        print(f"=== Tour {self.current_stage} ===")
        print(f"Joueur actuel: {self.current_player_idx}")
        
        player = self.players[self.current_player_idx]
        print(f"Santé: {player['health']}, Or: {player['gold']}, Niveau: {player['level']}")
        
        print("Plateau:")
        for i, unit in enumerate(player['board']):
            if unit is not None:
                print(f"  Position {i}: {unit['name']} (coût {unit['cost']})")
        
        print("Banc:")
        for i, unit in enumerate(player['bench']):
            if unit is not None:
                print(f"  Position {i}: {unit['name']} (coût {unit['cost']})")
        
        print("Boutique:")
        for i, champion in enumerate(self.shop):
            if champion is not None:
                print(f"  {i}: {champion['name']} (coût {champion['cost']})")
            else:
                print(f"  {i}: Vide") 