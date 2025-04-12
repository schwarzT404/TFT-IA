  # TFT AI - Reinforcement Learning Agent for Teamfight Tactics

## 📋 Overview
This project implements an AI agent for Teamfight Tactics (TFT) using the MuZero reinforcement learning algorithm. The agent learns to play TFT through a combination of imitation learning and self-play, demonstrating strategic decision-making in a complex, dynamic environment.

## 🎯 Features
- Fast game simulator (under 1 second per game)
- Hybrid learning approach combining imitation and reinforcement learning
- Advanced opponent action prediction
- Dynamic difficulty adjustment
- Comprehensive evaluation metrics

## 🛠️ Technical Stack
- Python 3.8+
- PyTorch
- OpenAI Gym
- NumPy
- TensorBoard (for visualization)

## 📦 Installation

### Prerequisites
```bash
# Clone the repository
git clone https://github.com/yourusername/tft-ai.git
cd tft-ai

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup
```bash
# Install game dependencies
pip install -e .
```

## 🚀 Usage

### Training the Agent
```python
from tft_ai import TFTGame, MuZeroAgent

# Initialize the game environment
env = TFTGame(config)

# Create and train the agent
agent = MuZeroAgent(config)
agent.train(env)
```

### Evaluating the Agent
```python
# Evaluate the trained agent
results = agent.evaluate(num_games=100)
print(f"Win Rate: {results['win_rate']:.2f}%")
print(f"Average Placement: {results['avg_placement']:.2f}")
```

## 📊 Project Structure
```
tft-ai/
├── src/
│   ├── agents/           # AI agent implementations
│   ├── environment/      # Game environment and simulator
│   ├── networks/         # Neural network architectures
│   ├── utils/            # Utility functions
│   └── config.py         # Configuration settings
├── tests/                # Unit and integration tests
├── data/                 # Training data and models
├── notebooks/            # Jupyter notebooks for analysis
└── requirements.txt      # Project dependencies
```

## 🔧 Configuration
The project uses a configuration file (`config.py`) to manage hyperparameters and settings:

```python
config = {
    'training': {
        'num_episodes': 10000,
        'batch_size': 32,
        'learning_rate': 0.001,
    },
    'environment': {
        'num_players': 4,
        'max_rounds': 20,
    },
    'agent': {
        'use_imitation_learning': True,
        'opponent_prediction': True,
    }
}
```

## 📈 Performance Metrics
The agent is evaluated on several metrics:
- Win Rate
- Average Placement
- Economy Efficiency
- Synergy Optimization
- Adaptation Speed

## 🤝 Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References
- [MuZero: Mastering Go, Chess, Shogi and Atari without Rules](https://arxiv.org/abs/1911.08265)
- [Teamfight Tactics Official Documentation](https://teamfighttactics.leagueoflegends.com/)
- [OpenAI Gym Documentation](https://gym.openai.com/docs/)

## 🙏 Acknowledgments
- DeepMind for the MuZero algorithm
- Riot Games for Teamfight Tactics
- The open-source AI community

## 📞 Contact
For questions or suggestions, please open an issue or contact the maintainers.

## 🧠 Technical Synthesis and Implementation Choices

### Algorithm Selection

#### Why MuZero?
- **Model-Free Learning**: MuZero can learn without explicit game rules, crucial for TFT's complex mechanics
- **Planning Capabilities**: Built-in Monte Carlo Tree Search (MCTS) for long-term strategy planning
- **State Representation**: Efficient handling of TFT's large state space through learned representations
- **Adaptability**: Can adjust to meta changes and new game mechanics

#### Hybrid Learning Approach
1. **Imitation Learning Phase**
   - Purpose: Bootstrap initial strategy from human gameplay
   - Implementation: Neural network trained on expert game replays
   - Benefits: Faster initial learning, human-like decision making

2. **Reinforcement Learning Phase**
   - Purpose: Refine and optimize strategies through self-play
   - Implementation: MuZero with modified reward structure
   - Benefits: Discovery of novel strategies, adaptation to meta

### Core Components

#### 1. Game Environment (`TFTGame`)
```python
class TFTGame:
    def __init__(self):
        self.state_compression = True  # Reduce state space complexity
        self.parallel_combat = True    # Speed up simulations
        self.reward_shaping = True     # Provide intermediate rewards
```
- **Justification**: Optimized for fast simulation and training
- **Key Features**: State compression, parallel processing, reward shaping

#### 2. Agent Architecture (`MuZeroAgent`)
```python
class MuZeroAgent:
    def __init__(self):
        self.representation_network = ...  # State encoding
        self.dynamics_network = ...        # State transition prediction
        self.prediction_network = ...      # Policy and value prediction
```
- **Justification**: Modular design for better training and debugging
- **Key Features**: Separate networks for different aspects of learning

#### 3. Opponent Modeling (`OpponentPredictor`)
```python
class OpponentPredictor:
    def __init__(self):
        self.pattern_recognition = True    # Identify opponent strategies
        self.action_prediction = True      # Predict next moves
```
- **Justification**: Address incomplete information challenge
- **Key Features**: Pattern recognition, action prediction

### Core Game Mechanics Modeling

#### 1. Economy Management
```python
class EconomyManager:
    def __init__(self):
        self.interest_thresholds = [10, 20, 30, 40, 50]
        self.game_stage_weights = {
            'early': 0.8,  # High value for economy in early game
            'mid': 0.5,    # Mixed value in mid-game
            'late': 0.3    # Lower value in late game
        }
    
    def calculate_econ_value(self, gold, health, stage, streak_status):
        interest = min(5, gold // 10)
        game_phase = 'early' if stage < 3 else 'mid' if stage < 5 else 'late'
        stage_modifier = self.game_stage_weights[game_phase]
        health_factor = 0.7 if game_phase == 'late' and health < 30 else 1.0
        streak_multiplier = 1.2 if streak_status in ['win3+', 'lose3+'] else 1.0
        return interest * stage_modifier * health_factor * streak_multiplier
```

#### 2. Health Management
```python
class HealthManager:
    def calculate_health_value(self, health, stage, players_alive):
        base_value = health * (1 + stage * 0.1)
        elimination_risk = max(0, 8 - players_alive) * 0.05
        risk_multiplier = 1.8 if health < 20 else 1.4 if health < 40 else 1.0
        return base_value * (1 + elimination_risk) * risk_multiplier
```

#### 3. Streak Management
```python
class StreakManager:
    def calculate_streak_value(self, current_streak, gold, stage):
        base_streak_value = min(3, abs(current_streak)) if abs(current_streak) > 2 else 0
        stage_multiplier = max(1.0, 1.5 - (stage * 0.2))
        gold_factor = min(1.0, 50 / max(10, gold)) if gold > 30 else 1.0
        return base_streak_value * stage_multiplier * gold_factor
```

#### 4. Power Spike Timing
```python
class PowerSpikeAnalyzer:
    def __init__(self):
        self.critical_rounds = {
            'fast_8': [4.1, 4.2],
            'reroll_1cost': [3.2],
            'reroll_2cost': [3.5],
            'reroll_3cost': [4.1],
            'stabilize': [3.2, 4.1]
        }
        self.champion_probabilities = {
            4: {1: 0.55, 2: 0.30, 3: 0.15, 4: 0.00, 5: 0.00},
            5: {1: 0.45, 2: 0.33, 3: 0.20, 4: 0.02, 5: 0.00},
            6: {1: 0.25, 2: 0.40, 3: 0.30, 4: 0.05, 5: 0.00},
            7: {1: 0.19, 2: 0.30, 3: 0.35, 4: 0.15, 5: 0.01},
            8: {1: 0.16, 2: 0.20, 3: 0.35, 4: 0.25, 5: 0.04},
            9: {1: 0.09, 2: 0.15, 3: 0.30, 4: 0.30, 5: 0.16}
        }
```

#### 5. Board Strength Evaluation
```python
class BoardEvaluator:
    def evaluate_board_strength(self, board, items, traits, enemy_boards):
        total_unit_value = sum(unit.star_level * unit.cost for unit in board.units)
        trait_value = self.calculate_trait_value(traits)
        item_value = self.calculate_item_value(items, board.units)
        synergy_value = self.calculate_synergy_value(board.units, items)
        positioning_value = self.evaluate_positioning(board, enemy_boards)
        
        return (total_unit_value * 0.4 + 
                trait_value * 0.25 + 
                item_value * 0.2 + 
                synergy_value * 0.1 + 
                positioning_value * 0.05)
```

#### 6. Integrated Decision System
```python
class TFTDecisionSystem:
    def __init__(self):
        self.econ_manager = EconomyManager()
        self.health_manager = HealthManager()
        self.streak_manager = StreakManager()
        self.spike_analyzer = PowerSpikeAnalyzer()
        self.board_evaluator = BoardEvaluator()
    
    def make_decision(self, game_state):
        # Calculate values for different aspects
        econ_value = self.econ_manager.calculate_econ_value(
            game_state.gold, game_state.health, game_state.stage, game_state.streak)
        health_value = self.health_manager.calculate_health_value(
            game_state.health, game_state.stage, game_state.players_alive)
        streak_value = self.streak_manager.calculate_streak_value(
            game_state.streak, game_state.gold, game_state.stage)
        
        # Evaluate potential decisions
        decisions = {
            'level_up': self.evaluate_level_up(game_state),
            'roll_down': self.evaluate_roll_down(game_state),
            'econ': econ_value,
            'buy_xp': self.evaluate_buy_xp(game_state)
        }
        
        return max(decisions.items(), key=lambda x: x[1])[0]
```

#### 7. Reward Shaping
```python
class TFTRewardShaper:
    def shape_reward(self, game_state, next_state, stage_reward):
        shaped_reward = stage_reward
        
        # Economy management bonus
        if next_state.gold >= 50 and game_state.stage < 3:
            shaped_reward += 0.1
            
        # Health preservation bonus
        if game_state.health - next_state.health < 5 and game_state.stage >= 4:
            shaped_reward += 0.05
            
        # Streak maintenance bonus
        if abs(game_state.streak) < abs(next_state.streak) and abs(next_state.streak) >= 3:
            shaped_reward += 0.08
            
        # Champion upgrade bonus
        if self.count_upgraded_units(next_state) > self.count_upgraded_units(game_state):
            shaped_reward += 0.15
            
        # Trait activation bonus
        trait_diff = self.calculate_trait_improvement(game_state, next_state)
        shaped_reward += trait_diff * 0.1
        
        return shaped_reward
```

These mathematical models form the foundation of our AI's decision-making process, allowing it to:
1. Optimize economy management based on game stage and health
2. Make informed decisions about health preservation
3. Maintain and leverage win/loss streaks effectively
4. Identify optimal power spike moments
5. Evaluate board strength comprehensively
6. Make integrated decisions considering all game aspects
7. Learn effectively through well-shaped rewards

The models are integrated into our MuZero architecture through the representation and dynamics networks, providing a solid foundation for the AI to learn and master TFT's complex mechanics.

## Set 14: Cyber City - Complete Synthesis

### Core Mechanics

#### Hacks System
- **Types**:
  - Augment Hacks: Bonus d'or
  - Shop Hacks: Objets changeants
  - PvE Hacks: Composants améliorés
  - Carousel Hacks: Récompenses boostées
  - Health Hacks: PV gratuits

#### Origins

##### BoomBots
- **Seuils**: 2/4/6
- **Effets**:
  - (2) 150 dégâts magiques
  - (4) 330 dégâts magiques
  - (6) Double missile (200 dégâts chacun)

##### Cypher
- **Seuils**: 3/4/5
- **Mécanique**: Infos par défaites
- **Effets**:
  - (3) 1x Infos, 25% AD & AP
  - (4) 1.5x Infos, 40% AD & AP
  - (5) 2x Infos, 50% AD & AP

##### Exotech
- **Seuils**: 3/5/7/10
- **Mécanique**: Objets uniques
- **Effets**:
  - (3) 30 PV, 2% vitesse d'attaque
  - (5) 110 PV, 3% vitesse d'attaque
  - (7) 200 PV, 7% vitesse d'attaque
  - (10) 500 PV, 40% vitesse d'attaque

##### Démon Urbain
- **Seuils**: 3/5/7/10
- **Mécanique**: Bonus hexagones peints
- **Effets**:
  - (3) +6% PV, 6 Puissance, 6% Dégâts
  - (5) +10% PV, 10 Puissance, 10% Dégâts
  - (7) +15% PV, 15 Puissance, 15% Dégâts
  - (10) +25% PV, 25 Puissance, 25% Dégâts

### Classes

#### Divinicorp
- **Seuils**: 1-7
- **Bonus par champion**:
  - Morgana → AP
  - Rhaast → Armure/Résistance
  - Senna → AD
  - Gragas → PV
  - Vex → Coup critique
  - Renekton → Vitesse d'attaque
  - Emblème → Omnivampirisme

### Champions Clés

#### Renekton
- **Traits**: Seigneur Suprême / Divinicorp / Bastion
- **Compétence**: 
  - Dégâts physiques + frénésie
  - Brûlure et blessure

#### Samira
- **Traits**: Démon urbain / A.M.P.
- **Compétence**:
  - Volée de balles
  - Invincibilité + omnivampirisme

#### Urgot
- **Traits**: Pègre / Dynamo
- **Compétence**:
  - Vitesse d'attaque
  - Exécution cibles faibles

#### Viego
- **Traits**: Soul Killer / Boeuf doré / Technophile
- **Compétence**:
  - Absorption d'âmes
  - Saut + dégâts magiques

### Stratégies

#### Early Game (1-3)
- Focus économie/santé
- Utilisation hacks
- Synergies de base

#### Mid Game (4-5)
- Stabilisation
- Renforcement composition
- Transition

#### Late Game (6+)
- Synergies complètes
- Hacks actifs
- Adaptation

### Compositions

#### BoomBots
- **Core**: 6 BoomBots
- **Supports**: Dégâts magiques
- **Items**: Puissance magique

#### Cypher
- **Core**: 5 Cypher
- **Stratégie**: Gestion défaites
- **Timing**: 3-3, 4-3

#### Exotech
- **Core**: 7 Exotech
- **Focus**: Items uniques
- **Timing**: Late game

### Statistiques

#### Probabilités Champions
```
Niveau 4: 1(55%) 2(30%) 3(15%) 4(0%) 5(0%)
Niveau 5: 1(45%) 2(33%) 3(20%) 4(2%) 5(0%)
Niveau 6: 1(25%) 2(40%) 3(30%) 4(5%) 5(0%)
Niveau 7: 1(19%) 2(30%) 3(35%) 4(15%) 5(1%)
Niveau 8: 1(16%) 2(20%) 3(35%) 4(25%) 5(4%)
Niveau 9: 1(9%) 2(15%) 3(30%) 4(30%) 5(16%)
```

### Points Clés

1. **Hacks**
   - Adaptation stratégique
   - Opportunités
   - Investissement modéré

2. **Power Spikes**
   - Rounds: 3-3, 3-7, 4-3, 4-7
   - Transitions
   - Rerolls

3. **Économie**
   - Intérêt max: 50 gold
   - Streaks
   - Roll downs

4. **Positionnement**
   - Adaptation
   - Hexagones peints
   - Protection carries

## Game Interface Architecture

The AI interfaces with TFT through a specialized system that ensures it only has access to information a human player would have, preserving competitive fairness and enabling realistic learning.

### Data Collection Methods

#### Screen Capture & Computer Vision
```python
class TFTScreenReader:
    def __init__(self):
        self.board_detector = BoardDetectionModel()
        self.champion_recognizer = ChampionRecognitionModel()
        self.item_recognizer = ItemRecognitionModel()
        self.gold_reader = NumericOCR()
        self.health_reader = NumericOCR()
        
    def capture_game_state(self):
        # Capture screen
        screen = self.take_screenshot()
        
        # Extract visible information
        game_state = {
            'board': self.board_detector.detect_board(screen),
            'bench': self.board_detector.detect_bench(screen),
            'shop': self.champion_recognizer.detect_shop(screen),
            'gold': self.gold_reader.read_gold(screen),
            'health': self.health_reader.read_health(screen),
            'round': self.read_round_info(screen),
            'opponents': self.detect_opponents(screen),
            'items': self.item_recognizer.detect_items(screen)
        }
        
        return game_state
```

#### Game Client API (Restricted)
```python
class TFTGameClient:
    def __init__(self):
        self.client_connector = ClientConnector()
        
    def get_visible_state(self):
        # Connect to game client
        raw_game_data = self.client_connector.get_game_data()
        
        # Filter to only visible data
        visible_data = self.filter_to_visible_data(raw_game_data)
        
        return visible_data
```

### Game Interaction System

The AI acts in the game through controlled mouse and keyboard inputs:

```python
class TFTGameController:
    def __init__(self):
        self.mouse_controller = MouseController()
        self.keyboard_controller = KeyboardController()
        self.ui_detector = UIElementDetector()
        
    def execute_action(self, action):
        action_type = action['type']
        
        if action_type == 'buy_champion':
            champion_index = action['index']
            self.buy_champion_from_shop(champion_index)
            
        elif action_type == 'place_champion':
            champion_id = action['champion_id']
            position = action['position']
            self.place_champion_on_board(champion_id, position)
            
        # Other actions: sell, refresh, buy XP, etc.
```

### Observable vs. Hidden Information

The system strictly limits the AI to access only:

**Observable Information:**
- Own board and bench
- Current shop
- Gold, health, level and XP
- Opponent boards during combat
- Visible items
- Current round and type
- Active traits
- Carousel champions
- Visible opponent champions
- Available augments

**Hidden Information (Excluded):**
- Opponent gold
- Opponent benches
- Opponent shops
- Champion pool contents not in shop
- Future PVE rounds
- Next opponents
- Future augment offerings
- Next carousel contents

### Training Data Capture

For imitation learning, expert gameplay is recorded:

```python
class TFTGameRecorder:
    def __init__(self, save_path="data/expert_games/"):
        self.screen_reader = TFTScreenReader()
        self.save_path = save_path
        self.current_game_data = []
        
    def capture_frame(self):
        # Capture current state
        game_state = self.screen_reader.capture_game_state()
        game_state['timestamp'] = time.time()
        self.current_game_data.append(game_state)
```

### Fair Play Monitoring

To ensure ethical AI behavior:

```python
class FairPlayMonitor:
    def __init__(self, agent):
        self.agent = agent
        self.last_screen_time = 0
        self.min_decision_time = 0.5  # seconds
        
    def check_decision_speed(self):
        # Ensure AI doesn't make decisions too quickly
        current_time = time.time()
        elapsed = current_time - self.last_screen_time
        
        if elapsed < self.min_decision_time:
            # Add delay to simulate human reaction time
            time.sleep(self.min_decision_time - elapsed)
```

### Complete Integration Architecture

The full system integrates all components:

```python
class TFTAISystem:
    def __init__(self, config):
        # Game interface
        self.screen_reader = TFTScreenReader()
        self.game_controller = TFTGameController()
        
        # AI agent
        self.agent = TFTSet14MuZeroAgent(config)
        
        # Fair play monitor
        self.fair_play = FairPlayMonitor(self.agent)
        
    def start(self):
        self.running = True
        
        while self.running:
            # Capture current state
            current_state = self.screen_reader.capture_game_state()
            
            # Verify using only legitimate information
            self.fair_play.check_information_access(current_state)
            
            # Make decision if state changed
            if self.state_has_changed(current_state):
                self.fair_play.check_decision_speed()
                action = self.agent.select_action(current_state)
                self.game_controller.execute_action(action)
```

This architecture ensures the AI interacts with TFT using only information available to human players, creating a level playing field and enabling realistic learning. Screen capture with computer vision is the most robust method to guarantee fair information access, as it literally "sees" what a human player would see.

## Optimisations et Améliorations

### 1. Architecture d'IA Améliorée

#### Hierarchical Reinforcement Learning
```python
class HierarchicalTFTAgent:
    def __init__(self):
        # High-level policy (strategic decisions)
        self.meta_controller = MetaController()
        
        # Low-level policies (tactical execution)
        self.controllers = {
            'economy': EconomyController(),
            'combat': CombatController(),
            'positioning': PositioningController(),
            'items': ItemController()
        }
    
    def select_action(self, state):
        # High-level goal selection
        current_goal = self.meta_controller.select_goal(state)
        
        # Low-level controller selection
        active_controller = self.controllers[current_goal.controller_type]
        
        # Tactical execution
        return active_controller.execute(state, current_goal)
```

**Justification**: La décomposition hiérarchique permet de gérer la complexité du jeu en séparant les décisions stratégiques (quand économiser vs dépenser) des décisions tactiques (quel champion acheter spécifiquement). Cela améliore l'apprentissage en créant une abstraction qui réduit l'espace d'états et facilite l'exploration.

#### Transformer-Based State Encoding
```python
class TFTTransformerEncoder:
    def __init__(self):
        self.champion_embedder = ChampionEmbedder()
        self.trait_embedder = TraitEmbedder()
        self.positioning_encoder = PositionalEncoder()
        self.transformer = TransformerEncoder(layers=4, heads=8)
    
    def encode_state(self, game_state):
        # Embed champions with traits
        champion_embeds = [
            self.champion_embedder(champ, pos) 
            for champ, pos in zip(game_state.champions, game_state.positions)
        ]
        
        # Add trait and positional information
        trait_embeds = self.trait_embedder(game_state.active_traits)
        
        # Combine with attention
        return self.transformer(champion_embeds, trait_embeds, game_state.meta_features)
```

**Justification**: Les architectures Transformer sont plus adaptées que les CNN/RNN pour capturer les relations complexes entre les champions, leurs positions, et les synergies. Leur mécanisme d'attention permet de mieux modéliser les interactions à long terme, cruciales pour TFT.

### 2. Optimisations de Performance

#### Simulation Vectorisée
```python
class VectorizedTFTSimulator:
    def __init__(self, num_envs=32):
        self.num_envs = num_envs
        self.states = np.zeros((num_envs, STATE_DIM))
        
    def simulate_batch(self, states, actions):
        # Vectorized computation for multiple simulations
        next_states = self.transition_model(states, actions)
        rewards = self.reward_model(states, actions, next_states)
        
        return next_states, rewards
    
    def transition_model(self, states, actions):
        # Parallelized state transitions
        # Uses numpy/JAX for acceleration
        return self._batch_compute_next_states(states, actions)
```

**Justification**: La simulation vectorisée permet de traiter plusieurs environnements en parallèle, accélérant considérablement l'entraînement par un facteur de 10-100x. Particulièrement utile pour MCTS et les algorithmes d'exploration qui nécessitent de nombreuses simulations.

#### Batch Inference Optimization
```python
class OptimizedInference:
    def __init__(self, model, batch_size=16):
        self.model = model
        self.batch_size = batch_size
        self.state_buffer = []
        self.jit_model = torch.jit.script(model)  # Using TorchScript
    
    def predict(self, state):
        self.state_buffer.append(state)
        
        if len(self.state_buffer) >= self.batch_size:
            # Batch processing
            batched_states = self.prepare_batch(self.state_buffer)
            with torch.no_grad():
                results = self.jit_model(batched_states)
            
            self.state_buffer = []
            return results
```

**Justification**: L'inférence par batch avec TorchScript ou ONNX Runtime réduit considérablement le temps de prédiction, crucial pendant les phases critiques du jeu où une prise de décision rapide est nécessaire.

### 3. Amélioration de l'Apprentissage

#### Curriculum Learning
```python
class TFTCurriculumTrainer:
    def __init__(self, agent):
        self.agent = agent
        self.stages = [
            {
                "name": "economy_basics",
                "difficulty": 0.2,
                "focus": "interest_management",
                "success_threshold": 0.7
            },
            {
                "name": "combat_basics",
                "difficulty": 0.4,
                "focus": "team_composition",
                "success_threshold": 0.65
            },
            # More stages...
        ]
        self.current_stage = 0
    
    def update_curriculum(self, metrics):
        if metrics["success_rate"] > self.stages[self.current_stage]["success_threshold"]:
            self.current_stage = min(self.current_stage + 1, len(self.stages) - 1)
            return True
        return False
    
    def get_environment_settings(self):
        return self.stages[self.current_stage]
```

**Justification**: L'apprentissage par curriculum permet d'aborder progressivement la complexité de TFT, en commençant par des aspects fondamentaux (gestion économique) avant d'introduire des concepts avancés (positionnement contre des comps spécifiques). Cela accélère l'apprentissage et améliore la performance finale.

#### Self-Play avec Banque de Stratégies
```python
class StrategyBankSelfPlay:
    def __init__(self):
        self.strategy_bank = []
        self.min_strategies = 10
        self.max_strategies = 50
        self.current_agent = None
    
    def add_strategy(self, agent, performance_metrics):
        # Add diverse strategies to bank
        strategy = {
            "agent": copy.deepcopy(agent),
            "metrics": performance_metrics,
            "style": self.classify_style(agent)
        }
        
        self.strategy_bank.append(strategy)
        
        # Prune if needed
        if len(self.strategy_bank) > self.max_strategies:
            self.prune_strategies()
    
    def sample_opponents(self, num_opponents=7):
        # Sample diverse opponents
        return self.diversity_sampling(self.strategy_bank, num_opponents)
```

**Justification**: La méthode traditionnelle de self-play souffre souvent d'un manque de diversité stratégique. Une banque de stratégies variées permet à l'agent d'être exposé à différents styles de jeu, augmentant sa robustesse et sa capacité à contrer diverses stratégies.

### 4. Amélioration de l'Interface

#### Vision par Ordinateur Robuste
```python
class RobustTFTVision:
    def __init__(self):
        # Base models
        self.detector = YOLOv5Detector(weights="tft_detector.pt")
        self.ocr = PaddleOCR(use_gpu=True)
        
        # Uncertainty handling
        self.confidence_threshold = 0.85
        self.history = TemporalStateBuffer(max_frames=5)
    
    def process_frame(self, frame):
        # Process with confidence scores
        detections = self.detector(frame)
        
        # Handle uncertain detections
        for detection in detections:
            if detection.confidence < self.confidence_threshold:
                # Use temporal information to improve detection
                improved_detection = self.history.enhance_with_temporal_context(detection)
                detection = improved_detection
                
        # Update history
        self.history.add_frame(detections)
        
        return self.build_game_state(detections)
```

**Justification**: La robustesse de la reconnaissance visuelle est cruciale pour l'interaction avec le jeu. Un système qui combine détection d'objets avancée (YOLOv5) avec des mécanismes de gestion d'incertitude et d'information temporelle réduit considérablement les erreurs d'interprétation.

#### Système d'Action Robuste
```python
class RobustActionSystem:
    def __init__(self):
        self.action_validators = {
            "buy": BuyActionValidator(),
            "place": PlacementValidator(),
            "sell": SellValidator(),
            "reroll": RerollValidator()
        }
        self.feedback_analyzer = ActionFeedbackAnalyzer()
    
    def execute_action(self, action, game_state):
        # Validate action before execution
        if not self.action_validators[action.type].validate(action, game_state):
            return self.find_alternative_action(action, game_state)
        
        # Execute with monitoring
        result = self.perform_action_with_retry(action)
        
        # Analyze feedback to confirm success
        action_success = self.feedback_analyzer.check_success(game_state, result)
        
        if not action_success:
            self.handle_failed_action(action, game_state)
```

**Justification**: Un système d'action robuste qui valide les actions, détecte les échecs et implémente des mécanismes de récupération est essentiel pour un agent qui interagit avec un environnement réel sujet aux incertitudes.

### 5. Métriques et Évaluation Avancées

#### Évaluation Multi-dimensionnelle
```python
class TFTPerformanceEvaluator:
    def __init__(self):
        self.metrics = {
            "placement": PlacementTracker(),
            "economy": EconomyEfficiencyTracker(),
            "combat": CombatEffectivenessTracker(),
            "adaptation": AdaptationSpeedTracker(),
            "optimization": ItemOptimizationTracker()
        }
    
    def evaluate_game(self, game_data):
        results = {}
        for name, tracker in self.metrics.items():
            results[name] = tracker.compute(game_data)
            
        # Compute composite score
        results["overall"] = self.compute_composite_score(results)
        
        return results
    
    def compute_composite_score(self, metrics):
        # Weighted sum based on correlations with placement
        weights = {
            "economy": 0.3,
            "combat": 0.35,
            "adaptation": 0.2,
            "optimization": 0.15
        }
        
        return sum(metrics[k] * v for k, v in weights.items())
```

**Justification**: Une évaluation qui va au-delà du simple placement final permet de diagnostiquer précisément les forces et faiblesses de l'agent. Cela guide mieux l'amélioration en identifiant les aspects spécifiques à optimiser.

#### Analyse de Style de Jeu
```python
class PlaystyleAnalyzer:
    def __init__(self):
        self.style_dimensions = {
            "aggression": AggressionAnalyzer(),   # Early rolls vs economy
            "flexibility": FlexibilityAnalyzer(), # Adaptation to items/units
            "consistency": ConsistencyAnalyzer(), # Variance in strategy
            "risk": RiskAnalyzer()                # High-roll vs safe play
        }
    
    def analyze_agent(self, game_histories):
        style_profile = {}
        
        for dim_name, analyzer in self.style_dimensions.items():
            style_profile[dim_name] = analyzer.compute(game_histories)
            
        # Generate style classification
        style_profile["archetype"] = self.classify_archetype(style_profile)
        
        return style_profile
```

**Justification**: Comprendre le style de jeu émergent de l'IA permet de s'assurer qu'elle développe diverses stratégies plutôt que d'exploiter une approche unique. Cela favorise également une IA plus intéressante et imprévisible face à des adversaires humains.

## Recent Balance Updates (Patch 14.1 - April 7th)

The recent mid-patch update introduces critical balance changes that directly impact our AI's evaluation models and strategic decision-making. Our systems have been updated to account for these changes:

### Trait Adjustments
```python
class TraitBalanceUpdater:
    def apply_patch_14_1_changes(self):
        # Anima Squad nerf
        self.traits["Anima Squad"].armor_mr_values = [10, 25, 35]  # Previously [10, 30, 45]
        self.traits["Anima Squad"].damage_amp = [0.05, 0.10, 0.15]  # Previously [0.05, 0.12, 0.20]
        
        # Update trait evaluation weights
        self.trait_power_ratings["Anima Squad"] *= 0.85  # Reduced power rating by 15%
```

### Unit Adjustments
```python
class UnitBalanceManager:
    def update_unit_stats(self):
        # Rengar nerfs
        self.units["Rengar"].base_ad = 63  # Previously 65
        self.units["Rengar"].ability_ad_scaling = 2.4  # Previously 2.5
        
        # Annie's Tibbers adjustments
        self.units["Annie"].tibbers_base_hp = [10, 100, 4000]  # Previously [300, 600, 4000]
        self.units["Annie"].tibbers_ap_scaling = [5.0, 7.0, 10.0]  # Previously 4.0 at all stars
        self.units["Annie"].tibbers_ad = [150, 225, 900]  # Previously [200, 300, 900]
        self.units["Annie"].tibbers_armor_mr = 60  # Previously 75
        
        # Kobuko nerfs
        self.units["Kobuko"].base_hp = 900  # Previously 1000
        self.units["Kobuko"].mana = [120, 240]  # Previously [100, 220]
        self.units["Kobuko"].armor_mr = 50  # Previously 60
        self.units["Kobuko"].shield_duration = 4  # Previously 8 seconds
        self.units["Kobuko"].damage_reduction = 0.33  # Previously 0.60
        self.units["Kobuko"].enemy_hp_percent = [0.03, 0.05, 1.0]  # Previously [0.09, 0.15, 1.0]
        
        # Samira nerfs
        self.units["Samira"].base_ad = 60  # Previously 65
        self.units["Samira"].armor_mr = 40  # Previously 50
        self.units["Samira"].ability_omnivamp = 0.12  # Previously 0.20
        self.units["Samira"].armor_shred = 3  # Previously 4
```

### Strategy Adaptation
```python
class StrategicAdjuster:
    def adjust_post_patch(self):
        # Reduce priority of Anima Squad compositions
        self.comp_priorities["Anima Squad"] *= 0.9
        
        # Adjust item priorities for Annie
        self.champion_item_priorities["Annie"]["AP"] += 20  # Increased priority for AP items
        
        # Adjust Kobuko's role in compositions
        self.champion_roles["Kobuko"] = "AP_BRUISER"  # Previously "TANK"
        
        # Adjust Samira's carry potential
        self.carry_potential["Samira"] *= 0.85  # Reduced by 15%
        
        # Adjust Rengar's priority in Executioner comps
        if "Executioner" in self.active_traits:
            self.unit_priorities["Rengar"] *= 0.95
```

### Augment Changes
```python
class AugmentEvaluator:
    def update_augment_timings(self):
        # 10,000 IQ availability change
        self.augment_availability["10,000 IQ"] = ["3-2"]  # Previously available earlier
        
        # Update value assessment for affected augments
        self.augment_value_by_stage["10,000 IQ"]["3-2"] *= 1.15  # Increase relative value at new stage
```

These adjustments align our AI's decision-making with the current meta following Patch 14.1. The major implications for strategy include:

1. **Anima Squad Compositions**: Reduced priority due to significant nerfs to defensive stats and damage amplification
2. **Annie Builds**: Increased focus on AP scaling rather than relying on Tibbers' base power
3. **Tank Evaluation**: Kobuko no longer considered a primary tank, repositioned as an AP bruiser
4. **Carry Assessment**: Samira's evaluation as a primary carry reduced, requiring more support to function effectively
5. **Economic Strategy**: Adjusted timing for 10,000 IQ augment selection, no longer viable for early game loss-streak strategies

The AI now favors more balanced team compositions until further meta developments, with special attention to effective counter-strategies against still-powerful units like Rengar in the early game.

## Game Simulation Architecture

La simulation précise de TFT constitue l'une des fondations critiques de notre système d'IA. Notre approche de simulation combine plusieurs méthodes pour maximiser la fidélité tout en optimisant l'efficacité d'entraînement.

### Approche Progressive Multi-niveaux

Notre architecture de simulation suit une approche progressive en quatre phases, chacune augmentant la fidélité et la performance:

```python
class TFTSimulationManager:
    def __init__(self, config):
        self.simulation_level = config.get("simulation_level", "advanced")
        self.vectorized = config.get("vectorized", True)
        self.num_envs = config.get("num_environments", 64)
        self.api_key = config.get("riot_api_key", None)
        
        # Initialiser le simulateur approprié
        if self.simulation_level == "basic":
            self.simulator = BasicTFTSimulator()
        elif self.simulation_level == "standard":
            self.simulator = StandardTFTSimulator()
        elif self.simulation_level == "advanced":
            self.simulator = HybridTFTSimulator(api_key=self.api_key)
        elif self.simulation_level == "expert":
            self.simulator = LearnedTFTSimulator()
        
        # Vectorisation pour entraînement parallèle
        if self.vectorized:
            self.simulator = VectorizedWrapper(self.simulator, self.num_envs)
```

#### Phase 1: Simulateur Fondamental

Notre simulateur de base implémente les mécaniques essentielles du jeu:

```python
class BasicTFTSimulator:
    def __init__(self):
        self.champion_pool = ChampionPool()
        self.item_pool = ItemPool()
        self.players = [Player(i) for i in range(8)]
        self.round_manager = RoundManager()
        self.current_stage = 1
        self.current_round = 1
    
    def simulate_round(self, player_actions):
        # Exécuter les actions des joueurs
        for player_id, actions in enumerate(player_actions):
            self._process_player_actions(self.players[player_id], actions)
        
        # Déterminer les matchups
        matchups = self._generate_matchups()
        
        # Simuler les combats
        results = []
        for p1_idx, p2_idx in matchups:
            winner, damage = self._simplified_combat(self.players[p1_idx], self.players[p2_idx])
            if winner == 1:
                self.players[p2_idx].health -= damage
                results.append((p1_idx, p2_idx, damage))
            else:
                self.players[p1_idx].health -= damage
                results.append((p2_idx, p1_idx, damage))
        
        # Avancer au round suivant
        self._advance_round()
        
        # Retourner le nouvel état et les résultats
        return self._get_state(), results
    
    def _simplified_combat(self, player1, player2):
        # Modèle de combat simplifié basé sur la force brute des boards
        power1 = self._calculate_board_power(player1.board)
        power2 = self._calculate_board_power(player2.board)
        
        # Ajouter un élément aléatoire
        power1 *= random.uniform(0.9, 1.1)
        power2 *= random.uniform(0.9, 1.1)
        
        # Déterminer le vainqueur et les dégâts
        if power1 > power2:
            damage = max(1, int(math.sqrt(power1 - power2) / 10) + self.current_stage)
            return 1, damage
        else:
            damage = max(1, int(math.sqrt(power2 - power1) / 10) + self.current_stage)
            return 2, damage
    
    def _calculate_board_power(self, board):
        # Calcul simplifié de la puissance d'un board
        total_power = 0
        for unit in board.units:
            # Puissance de base: coût × niveau d'étoile²
            unit_power = unit.cost * (unit.star_level ** 2)
            
            # Bonus d'objets: +20% par objet
            item_multiplier = 1 + (0.2 * len(unit.items))
            unit_power *= item_multiplier
            
            total_power += unit_power
        
        # Bonus de traits (simplifié)
        trait_multiplier = 1
        for trait, count in board.active_traits.items():
            # Bonus de +10% par breakpoint de trait actif
            thresholds = self.traits_config[trait]["thresholds"]
            for threshold in thresholds:
                if count >= threshold:
                    trait_multiplier += 0.1
        
        return total_power * trait_multiplier
```

#### Phase 2: Simulateur Hybride avec Données Réelles

Notre simulateur de niveau intermédiaire intègre des données réelles pour améliorer sa précision:

```python
class HybridTFTSimulator(BasicTFTSimulator):
    def __init__(self, api_key=None):
        super().__init__()
        self.data_models = {}
        
        if api_key:
            self.data_collector = RiotAPIDataCollector(api_key)
            self._initialize_data_models()
    
    def _initialize_data_models(self):
        # Collecter et traiter les données
        match_data = self.data_collector.collect_high_elo_matches(500)
        
        # Entraîner des modèles pour les aspects complexes
        self.data_models["combat"] = self._train_combat_model(match_data)
        self.data_models["placement"] = self._train_placement_model(match_data)
        self.data_models["itemization"] = self._train_itemization_model(match_data)
        self.data_models["transitions"] = self._train_transition_model(match_data)
    
    def _train_combat_model(self, match_data):
        # Extraire les caractéristiques de combat et résultats
        X_combat, y_combat = [], []
        
        for match in match_data:
            for round_data in match["rounds"]:
                if round_data["type"] == "PVP":
                    # Extraire les caractéristiques des deux boards
                    board1 = self._extract_board_features(round_data["player1_board"])
                    board2 = self._extract_board_features(round_data["player2_board"])
                    
                    # Résultat: [vainqueur, dégâts]
                    result = [
                        1 if round_data["winner"] == round_data["player1_id"] else 0,
                        round_data["damage"]
                    ]
                    
                    # Ajouter dans les deux sens pour l'équilibre
                    X_combat.append(np.concatenate([board1, board2]))
                    y_combat.append(result)
                    
                    X_combat.append(np.concatenate([board2, board1]))
                    y_combat.append([1 - result[0], result[1]])
        
        # Entraîner un modèle (ex: RandomForest ou Neural Network)
        model = RandomForestClassifier(n_estimators=100)
        model.fit(np.array(X_combat), np.array(y_combat))
        
        return model
    
    def _extract_board_features(self, board):
        # Convertir un board en vecteur de caractéristiques
        features = []
        
        # Caractéristiques globales du board
        total_cost = sum(unit["cost"] for unit in board["units"])
        total_stars = sum(unit["star_level"] for unit in board["units"])
        average_cost = total_cost / len(board["units"]) if board["units"] else 0
        
        features.extend([total_cost, total_stars, average_cost, len(board["units"])])
        
        # Traits actifs (one-hot encoding)
        trait_vector = np.zeros(len(self.traits_config))
        for trait, count in board["traits"].items():
            trait_idx = self.trait_to_idx[trait]
            trait_level = 0
            for threshold in sorted(self.traits_config[trait]["thresholds"]):
                if count >= threshold:
                    trait_level += 1
            trait_vector[trait_idx] = trait_level
        
        features.extend(trait_vector)
        
        # Unités (agrégées)
        unit_cost_distribution = np.zeros(5)  # 1-5 cost
        for unit in board["units"]:
            unit_cost_distribution[unit["cost"] - 1] += 1
        
        features.extend(unit_cost_distribution)
        
        return np.array(features)
    
    def simulate_combat(self, player1, player2):
        # Utiliser le modèle entraîné si disponible
        if "combat" in self.data_models:
            board1_features = self._extract_board_features(player1.board)
            board2_features = self._extract_board_features(player2.board)
            
            features = np.concatenate([board1_features, board2_features])
            prediction = self.data_models["combat"].predict([features])[0]
            
            winner = 1 if prediction[0] > 0.5 else 2
            damage = int(prediction[1])
            
            return winner, damage
        else:
            # Fallback sur le combat simplifié
            return super()._simplified_combat(player1, player2)
```

#### Phase 3: Simulateur Avancé avec Apprentissage

Notre simulateur le plus avancé utilise des modèles d'apprentissage pour tous les aspects complexes:

```python
class LearnedTFTSimulator:
    def __init__(self):
        # Modèles pour chaque composant complexe
        self.models = {
            "combat": self._build_combat_model(),
            "state_transition": self._build_transition_model(),
            "player_behavior": self._build_behavior_model(),
            "reward": self._build_reward_model()
        }
        
        # Composants de base du simulateur
        self.champion_pool = ChampionPool()
        self.item_pool = ItemPool()
        self.current_stage = 1
        self.current_round = 1
    
    def _build_combat_model(self):
        # Créer un modèle de réseau neuronal pour le combat
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(STATE_COMBAT_DIM,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(2)  # [probabilité de victoire, dégâts]
        ])
        
        model.compile(
            optimizer='adam',
            loss=['binary_crossentropy', 'mse'],
            metrics=['accuracy', 'mae']
        )
        
        return model
    
    def simulate_full_game(self, agent_strategies):
        # Initialiser un nouvel état de jeu
        state = self._initialize_game_state()
        
        while not self._is_game_over(state):
            # Chaque joueur prend des décisions
            actions = {}
            for player_id, strategy in enumerate(agent_strategies):
                if state["players"][player_id]["health"] > 0:
                    player_state = self._extract_player_state(state, player_id)
                    actions[player_id] = strategy.decide_action(player_state)
            
            # Appliquer les actions au jeu
            next_state = self._apply_actions(state, actions)
            
            # Simuler les combats
            combat_results = self._simulate_combats(next_state)
            
            # Mettre à jour l'état avec les résultats
            state = self._update_state_after_combat(next_state, combat_results)
            
            # Avancer au round suivant
            state = self._advance_round(state)
        
        # Retourner les résultats finaux
        return self._get_final_results(state)
    
    def _simulate_combats(self, state):
        # Générer les matchups
        matchups = self._generate_matchups(state)
        results = {}
        
        for p1_id, p2_id in matchups:
            board1 = state["players"][p1_id]["board"]
            board2 = state["players"][p2_id]["board"]
            
            # Encoder les boards pour le modèle
            features = self._encode_combat_features(board1, board2, state)
            
            # Prédire le résultat avec le modèle
            prediction = self.models["combat"].predict([features])[0]
            
            # Interpréter la prédiction
            winner_id = p1_id if prediction[0] > 0.5 else p2_id
            loser_id = p2_id if winner_id == p1_id else p1_id
            damage = max(1, int(prediction[1]))
            
            results[(p1_id, p2_id)] = {
                "winner": winner_id,
                "loser": loser_id,
                "damage": damage
            }
        
        return results
```

#### Phase 4: Vectorisation pour Entraînement Parallèle

Pour accélérer l'entraînement, nous utilisons une couche de vectorisation:

```python
class VectorizedWrapper:
    def __init__(self, base_simulator, num_envs=64):
        self.base_simulator = base_simulator
        self.num_envs = num_envs
        
        # Créer des copies indépendantes du simulateur
        self.simulators = [copy.deepcopy(base_simulator) for _ in range(num_envs)]
        
        # État actuel de tous les environnements
        self.states = [sim._initialize_game_state() for sim in self.simulators]
        self.dones = np.zeros(num_envs, dtype=bool)
    
    def reset(self):
        # Réinitialiser tous les environnements
        for i, sim in enumerate(self.simulators):
            self.states[i] = sim._initialize_game_state()
        
        self.dones = np.zeros(self.num_envs, dtype=bool)
        return self._vectorize_states(self.states)
    
    def step(self, vectorized_actions):
        # Convertir actions vectorisées en actions par environnement
        actions_per_env = self._devectorize_actions(vectorized_actions)
        
        # Exécuter les actions sur chaque simulateur non terminé
        next_states = []
        rewards = []
        new_dones = []
        
        for i, (sim, state, actions, done) in enumerate(zip(
                self.simulators, self.states, actions_per_env, self.dones)):
            
            if done:
                # Si l'environnement est déjà terminé, pas de changement
                next_states.append(state)
                rewards.append(np.zeros(8))  # 8 joueurs
                new_dones.append(True)
            else:
                # Faire avancer la simulation
                next_state = sim._apply_actions(state, actions)
                combat_results = sim._simulate_combats(next_state)
                next_state = sim._update_state_after_combat(next_state, combat_results)
                next_state = sim._advance_round(next_state)
                
                # Calculer les récompenses
                reward = self._calculate_rewards(state, next_state)
                
                # Vérifier si l'épisode est terminé
                done = sim._is_game_over(next_state)
                
                next_states.append(next_state)
                rewards.append(reward)
                new_dones.append(done)
        
        # Mettre à jour l'état interne
        self.states = next_states
        self.dones = np.array(new_dones)
        
        # Vectoriser pour le retour
        return (
            self._vectorize_states(next_states),
            np.array(rewards),
            np.array(new_dones)
        )
    
    def _vectorize_states(self, states):
        # Convertir une liste d'états en une matrice numpy
        vectorized = np.zeros((self.num_envs, STATE_DIM))
        
        for i, state in enumerate(states):
            vectorized[i] = self._state_to_vector(state)
        
        return vectorized
    
    def _state_to_vector(self, state):
        # Convertir un état en vecteur
        vector = []
        
        # Informations globales
        vector.extend([state["current_stage"], state["current_round"]])
        
        # Pour chaque joueur
        for player in state["players"]:
            # Infos de base
            vector.extend([
                player["health"], 
                player["level"],
                player["xp"],
                player["gold"],
                player["win_streak"],
                player["loss_streak"]
            ])
            
            # Board simplifié (unités par coût, niveau d'étoile)
            board_summary = np.zeros(15)  # 5 coûts × 3 niveaux d'étoile
            for unit in player["board"]["units"]:
                idx = (unit["cost"] - 1) * 3 + (unit["star_level"] - 1)
                board_summary[idx] += 1
            
            vector.extend(board_summary)
            
            # Traits actifs (principaux seulement)
            trait_summary = np.zeros(10)  # Top 10 traits
            for i, (trait, count) in enumerate(sorted(
                    player["board"]["traits"].items(), 
                    key=lambda x: x[1], 
                    reverse=True)[:10]):
                trait_summary[i] = count
            
            vector.extend(trait_summary)
        
        return np.array(vector)
```

### Architecture Complète du Système de Simulation

Notre système de simulation complet intègre toutes ces approches dans une architecture unifiée:

```python
class TFTSimulationSystem:
    def __init__(self, config):
        self.config = config
        self.simulation_manager = TFTSimulationManager(config)
        self.data_manager = TFTDataManager(config)
        self.validation_system = TFTValidationSystem()
        
        # Phase actuelle de développement
        self.current_phase = config.get("development_phase", 1)
        
        # Initialiser les composants en fonction de la phase
        self._initialize_phase_components()
    
    def _initialize_phase_components(self):
        if self.current_phase >= 1:
            # Phase 1: Simulateur de base
            self.basic_simulator = BasicTFTSimulator()
        
        if self.current_phase >= 2:
            # Phase 2: Collecte et intégration de données
            api_key = self.config.get("riot_api_key")
            if api_key:
                self.data_collector = RiotAPIDataCollector(api_key)
                match_data = self.data_collector.collect_matches()
                self.data_manager.store_match_data(match_data)
            
            # Créer le simulateur hybride
            self.hybrid_simulator = HybridTFTSimulator(self.data_manager)
        
        if self.current_phase >= 3:
            # Phase 3: Modèles d'apprentissage
            self.model_trainer = TFTModelTrainer(self.data_manager)
            self.learned_simulator = LearnedTFTSimulator(self.model_trainer.trained_models)
            
            # Ajouter vectorisation pour entraînement parallèle
            num_envs = self.config.get("num_environments", 64)
            self.vectorized_simulator = VectorizedWrapper(self.learned_simulator, num_envs)
        
        if self.current_phase >= 4:
            # Phase 4: Validation avec jeu réel
            self.validation_system.initialize(self.config)
    
    def get_appropriate_simulator(self, purpose="training"):
        """Retourne le simulateur le plus approprié selon l'usage et la phase."""
        if purpose == "training" and self.current_phase >= 3:
            return self.vectorized_simulator
        elif purpose == "evaluation" and self.current_phase >= 3:
            return self.learned_simulator
        elif self.current_phase >= 2:
            return self.hybrid_simulator
        else:
            return self.basic_simulator
    
    def validate_against_real_game(self, agent, num_games=20):
        """Valide la performance de l'agent sur le jeu réel."""
        if self.current_phase >= 4:
            return self.validation_system.validate_agent(agent, num_games)
        else:
            print("Validation avec jeu réel disponible uniquement en phase 4")
            return None
```

### Avantages de Notre Approche de Simulation

1. **Progression adaptative**: Evolution du simulateur en parallèle avec la sophistication de l'IA
2. **Efficacité d'entraînement**: Vectorisation pour accélérer l'apprentissage
3. **Intégration de données réelles**: Calibration avec des parties de haut niveau
4. **Équilibre fidélité/performance**: Modèles d'apprentissage pour les aspects complexes
5. **Validation rigoureuse**: Comparaison avec le jeu réel pour assurer la pertinence

Cette architecture de simulation progressive nous permet de développer et d'améliorer notre IA TFT de manière itérative, en adaptant la fidélité de la simulation aux besoins de chaque étape du développement.

## Détails d'Implémentation du Simulateur

Certains aspects de la simulation nécessitent une attention particulière pour garantir la fidélité et les performances du système.

### Système de Combat Avancé

Le système de combat est l'élément le plus complexe à simuler en raison des nombreuses interactions entre champions, objets et traits:

```python
class AdvancedCombatSystem:
    def __init__(self, set_version="14"):
        self.ability_resolver = AbilityResolver(set_version)
        self.damage_calculator = DamageCalculator()
        self.targeting_system = TargetingSystem()
        self.hex_grid = HexGrid(rows=4, cols=7)
        self.trait_manager = TraitManager(set_version)
        self.item_effect_manager = ItemEffectManager(set_version)
    
    def simulate_combat(self, board1, board2):
        # Copier les boards pour préserver l'état original
        combat_board1 = self._prepare_board_for_combat(copy.deepcopy(board1))
        combat_board2 = self._prepare_board_for_combat(copy.deepcopy(board2))
        
        # Positionner les unités sur la grille hexagonale
        self._position_units(combat_board1, combat_board2)
        
        # Appliquer les effets de traits pré-combat
        self._apply_trait_effects(combat_board1, pre_combat=True)
        self._apply_trait_effects(combat_board2, pre_combat=True)
        
        # Initialiser les états des unités pour le combat
        self._initialize_combat_states(combat_board1, combat_board2)
        
        # Boucle principale de combat
        time_elapsed = 0
        max_combat_time = 30  # secondes
        combat_log = []
        
        while time_elapsed < max_combat_time:
            # Déterminer la prochaine unité à agir
            next_unit, next_action_time = self._get_next_action(combat_board1, combat_board2)
            if next_unit is None:
                break  # Combat terminé si plus d'unités actives
                
            # Avancer le temps jusqu'à la prochaine action
            time_elapsed = next_action_time
            
            # Exécuter l'action de l'unité
            if next_unit.can_cast_ability():
                # Lancer une compétence
                targets = self.targeting_system.get_ability_targets(next_unit, combat_board1, combat_board2)
                effects = self.ability_resolver.resolve_ability(next_unit, targets)
                self._apply_ability_effects(effects, combat_board1, combat_board2)
                combat_log.append(self._log_ability_cast(next_unit, targets, effects, time_elapsed))
            else:
                # Effectuer une attaque basique
                target = self.targeting_system.get_attack_target(next_unit, combat_board1, combat_board2)
                if target:
                    damage = self.damage_calculator.calculate_attack_damage(next_unit, target)
                    self._apply_damage(target, damage, source=next_unit, is_ability=False)
                    combat_log.append(self._log_attack(next_unit, target, damage, time_elapsed))
            
            # Mettre à jour l'état de l'unité
            self._update_unit_state(next_unit, time_elapsed)
            
            # Vérifier la fin du combat
            if self._is_combat_over(combat_board1, combat_board2):
                break
        
        # Déterminer le vainqueur et calculer les dégâts
        result = self._calculate_combat_result(combat_board1, combat_board2, time_elapsed)
        result["combat_log"] = combat_log
        
        return result
    
    def _prepare_board_for_combat(self, board):
        # Appliquer les bonus de traits
        active_traits = self.trait_manager.calculate_active_traits(board)
        board.active_traits = active_traits
        
        # Appliquer les effets d'objets
        for unit in board.units:
            for item in unit.items:
                self.item_effect_manager.apply_passive_effects(unit, item)
        
        return board
    
    def _position_units(self, board1, board2):
        # Positionner le board1 à gauche
        for unit in board1.units:
            position = unit.position
            self.hex_grid.place_unit(unit, position)
        
        # Positionner le board2 à droite (en miroir)
        for unit in board2.units:
            # Inverser la position X pour le miroir
            mirrored_position = (6 - unit.position[0], unit.position[1])
            self.hex_grid.place_unit(unit, mirrored_position)
    
    def _initialize_combat_states(self, board1, board2):
        all_units = board1.units + board2.units
        
        for unit in all_units:
            # Initialiser le mana
            unit.current_mana = unit.starting_mana
            
            # Calculer le délai de la première action
            attack_speed = unit.get_stat("attack_speed")
            unit.next_action_time = 1.0 / attack_speed  # Délai avant première attaque
            
            # Initialiser d'autres états de combat
            unit.target = None
            unit.has_acted = False
    
    def _get_next_action(self, board1, board2):
        all_units = [u for u in board1.units + board2.units if u.is_alive()]
        
        if not all_units:
            return None, 0
        
        # Trouver l'unité avec la prochaine action
        next_unit = min(all_units, key=lambda u: u.next_action_time)
        return next_unit, next_unit.next_action_time
    
    def _update_unit_state(self, unit, current_time):
        # Mettre à jour le délai avant prochaine action
        attack_speed = unit.get_stat("attack_speed")
        unit.next_action_time = current_time + (1.0 / attack_speed)
        
        # Mettre à jour le mana si l'unité a attaqué
        if not unit.can_cast_ability():
            unit.current_mana += unit.mana_per_attack
        
        # Appliquer les effets actifs (buffs/debuffs)
        self._process_active_effects(unit, current_time)
    
    def _is_combat_over(self, board1, board2):
        # Vérifier si l'un des boards n'a plus d'unités vivantes
        living_units1 = [u for u in board1.units if u.is_alive()]
        living_units2 = [u for u in board2.units if u.is_alive()]
        
        return len(living_units1) == 0 or len(living_units2) == 0
    
    def _calculate_combat_result(self, board1, board2, combat_time):
        living_units1 = [u for u in board1.units if u.is_alive()]
        living_units2 = [u for u in board2.units if u.is_alive()]
        
        if len(living_units1) > 0 and len(living_units2) == 0:
            # Board1 gagne
            remaining_health = sum(u.current_health / u.max_health for u in living_units1)
            remaining_stars = sum(u.star_level for u in living_units1)
            damage = max(2, int(remaining_stars + board1.player_level / 2))
            
            return {
                "winner": 1,
                "loser": 2,
                "damage": damage,
                "remaining_units": len(living_units1),
                "combat_time": combat_time
            }
        elif len(living_units2) > 0 and len(living_units1) == 0:
            # Board2 gagne
            remaining_health = sum(u.current_health / u.max_health for u in living_units2)
            remaining_stars = sum(u.star_level for u in living_units2)
            damage = max(2, int(remaining_stars + board2.player_level / 2))
            
            return {
                "winner": 2,
                "loser": 1,
                "damage": damage,
                "remaining_units": len(living_units2),
                "combat_time": combat_time
            }
        else:
            # Match nul ou timeout
            return {
                "winner": 0,
                "loser": 0,
                "damage": 0,
                "remaining_units1": len(living_units1),
                "remaining_units2": len(living_units2),
                "combat_time": combat_time
            }
```

### Intégration des Mécaniques du Set 14 (Cyber City)

La simulation des hacks du Set 14 nécessite un système spécifique:

```python
class Set14HackSystem:
    def __init__(self):
        self.available_hacks = {
            "augment": {
                "effects": {
                    "gold_bonus": 5,
                    "augment_tier_boost": 1
                },
                "probability": 0.2
            },
            "shop": {
                "effects": {
                    "item_transform": True,  # À la Pandora
                    "reroll_discount": 1
                },
                "probability": 0.2
            },
            "pve": {
                "effects": {
                    "component_quality": 1.5,
                    "drop_rate_increase": 0.25
                },
                "probability": 0.2
            },
            "carousel": {
                "effects": {
                    "item_tier_boost": 1,
                    "champion_tier_boost": 1
                },
                "probability": 0.2
            },
            "health": {
                "effects": {
                    "health_bonus": 15,
                    "shield_percent": 0.1
                },
                "probability": 0.2
            }
        }
        
        # Hacks actifs dans la partie actuelle
        self.active_hacks = []
        
        # Rounds où les hacks peuvent apparaître avec certitude
        self.guaranteed_hack_rounds = [(2, 1), (3, 2), (4, 2)]  # (stage, round)
    
    def check_hack_opportunity(self, stage, round_num):
        # Vérifier si ce round est garanti pour un hack
        if (stage, round_num) in self.guaranteed_hack_rounds:
            return True
        
        # Chance aléatoire pour les autres rounds (15%)
        return random.random() < 0.15
    
    def generate_hack_options(self, num_options=2):
        # Générer des options de hacks basées sur les probabilités
        hack_types = list(self.available_hacks.keys())
        weights = [self.available_hacks[h]["probability"] for h in hack_types]
        
        # Ajuster les poids en fonction des hacks déjà actifs
        for hack in self.active_hacks:
            if hack["type"] in hack_types:
                idx = hack_types.index(hack["type"])
                weights[idx] *= 0.3  # Réduire la probabilité des types déjà actifs
        
        # Sélectionner les options
        options = []
        for _ in range(num_options):
            if not weights or sum(weights) == 0:
                break
                
            selected_idx = random.choices(range(len(hack_types)), weights=weights)[0]
            selected_type = hack_types[selected_idx]
            
            options.append({
                "type": selected_type,
                "effects": self.available_hacks[selected_type]["effects"],
                "description": self._generate_description(selected_type)
            })
            
            # Retirer le type sélectionné pour éviter les doublons
            weights.pop(selected_idx)
            hack_types.pop(selected_idx)
        
        return options
    
    def apply_hack(self, hack_type, game_state):
        # Appliquer les effets du hack au jeu
        effects = self.available_hacks[hack_type]["effects"]
        
        # Créer une instance du hack actif
        active_hack = {
            "type": hack_type,
            "effects": effects,
            "applied_time": game_state["current_stage"] * 10 + game_state["current_round"],
            "duration": "permanent"  # La plupart des hacks sont permanents
        }
        
        # Ajouter aux hacks actifs
        self.active_hacks.append(active_hack)
        
        # Appliquer les effets immédiats
        if hack_type == "health":
            for player in game_state["players"]:
                player["health"] = min(100, player["health"] + effects["health_bonus"])
        
        elif hack_type == "augment" and "current_augment_options" in game_state:
            # Améliorer les options d'augment
            for option in game_state["current_augment_options"]:
                option["tier"] = min(3, option["tier"] + effects["augment_tier_boost"])
        
        return game_state
    
    def _generate_description(self, hack_type):
        """Génère une description du hack pour l'interface utilisateur"""
        if hack_type == "augment":
            return f"Augment Hack: +{self.available_hacks[hack_type]['effects']['gold_bonus']} or et amélioration du niveau des prochains augments."
        elif hack_type == "shop":
            return "Shop Hack: Les objets changent de forme à chaque tour, et -1 coût de reroll."
        elif hack_type == "pve":
            return "PvE Hack: Amélioration des composants aux rounds PvE et augmentation du taux de drop."
        elif hack_type == "carousel":
            return "Carousel Hack: Amélioration du niveau des champions et objets au carousel."
        elif hack_type == "health":
            return f"Health Hack: +{self.available_hacks[hack_type]['effects']['health_bonus']} PV pour tous les joueurs."
        else:
            return "Hack mystérieux"
```

### Collecte et Prétraitement des Données

La qualité des données est cruciale pour le simulateur hybride:

```python
class RiotAPIDataCollector:
    def __init__(self, api_key, region="euw1"):
        self.api_key = api_key
        self.region = region
        self.base_url = f"https://{region}.api.riotgames.com/tft"
        self.headers = {
            "X-Riot-Token": api_key
        }
        self.rate_limiter = RateLimiter(100, 120)  # 100 requêtes par 120 secondes
    
    def collect_high_elo_matches(self, num_matches=500, min_rank="DIAMOND"):
        """Collecte des parties de haut niveau"""
        # Récupérer les meilleurs joueurs
        challenger_players = self._get_challenger_players()
        grandmaster_players = self._get_grandmaster_players()
        master_players = self._get_master_players()
        
        # Combiner les joueurs
        top_players = challenger_players + grandmaster_players + master_players
        
        # Collecter les PUUID des joueurs
        puuids = [self._get_puuid_by_summoner_id(player["summonerId"]) 
                  for player in top_players[:100]]
        
        # Collecter les match IDs
        match_ids = []
        for puuid in puuids:
            player_matches = self._get_match_ids_by_puuid(puuid, count=20)
            match_ids.extend(player_matches)
            
            if len(set(match_ids)) >= num_matches:
                break
        
        # Éliminer les doublons
        unique_match_ids = list(set(match_ids))[:num_matches]
        
        # Collecter les détails des matches
        matches = []
        for match_id in unique_match_ids:
            match_data = self._get_match_by_id(match_id)
            if self._is_valid_match(match_data):
                processed_match = self._process_match_data(match_data)
                matches.append(processed_match)
        
        return matches
    
    def _process_match_data(self, match_data):
        """Traite les données brutes de match pour extraction de features"""
        processed_match = {
            "match_id": match_data["metadata"]["match_id"],
            "game_version": match_data["info"]["game_version"],
            "game_length": match_data["info"]["game_length"],
            "players": [],
            "rounds": []
        }
        
        # Traiter les données de chaque joueur
        for participant in match_data["info"]["participants"]:
            player_data = {
                "puuid": participant["puuid"],
                "placement": participant["placement"],
                "level": participant["level"],
                "last_round": participant["last_round"],
                "total_damage_to_players": participant["total_damage_to_players"],
                "eliminated_by": participant.get("eliminated_by", None),
                "augments": participant["augments"],
                "traits": self._process_traits(participant["traits"]),
                "units": self._process_units(participant["units"]),
                "economy": {
                    "gold_left": participant["gold_left"],
                    "total_gold_earned": participant.get("total_gold_earned", 0)
                }
            }
            processed_match["players"].append(player_data)
        
        # Reconstruire les rounds et combats
        processed_match["rounds"] = self._reconstruct_rounds(match_data)
        
        return processed_match
    
    def _process_traits(self, traits_data):
        """Extrait les données de traits actifs"""
        processed_traits = {}
        
        for trait in traits_data:
            if trait["tier_current"] > 0:  # Trait actif
                processed_traits[trait["name"]] = {
                    "count": trait["num_units"],
                    "tier": trait["tier_current"],
                    "style": trait["style"]
                }
        
        return processed_traits
    
    def _process_units(self, units_data):
        """Extrait les données des unités"""
        processed_units = []
        
        for unit in units_data:
            processed_unit = {
                "character_id": unit["character_id"],
                "name": unit["character_id"].split("_")[1],  # Ex: "TFT4_Ahri" -> "Ahri"
                "tier": unit["tier"],
                "rarity": unit["rarity"],
                "items": unit["items"],
                "stats": {
                    "hp": unit.get("hp", 0),
                    "mana": unit.get("mana", 0),
                    "armor": unit.get("armor", 0),
                    "mr": unit.get("magicResist", 0),
                    "ad": unit.get("damage", 0)
                }
            }
            processed_units.append(processed_unit)
        
        return processed_units
    
    def _reconstruct_rounds(self, match_data):
        """Tente de reconstruire les rounds et combats à partir des données disponibles"""
        # Note: Cette reconstruction est approximative car l'API ne fournit pas
        # toutes les données détaillées de chaque round
        
        rounds = []
        
        # Extraire les informations de base sur les rounds
        if "rounds" not in match_data["info"]:
            return rounds  # Aucune donnée de round disponible
        
        for round_data in match_data["info"]["rounds"]:
            processed_round = {
                "round_type": round_data["stage_id"],  # PVE, PVP, Carousel, etc.
                "round_num": round_data["round_num"],
                "stage": round_data["stage_id"] // 10,
                "matchups": []
            }
            
            # Reconstruire les matchups
            for matchup in round_data.get("matchups", []):
                if "participants" not in matchup or len(matchup["participants"]) < 2:
                    continue
                    
                p1_id = matchup["participants"][0]
                p2_id = matchup["participants"][1]
                
                p1_data = next((p for p in match_data["info"]["participants"] 
                               if p["puuid"] == p1_id), None)
                p2_data = next((p for p in match_data["info"]["participants"] 
                               if p["puuid"] == p2_id), None)
                
                if not p1_data or not p2_data:
                    continue
                
                # Extraire les données de board au moment du combat (si disponible)
                p1_board = self._extract_board_state(p1_data, round_data["round_num"])
                p2_board = self._extract_board_state(p2_data, round_data["round_num"])
                
                # Déterminer le vainqueur (si l'information est disponible)
                winner_id = matchup.get("winner", None)
                
                processed_matchup = {
                    "player1_id": p1_id,
                    "player2_id": p2_id,
                    "player1_board": p1_board,
                    "player2_board": p2_board,
                    "winner": winner_id,
                    "damage": matchup.get("damage", 0)
                }
                
                processed_round["matchups"].append(processed_matchup)
            
            rounds.append(processed_round)
        
        return rounds
    
    def _extract_board_state(self, player_data, round_num):
        """Extrait l'état du board d'un joueur pour un round spécifique"""
        # Note: Cette fonction est approximative, car l'API ne fournit pas
        # l'état précis du board à chaque round
        
        # Pour les besoins de la simulation, nous utilisons l'état final
        # et effectuons des ajustements en fonction du round
        board = {
            "units": [],
            "traits": {}
        }
        
        # Estimer les unités probablement sur le board à ce round
        estimated_level = min(player_data["level"], 5 + round_num // 4)
        estimated_units = sorted(player_data["units"], 
                                key=lambda u: u["tier"] * 10 + u["rarity"], 
                                reverse=True)[:estimated_level]
        
        board["units"] = estimated_units
        
        # Recalculer les traits actifs
        traits = {}
        for unit in estimated_units:
            unit_name = unit["character_id"].split("_")[1]
            unit_traits = self._get_unit_traits(unit_name)
            
            for trait in unit_traits:
                traits[trait] = traits.get(trait, 0) + 1
        
        # Déterminer les niveaux de traits actifs
        for trait, count in traits.items():
            thresholds = self._get_trait_thresholds(trait)
            tier = 0
            for threshold in thresholds:
                if count >= threshold:
                    tier += 1
                else:
                    break
            
            if tier > 0:
                board["traits"][trait] = {
                    "count": count,
                    "tier": tier
                }
        
        return board
```

### Intégration avec l'Architecture MuZero

L'intégration du simulateur avec notre architecture MuZero est essentielle pour l'entraînement efficace de l'IA:

```python
class MuZeroTFTTrainer:
    def __init__(self, config):
        self.config = config
        self.simulation_system = TFTSimulationSystem(config)
        self.replay_buffer = ReplayBuffer(config.get("replay_buffer_size", 10000))
        
        # Modèles MuZero
        self.representation_model = RepresentationNetwork()
        self.dynamics_model = DynamicsNetwork()
        self.prediction_model = PredictionNetwork()
        
        # Optimiseur
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.get("learning_rate", 0.001),
            weight_decay=config.get("weight_decay", 1e-4)
        )
    
    def train(self, num_iterations):
        """Entraîne le modèle MuZero"""
        # Obtenir le simulateur approprié pour l'entraînement
        simulator = self.simulation_system.get_appropriate_simulator(purpose="training")
        
        for iteration in range(num_iterations):
            # Générer de nouvelles parties avec exploration
            self._generate_self_play_games(simulator, num_games=16)
            
            # Entraîner sur les données collectées
            train_loss = self._update_weights(num_batches=32)
            
            # Évaluer périodiquement
            if iteration % 10 == 0:
                eval_stats = self._evaluate_agent(num_games=32)
                
                print(f"Iteration {iteration}: Loss = {train_loss}, " +
                      f"Avg Placement = {eval_stats['avg_placement']}, " +
                      f"Win Rate = {eval_stats['win_rate']}")
                
                # Évaluation contre le vrai jeu (si en phase 4)
                if self.config.get("development_phase", 1) >= 4 and iteration % 50 == 0:
                    real_game_stats = self.simulation_system.validate_against_real_game(
                        self.get_agent(), num_games=5)
                    print(f"Real Game Evaluation: {real_game_stats}")
    
    def _generate_self_play_games(self, simulator, num_games):
        """Génère des parties en self-play pour l'entraînement"""
        # Réinitialiser l'environnement
        states = simulator.reset()
        
        # Créer les agents pour chaque environnement
        agents = [MuZeroAgent(self.representation_model, 
                              self.dynamics_model, 
                              self.prediction_model, 
                              mcts_simulations=self.config.get("mcts_simulations", 50),
                              exploration_weight=self.config.get("exploration_weight", 1.0))
                  for _ in range(simulator.num_envs)]
        
        dones = np.zeros(simulator.num_envs, dtype=bool)
        game_memories = [[] for _ in range(simulator.num_envs)]
        
        while not np.all(dones):
            # Sélectionner les actions avec MCTS
            actions = []
            for i, agent in enumerate(agents):
                if not dones[i]:
                    action = agent.select_action(states[i])
                    actions.append(action)
                else:
                    actions.append(np.zeros_like(actions[0]))  # Action factice
            
            # Exécuter les actions dans l'environnement
            next_states, rewards, new_dones = simulator.step(actions)
            
            # Stocker les transitions dans la mémoire de jeu
            for i in range(simulator.num_envs):
                if not dones[i]:
                    game_memories[i].append({
                        "state": states[i],
                        "action": actions[i],
                        "reward": rewards[i],
                        "next_state": next_states[i],
                        "done": new_dones[i]
                    })
            
            # Mettre à jour les états et drapeaux
            states = next_states
            dones = new_dones
        
        # Traiter les jeux terminés
        for memory in game_memories:
            # Calculer les valeurs cibles en utilisant les récompenses
            values = self._compute_target_values(memory)
            
            # Ajouter au buffer de replay
            for i, transition in enumerate(memory):
                self.replay_buffer.add({
                    "state": transition["state"],
                    "action": transition["action"],
                    "value": values[i],
                    "reward": transition["reward"],
                    "policy": transition.get("policy", None)  # Si disponible
                })
    
    def _update_weights(self, num_batches):
        """Met à jour les poids des modèles"""
        total_loss = 0
        
        for _ in range(num_batches):
            # Échantillonner un batch du buffer de replay
            batch = self.replay_buffer.sample(self.config.get("batch_size", 128))
            
            with tf.GradientTape() as tape:
                # Calculer les prédictions des modèles
                representation_loss = self._compute_representation_loss(batch)
                dynamics_loss = self._compute_dynamics_loss(batch)
                prediction_loss = self._compute_prediction_loss(batch)
                
                # Perte totale
                loss = representation_loss + dynamics_loss + prediction_loss
                
                # Régularisation L2
                for model in [self.representation_model, self.dynamics_model, self.prediction_model]:
                    for variable in model.trainable_variables:
                        loss += self.config.get("weight_decay", 1e-4) * tf.nn.l2_loss(variable)
            
            # Calculer et appliquer les gradients
            grads = tape.gradient(loss, self._get_all_trainable_variables())
            self.optimizer.apply_gradients(zip(grads, self._get_all_trainable_variables()))
            
            total_loss += loss.numpy()
        
        return total_loss / num_batches
    
    def _get_all_trainable_variables(self):
        """Récupère toutes les variables entraînables des modèles"""
        variables = []
        variables.extend(self.representation_model.trainable_variables)
        variables.extend(self.dynamics_model.trainable_variables)
        variables.extend(self.prediction_model.trainable_variables)
        return variables
    
    def get_agent(self):
        """Retourne l'agent MuZero avec les modèles actuels"""
        return MuZeroAgent(self.representation_model, 
                          self.dynamics_model, 
                          self.prediction_model,
                          mcts_simulations=self.config.get("evaluation_simulations", 100),
                          exploration_weight=0.0)  # Pas d'exploration en évaluation
```

Ces implémentations détaillées montrent comment les aspects les plus complexes de TFT sont simulés et intégrés avec l'architecture MuZero pour l'entraînement efficace de notre IA.

---

<div align="center">
  <sub>Built with ❤️ by the TFT AI Team</sub>
</div> 