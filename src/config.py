"""
Configuration pour le projet TFT IA.
Ce fichier contient tous les paramètres et hyperparamètres pour l'environnement de jeu et l'agent IA.
"""

config = {
    'training': {
        'num_episodes': 10000,
        'batch_size': 32,
        'learning_rate': 0.001,
        'discount_factor': 0.99,
        'optimizer': 'adam',
        'replay_buffer_size': 10000,
        'min_replay_buffer_size': 1000,
        'num_unroll_steps': 5,
        'td_steps': 10,
        'weight_decay': 1e-4,
    },
    'environment': {
        'num_players': 4,
        'max_rounds': 20,
        'state_compression': True,
        'parallel_combat': True,
        'reward_shaping': True,
    },
    'agent': {
        'use_imitation_learning': True,
        'opponent_prediction': True,
        'representation_channels': 128,
        'dynamics_channels': 128,
        'prediction_channels': 128,
        'num_simulations': 50,
        'root_dirichlet_alpha': 0.25,
        'root_exploration_fraction': 0.25,
    },
    'game_mechanics': {
        'interest_thresholds': [10, 20, 30, 40, 50],
        'game_stage_weights': {
            'early': 0.8,
            'mid': 0.5,
            'late': 0.3
        },
        'critical_rounds': {
            'fast_8': [4.1, 4.2],
            'reroll_1cost': [3.2],
            'reroll_2cost': [3.5],
            'reroll_3cost': [4.1],
            'stabilize': [3.2, 4.1]
        },
        'champion_probabilities': {
            4: {1: 0.55, 2: 0.30, 3: 0.15, 4: 0.00, 5: 0.00},
            5: {1: 0.45, 2: 0.33, 3: 0.20, 4: 0.02, 5: 0.00},
            6: {1: 0.25, 2: 0.40, 3: 0.30, 4: 0.05, 5: 0.00},
            7: {1: 0.19, 2: 0.30, 3: 0.35, 4: 0.15, 5: 0.01},
            8: {1: 0.16, 2: 0.20, 3: 0.35, 4: 0.25, 5: 0.04},
            9: {1: 0.09, 2: 0.15, 3: 0.30, 4: 0.30, 5: 0.16}
        }
    }
} 