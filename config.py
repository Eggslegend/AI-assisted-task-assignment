# Configuration Settings
CONFIG = {
    'data_path': 'Training_data.csv',
    'random_seed': 42,
    'hidden_layers': [64, 32, 16],
    'dropout_rate': 0.2,
    'n_folds': 4,
    'epochs': 200,
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'early_stopping_patience': 20,
    'people': ['A', 'B', 'C', 'D'],
    'n_tasks': 4,
    'task_names': {
        1: 'Coding',
        2: 'Money Laundering',
        3: 'Customer Searching',
        4: 'Accounting'
    }
}

SIMULATOR_CONFIG = {
    'd_model': 64,
    'n_heads': 4,
    'n_layers': 2,
    'dim_feedforward': 128,
    'dropout': 0.1,
    'sim_epochs': 150,
    'sim_batch_size': 32,
    'sim_learning_rate': 0.0005,
    'skill_growth_rate': 0.02,
    'max_skill_level': 5.0,
    'growth_decay': 0.95,
    'high_difficulty_threshold': 4,
    'low_skill_threshold': 3,
    'penalty_factor': 0.4,
    'penalty_noise_std': 0.1,
    'min_score': 0.0,
    'max_score': 100,
}

DEVICE_MODE = 'cpu'