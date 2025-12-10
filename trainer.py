import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from models.predictors import ScorePredictor
from scipy.optimize import linear_sum_assignment

class TaskDataset(Dataset):
    """PyTorch Dataset for task assignment data."""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

class Trainer:
    """Handles model training with cross-validation."""

    def __init__(self, config, device):
        self.config = config
        self.device = device

    def train_epoch(self, model, train_loader, optimizer, criterion):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(X_batch)

        return total_loss / len(train_loader.dataset)

    def validate(self, model, val_loader, criterion):
        model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                total_loss += loss.item() * len(X_batch)

                all_preds.extend(predictions.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        mse = total_loss / len(val_loader.dataset)
        mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))

        ss_res = np.sum((np.array(all_targets) - np.array(all_preds)) ** 2)
        ss_tot = np.sum((np.array(all_targets) - np.mean(all_targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return mse, mae, r2, np.array(all_preds), np.array(all_targets)

    def train_fold(self, model, train_loader, val_loader, fold_num):
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            val_mse, val_mae, val_r2, _, _ = self.validate(model, val_loader, criterion)

            scheduler.step(val_mse)

            if val_mse < best_val_loss:
                best_val_loss = val_mse
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= self.config['early_stopping_patience']:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch + 1}: Train Loss={train_loss:.4f}, "
                      f"Val MSE={val_mse:.4f}, Val MAE={val_mae:.4f}, Val R²={val_r2:.4f}")

        model.load_state_dict(best_model_state)
        return model

    def cross_validate(self, dataset, preprocessor):
        kfold = KFold(n_splits=self.config['n_folds'], shuffle=True,
                      random_state=self.config['random_seed'])

        fold_results = []
        all_predictions = []
        all_targets = []
        best_model = None
        best_val_loss = float('inf')

        input_dim = dataset.X.shape[1]

        print(f"\n{'='*60}")
        print(f"Starting {self.config['n_folds']}-Fold Cross-Validation")
        print(f"{'='*60}")

        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            print(f"\nFold {fold + 1}/{self.config['n_folds']}")
            print("-" * 40)

            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=self.config['batch_size'],
                                       shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=self.config['batch_size'])

            model = ScorePredictor(
                input_dim=input_dim,
                hidden_layers=self.config['hidden_layers'],
                dropout_rate=self.config['dropout_rate']
            ).to(self.device)

            model = self.train_fold(model, train_loader, val_loader, fold)

            criterion = nn.MSELoss()
            val_mse, val_mae, val_r2, preds, targets = self.validate(
                model, val_loader, criterion
            )

            fold_results.append({
                'fold': fold + 1,
                'mse': val_mse,
                'rmse': np.sqrt(val_mse),
                'mae': val_mae,
                'r2': val_r2
            })

            all_predictions.extend(preds)
            all_targets.extend(targets)

            print(f"  Final - MSE: {val_mse:.4f}, RMSE: {np.sqrt(val_mse):.4f}, "
                  f"MAE: {val_mae:.4f}, R²: {val_r2:.4f}")

            if val_mse < best_val_loss:
                best_val_loss = val_mse
                best_model = model.state_dict().copy()

        # Summary
        print(f"\n{'='*60}")
        print("Cross-Validation Summary")
        print(f"{'='*60}")

        results_df = pd.DataFrame(fold_results)
        print(f"\nPer-Fold Results:")
        print(results_df.to_string(index=False))

        print(f"\nOverall Metrics (mean ± std):")
        for metric in ['mse', 'rmse', 'mae', 'r2']:
            mean_val = results_df[metric].mean()
            std_val = results_df[metric].std()
            print(f"  {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")

        return best_model, fold_results, np.array(all_predictions), np.array(all_targets)


class TaskAssigner:
    """Handles optimal task assignment using Hungarian algorithm."""

    def __init__(self, model, preprocessor, device):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self.people = preprocessor.config['people']
        self.task_names = preprocessor.config.get('task_names', {})

    def get_task_name(self, task_id):
        """Get task name from task ID."""
        return self.task_names.get(task_id, f'Unknown Task {task_id}')

    def predict_scores(self, task_slots):
        """
        Predict scores for all person-task combinations.

        Args:
            task_slots: List of 4 tuples [(task_id, difficulty), ...]

        Returns:
            4x4 numpy array of predicted scores
        """
        features = self.preprocessor.create_inference_features(task_slots)
        features_scaled = self.preprocessor.transform(features)

        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(features_scaled).to(self.device)
            predictions = self.model(X).cpu().numpy().flatten()

        score_matrix = predictions.reshape(4, 4)
        return score_matrix

    def find_optimal_assignment(self, task_slots):
        """
        Find optimal bijective assignment using Hungarian algorithm.

        Args:
            task_slots: List of 4 tuples [(task_id, difficulty), ...]

        Returns:
            Dictionary with assignment details
        """
        score_matrix = self.predict_scores(task_slots)

        # Hungarian algorithm (maximize by negating)
        cost_matrix = -score_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assignments = []
        total_score = 0

        for person_idx, slot_idx in zip(row_ind, col_ind):
            person = self.people[person_idx]
            task_id, difficulty = task_slots[slot_idx]
            task_name = self.get_task_name(task_id)
            predicted_score = score_matrix[person_idx, slot_idx]

            assignments.append({
                'person': person,
                'slot': slot_idx + 1,
                'task_id': task_id,
                'task_name': task_name,
                'difficulty': difficulty,
                'predicted_score': predicted_score
            })
            total_score += predicted_score

        return {
            'assignments': assignments,
            'total_score': total_score,
            'score_matrix': score_matrix,
            'task_slots': task_slots
        }

class SimulatorTrainer:
    """
    Training pipeline for the Task Execution Simulator.
    """

    def __init__(self, simulator, sim_config, device):
        self.simulator = simulator
        self.sim_config = sim_config
        self.device = device

    def prepare_training_data(self, df, preprocessor):
        """
        Prepare training data for the simulator from original dataset.
        """
        features = []
        targets = []

        for _, row in df.iterrows():
            person = row['person_id']
            task_id = int(row['task_id'])
            difficulty = row['task_difficulty']
            score = row['score']

            # Get skill profile
            skills = preprocessor.skill_profiles[person]

            feature = [
                preprocessor.person_to_idx[person],
                task_id,
                difficulty,
                skills[f'skill_{task_id}'],
                skills['skill_1'],
                skills['skill_2'],
                skills['skill_3'],
                skills['skill_4'],
                1.0,  # Initial experience factor
                0.0,  # Initial task count
            ]

            features.append(feature)
            targets.append(score)

        X = np.array(features, dtype=np.float32)
        y = np.array(targets, dtype=np.float32).reshape(-1, 1)

        return X, y

    def train(self, X, y, verbose=True):
        """
        Train the simulator's transformer model.
        """
        # Fit scaler
        self.simulator.scaler.fit(X)
        self.simulator.scaler_fitted = True
        X_scaled = self.simulator.scaler.transform(X)

        # Create dataset
        dataset = TaskDataset(X_scaled, y)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.sim_config['sim_batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.sim_config['sim_batch_size']
        )

        # Optimizer and loss
        optimizer = optim.AdamW(
            self.simulator.model.parameters(),
            lr=self.sim_config['sim_learning_rate'],
            weight_decay=1e-4
        )
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.sim_config['sim_epochs']
        )

        best_val_loss = float('inf')
        best_model_state = None

        if verbose:
            print("\n" + "="*60)
            print("TRAINING TASK EXECUTION SIMULATOR (Transformer)")
            print("="*60)

        for epoch in range(self.sim_config['sim_epochs']):
            # Training
            self.simulator.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                predictions = self.simulator.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(X_batch)

            train_loss /= len(train_loader.dataset)

            # Validation
            self.simulator.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    predictions = self.simulator.model(X_batch)
                    loss = criterion(predictions, y_batch)
                    val_loss += loss.item() * len(X_batch)

            val_loss /= len(val_loader.dataset)

            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.simulator.model.state_dict().copy()

            if verbose and (epoch + 1) % 30 == 0:
                print(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        # Load best model
        self.simulator.model.load_state_dict(best_model_state)

        if verbose:
            print(f"\nTraining complete. Best Val Loss: {best_val_loss:.4f}")

        return best_val_loss