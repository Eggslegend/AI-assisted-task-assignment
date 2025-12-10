import torch
from scipy.optimize import linear_sum_assignment
import numpy as np

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