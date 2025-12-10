import pandas as pd
import numpy as np

class SyntheticDataGenerator:
    """
    Generates synthetic training data using the simulator.

    Pipeline:
    1. Primary model assigns tasks
    2. Simulator executes and generates ground truth scores
    3. Results are formatted as training data for primary model
    """

    def __init__(self, assigner, simulator, config):
        self.assigner = assigner
        self.simulator = simulator
        self.config = config

    def generate_random_task_slots(self):
        """Generate random task slots."""
        return [
            (np.random.randint(1, 5), np.random.randint(1, 6))
            for _ in range(4)
        ]

    def generate_batch(self, n_rounds=50, use_model=True, verbose=False):
        """
        Generate a batch of synthetic training data.

        Args:
            n_rounds: Number of assignment rounds to simulate
            use_model: Whether to use transformer model in simulator
            verbose: Whether to print progress

        Returns:
            DataFrame with synthetic training data
        """
        synthetic_data = []

        if verbose:
            print(f"\nGenerating {n_rounds} rounds of synthetic data...")

        for round_idx in range(n_rounds):
            # Generate random task slots
            task_slots = self.generate_random_task_slots()

            # Primary model makes assignment
            assignment_result = self.assigner.find_optimal_assignment(task_slots)

            # Simulator executes and generates ground truth
            execution_results = self.simulator.execute_assignment(
                assignment_result,
                use_model=use_model
            )

            # Convert to training data format
            for exec_result in execution_results:
                person = exec_result['person']
                skills = self.simulator.growth_tracker.get_current_skills(person)

                synthetic_data.append({
                    'person_id': person,
                    'task_difficulty': exec_result['difficulty'],
                    'task_id': exec_result['task_id'],
                    'skill_1': skills['skill_1'],
                    'skill_2': skills['skill_2'],
                    'skill_3': skills['skill_3'],
                    'skill_4': skills['skill_4'],
                    'score': exec_result['final_score'],  # Ground truth from simulator
                    'predicted_score': exec_result['predicted_score'],
                    'penalty_triggered': exec_result['penalty_triggered'],
                    'round': round_idx
                })

            if verbose and (round_idx + 1) % 10 == 0:
                print(f"  Completed round {round_idx + 1}/{n_rounds}")

        df = pd.DataFrame(synthetic_data)

        if verbose:
            print(f"\nGenerated {len(df)} synthetic training samples")
            print(f"  Penalty triggered: {df['penalty_triggered'].sum()} times ({100*df['penalty_triggered'].mean():.1f}%)")
            avg_error = (df['score'] - df['predicted_score']).abs().mean()
            print(f"  Average prediction error: {avg_error:.3f}")

        return df

    def augment_training_data(self, original_df, n_rounds=50, use_model=True):
        """
        Augment original training data with synthetic data.

        Args:
            original_df: Original Training_data.xlsx DataFrame
            n_rounds: Number of simulation rounds
            use_model: Whether to use transformer in simulator

        Returns:
            Combined DataFrame with original + synthetic data
        """
        # Generate synthetic data
        synthetic_df = self.generate_batch(n_rounds=n_rounds, use_model=use_model, verbose=True)

        # Keep only columns that match original format
        synthetic_subset = synthetic_df[['person_id', 'task_difficulty', 'task_id',
                                          'skill_1', 'skill_2', 'skill_3', 'skill_4', 'score']]

        # Combine
        combined_df = pd.concat([original_df, synthetic_subset], ignore_index=True)

        print(f"\nAugmented dataset:")
        print(f"  Original samples: {len(original_df)}")
        print(f"  Synthetic samples: {len(synthetic_subset)}")
        print(f"  Total samples: {len(combined_df)}")

        return combined_df