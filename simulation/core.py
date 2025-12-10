import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from models.predictors import TransformerScorePredictor

class PersonalGrowthTracker:
    """
    Tracks task completion history and manages skill growth over time.

    As a person completes more tasks, their skills gradually improve,
    leading to better scores on future tasks.
    """

    def __init__(self, people, initial_skills, config):
        """
        Args:
            people: List of person IDs ['A', 'B', 'C', 'D']
            initial_skills: Dict of initial skill profiles from preprocessor
            config: SIMULATOR_CONFIG dictionary
        """
        self.people = people
        self.config = config

        # Initialize current skills (mutable, will grow over time)
        self.current_skills = {}
        for person in people:
            self.current_skills[person] = {
                f'skill_{i}': initial_skills[person][f'skill_{i}']
                for i in range(1, 5)
            }

        # Track task completion history
        self.task_history = {person: [] for person in people}
        self.total_tasks_completed = {person: 0 for person in people}

    def get_current_skills(self, person):
        """Get current skill levels for a person (after growth)."""
        return self.current_skills[person].copy()

    def get_relevant_skill(self, person, task_id):
        """Get the skill level relevant to a specific task."""
        return self.current_skills[person][f'skill_{task_id}']

    def record_task_completion(self, person, task_id, difficulty, score):
        """
        Record a completed task and update skills accordingly.

        Args:
            person: Person ID
            task_id: Type of task completed (1-4)
            difficulty: Task difficulty (1-5)
            score: Score achieved
        """
        # Record history
        self.task_history[person].append({
            'task_id': task_id,
            'difficulty': difficulty,
            'score': score
        })
        self.total_tasks_completed[person] += 1

        # Calculate skill growth
        n_tasks = self.total_tasks_completed[person]
        base_growth = self.config['skill_growth_rate']
        decay = self.config['growth_decay'] ** (n_tasks - 1)  # Diminishing returns
        growth = base_growth * decay

        # Apply growth to the relevant skill
        skill_key = f'skill_{task_id}'
        current = self.current_skills[person][skill_key]
        new_skill = min(current + growth, self.config['max_skill_level'])
        self.current_skills[person][skill_key] = new_skill

        # Small spillover growth to other skills (learning transfer)
        spillover = growth * 0.25
        for i in range(1, 5):
            if i != task_id:
                other_key = f'skill_{i}'
                current_other = self.current_skills[person][other_key]
                new_other = min(current_other + spillover, self.config['max_skill_level'])
                self.current_skills[person][other_key] = new_other

    def get_experience_factor(self, person):
        """
        Get experience multiplier based on total tasks completed.
        More experience = slightly better baseline performance.
        """
        n_tasks = self.total_tasks_completed[person]
        # Logarithmic growth: caps at ~1.2x after many tasks
        return 1.0 + 0.05 * np.log1p(n_tasks)

    def reset(self, initial_skills):
        """Reset all growth and history."""
        for person in self.people:
            self.current_skills[person] = {
                f'skill_{i}': initial_skills[person][f'skill_{i}']
                for i in range(1, 5)
            }
        self.task_history = {person: [] for person in self.people}
        self.total_tasks_completed = {person: 0 for person in self.people}

    def print_status(self):
        """Print current skill levels and task counts."""
        print("\n" + "="*60)
        print("PERSONAL GROWTH STATUS")
        print("="*60)
        for person in self.people:
            skills = self.current_skills[person]
            n_tasks = self.total_tasks_completed[person]
            exp_factor = self.get_experience_factor(person)
            print(f"\nPerson {person} ({n_tasks} tasks completed, exp factor: {exp_factor:.3f}):")
            print(f"  Skills: [{skills['skill_1']:.2f}, {skills['skill_2']:.2f}, "
                  f"{skills['skill_3']:.2f}, {skills['skill_4']:.2f}]")

class TaskExecutionSimulator:
    """
    Simulates realistic task execution with:
    - Transformer-based score prediction
    - Difficulty-skill penalty system
    - Personal growth tracking

    This model generates "ground truth" scores for training the primary model.
    """

    def __init__(self, preprocessor, config, sim_config, device):
        """
        Args:
            preprocessor: DataPreprocessor with skill profiles
            config: Main CONFIG dictionary
            sim_config: SIMULATOR_CONFIG dictionary
            device: torch device
        """
        self.preprocessor = preprocessor
        self.config = config
        self.sim_config = sim_config
        self.device = device

        # Initialize growth tracker
        self.growth_tracker = PersonalGrowthTracker(
            people=config['people'],
            initial_skills=preprocessor.skill_profiles,
            config=sim_config
        )

        # Initialize transformer model
        # Input: [person_idx, task_id, difficulty, relevant_skill, skill_1-4, experience_factor, n_tasks_completed]
        self.input_dim = 10  # Extended features
        self.model = TransformerScorePredictor(
            input_dim=self.input_dim,
            config=sim_config
        ).to(device)

        # Feature scaler for simulator
        self.scaler = StandardScaler()
        self.scaler_fitted = False

    def compute_difficulty_penalty(self, difficulty, skill_level):
        """
        Compute penalty factor when difficulty is high and skill is low.

        Returns a multiplier between penalty_factor and 1.0
        """
        if (difficulty > self.sim_config['high_difficulty_threshold'] and
            skill_level < self.sim_config['low_skill_threshold']):
            # Substantial penalty
            base_penalty = self.sim_config['penalty_factor']
            # Add some noise for realism
            noise = np.random.normal(0, self.sim_config['penalty_noise_std'])
            penalty = np.clip(base_penalty + noise, 0.2, 0.6)
            return penalty

        # Gradual penalty for moderate mismatches
        elif difficulty > skill_level + 1:
            # Mild penalty proportional to gap
            gap = difficulty - skill_level
            penalty = max(0.7, 1.0 - 0.1 * gap)
            return penalty

        return 1.0  # No penalty

    def create_extended_features(self, person, task_id, difficulty):
        """
        Create extended feature vector including growth-related features.
        """
        skills = self.growth_tracker.get_current_skills(person)
        relevant_skill = skills[f'skill_{task_id}']
        experience_factor = self.growth_tracker.get_experience_factor(person)
        n_tasks = self.growth_tracker.total_tasks_completed[person]

        features = [
            self.preprocessor.person_to_idx[person],  # Person index
            task_id,                                   # Task type
            difficulty,                                # Task difficulty
            relevant_skill,                           # Relevant skill (with growth)
            skills['skill_1'],                        # All current skills
            skills['skill_2'],
            skills['skill_3'],
            skills['skill_4'],
            experience_factor,                        # Experience multiplier
            n_tasks / 10.0,                          # Normalized task count
        ]

        return np.array(features, dtype=np.float32)

    def execute_task(self, person, task_id, difficulty, use_model=True):
        """
        Simulate task execution and return realistic score.

        Args:
            person: Person ID ('A', 'B', 'C', 'D')
            task_id: Task type (1-4)
            difficulty: Task difficulty (1-5)
            use_model: If True, use transformer; else use rule-based simulation

        Returns:
            Dictionary with execution results
        """
        # Get current skills (with growth)
        skills = self.growth_tracker.get_current_skills(person)
        relevant_skill = skills[f'skill_{task_id}']
        experience_factor = self.growth_tracker.get_experience_factor(person)

        if use_model and self.scaler_fitted:
            # Use transformer model
            features = self.create_extended_features(person, task_id, difficulty)
            features_scaled = self.scaler.transform(features.reshape(1, -1))

            self.model.eval()
            with torch.no_grad():
                x = torch.FloatTensor(features_scaled).to(self.device)
                base_score = self.model(x).cpu().item()
        else:
            # Rule-based simulation (used before model is trained)
            # Base score from skill-difficulty interaction
            skill_factor = relevant_skill / 5.0  # Normalize to 0-1
            difficulty_factor = (6 - difficulty) / 5.0  # Easier = higher base

            base_score = (skill_factor * 0.6 + difficulty_factor * 0.3 + 0.1) * 5.0
            base_score *= experience_factor

            # Add randomness
            noise = np.random.normal(0, 0.3)
            base_score += noise

        # Apply difficulty-skill penalty
        penalty = self.compute_difficulty_penalty(difficulty, relevant_skill)
        final_score = base_score * penalty

        # Clamp to valid range
        final_score = np.clip(final_score,
                              self.sim_config['min_score'],
                              self.sim_config['max_score'])

        # Record task completion and update growth
        self.growth_tracker.record_task_completion(person, task_id, difficulty, final_score)

        return {
            'person': person,
            'task_id': task_id,
            'task_name': self.config['task_names'].get(task_id, f'Task {task_id}'),
            'difficulty': difficulty,
            'relevant_skill': relevant_skill,
            'experience_factor': experience_factor,
            'penalty_applied': penalty,
            'base_score': base_score if not use_model else None,
            'final_score': final_score,
            'penalty_triggered': penalty < 1.0
        }

    def execute_assignment(self, assignment_result, use_model=True):
        """
        Execute a full assignment from the primary model.

        Args:
            assignment_result: Output from TaskAssigner.find_optimal_assignment()
            use_model: Whether to use transformer model

        Returns:
            List of execution results with ground truth scores
        """
        execution_results = []

        for assignment in assignment_result['assignments']:
            result = self.execute_task(
                person=assignment['person'],
                task_id=assignment['task_id'],
                difficulty=assignment['difficulty'],
                use_model=use_model
            )
            result['predicted_score'] = assignment['predicted_score']
            result['prediction_error'] = result['final_score'] - assignment['predicted_score']
            execution_results.append(result)

        return execution_results

    def reset_growth(self):
        """Reset personal growth to initial state."""
        self.growth_tracker.reset(self.preprocessor.skill_profiles)