import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    """Handles all data loading and preprocessing."""

    def __init__(self, config):
        self.config = config
        self.skill_profiles = None
        # 建立人員 ID 到索引的映射，用於 Embedding 或數值輸入
        self.person_to_idx = {p: i for i, p in enumerate(config['people'])}
        self.idx_to_person = {i: p for p, i in self.person_to_idx.items()}
        self.scaler = StandardScaler()

    def load_and_preprocess(self, data_path):
        """Load raw data and compute fixed skill profiles."""
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            df = pd.read_excel(data_path)
            
        # 確保欄位名稱一致
        if data_path.endswith('.csv'):
            # CSV 格式: Person_ID,Task_Difficulty (1-5),Task_ID,Person_Skill_Task1 (1-5),Person_Skill_Task2 (1-5),Person_Skill_Task3 (1-5),Person_Skill_Task4 (1-5),Completion_Score (0-100)
            df.columns = ['person_id', 'task_difficulty', 'task_id',
                          'skill_1', 'skill_2', 'skill_3', 'skill_4', 'score']
        else:
            # Excel 格式 (假設與 CSV 類似或舊格式)
            df.columns = ['person_id', 'task_difficulty', 'task_id',
                          'skill_1', 'skill_2', 'skill_3', 'skill_4', 'score']

        # Compute average skill profiles per person
        self.skill_profiles = df.groupby('person_id')[
            ['skill_1', 'skill_2', 'skill_3', 'skill_4']
        ].mean().to_dict('index')

        print("Computed Skill Profiles (averaged):")
        for person, skills in self.skill_profiles.items():
            print(f"  {person}: {[f'{v:.2f}' for v in skills.values()]}")

        return df

    def create_features(self, df):
        """Create feature matrix from dataframe."""
        features = []
        targets = []

        for _, row in df.iterrows():
            person = row['person_id']
            task_id = int(row['task_id'])
            difficulty = row['task_difficulty']
            
            # 使用預先計算的技能檔案
            skills = self.skill_profiles[person]
            relevant_skill = skills[f'skill_{task_id}']

            feature = [
                self.person_to_idx[person],
                task_id,
                difficulty,
                relevant_skill,
                skills['skill_1'],
                skills['skill_2'],
                skills['skill_3'],
                skills['skill_4'],
            ]

            features.append(feature)
            targets.append(row['score'])

        X = np.array(features, dtype=np.float32)
        y = np.array(targets, dtype=np.float32).reshape(-1, 1)
        return X, y

    def fit_scaler(self, X):
        """Fit the StandardScaler on training data."""
        self.scaler.fit(X)

    def transform(self, X):
        """Transform data using the fitted scaler."""
        return self.scaler.transform(X)

    def create_inference_features(self, task_slots):
        """
        Create features for all person-task combinations during inference.

        Args:
            task_slots: List of 4 tuples [(task_id, difficulty), ...]
        """
        features = []

        for person in self.config['people']:
            for task_id, difficulty in task_slots:
                skills = self.skill_profiles[person]
                relevant_skill = skills[f'skill_{task_id}']

                feature = [
                    self.person_to_idx[person],
                    task_id,
                    difficulty,
                    relevant_skill,
                    skills['skill_1'],
                    skills['skill_2'],
                    skills['skill_3'],
                    skills['skill_4'],
                ]
                features.append(feature)

        return np.array(features, dtype=np.float32)

    def restore_state(self, save_dict):
        """
        Restore preprocessor state from a saved dictionary.
        Used when loading a trained model.
        """
        self.skill_profiles = save_dict['skill_profiles']
        self.scaler.mean_ = save_dict['scaler_mean']
        self.scaler.scale_ = save_dict['scaler_scale']
        self.person_to_idx = save_dict['person_to_idx']
        self.idx_to_person = {i: p for p, i in self.person_to_idx.items()}

