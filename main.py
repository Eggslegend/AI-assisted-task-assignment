from config import CONFIG, SIMULATOR_CONFIG
from utils import set_seed, get_device, save_model, load_model, get_task_name, run_inference, print_assignment_result, plot_results, print_execution_results
from data.preprocessor import DataPreprocessor
from data.dataset import TaskDataset
from models.predictors import ScorePredictor
from models.assigner import TaskAssigner
from trainer import Trainer, SimulatorTrainer
from simulation.core import TaskExecutionSimulator
from simulation.generator import SyntheticDataGenerator
import visualization as viz
import numpy as np
import torch


def initial_train():
    # 1. Setup
    set_seed(CONFIG['random_seed'])
    DEVICE = get_device()

    # 2. Preprocessing
    preprocessor = DataPreprocessor(CONFIG)
    df = preprocessor.load_and_preprocess(CONFIG['data_path'])
    X, y = preprocessor.create_features(df)
    preprocessor.fit_scaler(X)
    dataset = TaskDataset(preprocessor.transform(X), y)

    # 3. Train Main Model
    trainer = Trainer(CONFIG, DEVICE)
    best_state, _, _, _ = trainer.cross_validate(dataset, preprocessor)
    
    final_model = ScorePredictor(X.shape[1], CONFIG['hidden_layers'], CONFIG['dropout_rate']).to(DEVICE)
    final_model.load_state_dict(best_state)
    assigner = TaskAssigner(final_model, preprocessor, DEVICE)

    # 4. Train Simulator
    simulator = TaskExecutionSimulator(preprocessor, CONFIG, SIMULATOR_CONFIG, DEVICE)
    sim_trainer = SimulatorTrainer(simulator, SIMULATOR_CONFIG, DEVICE)
    X_sim, y_sim = sim_trainer.prepare_training_data(df, preprocessor)
    sim_trainer.train(X_sim, y_sim)

    # 5. Run Demo
    simulator.reset_growth()
    
    return assigner, simulator, final_model, preprocessor,df, DEVICE

def demo(assigner, simulator):
    all_execution_history = []
    growth_history = []
    for round_num in range(3):
        print(f"\n{'='*70}")
        print(f"ROUND {round_num + 1}")
        print("="*70)

        # Create task slots (including high-difficulty tasks to trigger penalties)
        if round_num == 0:
            task_slots = [(1, 3), (2, 2), (3, 4), (4, 3)]  # Normal difficulty
        elif round_num == 1:
            task_slots = [(1, 5), (2, 5), (3, 5), (4, 5)]  # High difficulty - may trigger penalties
        else:
            task_slots = [(1, 4), (2, 3), (3, 4), (4, 4)]  # Medium-high

        # Step 1: Primary model assigns tasks
        print("\n--- Step 1: Primary Model Assignment ---")
        assignment_result = assigner.find_optimal_assignment(task_slots)
        print_assignment_result(assignment_result)

        # Step 2: Simulator executes and generates ground truth
        print("\n--- Step 2: Simulator Execution ---")
        execution_results = simulator.execute_assignment(assignment_result, use_model=True)
        print_execution_results(execution_results)
        
        # 收集執行歷史
        all_execution_history.append(execution_results)
        
        # 收集成長數據
        round_growth = {}
        for person in CONFIG['people']:
            skills = simulator.growth_tracker.get_current_skills(person)
            avg_skill = np.mean([skills[f'skill_{i}'] for i in range(1, 5)])
            score = next((r['final_score'] for r in execution_results if r['person'] == person), 0)
            round_growth[person] = {'avg_skill': avg_skill, 'score': score}
        growth_history.append(round_growth)
        
        # === 視覺化每一輪的結果 ===
        print("\n--- Step 3: Visualization ---")
        viz.visualize_assignment_and_execution(assignment_result, execution_results)

    # Show final growth status
    simulator.growth_tracker.print_status()

    # === 視覺化技能成長 ===
    print("\n" + "="*70)
    print("VISUALIZATION: Skill Growth Over Rounds")
    print("="*70)
    viz.visualize_growth_over_rounds(growth_history)

# === 視覺化 Penalty 分析 ===
    print("\n" + "="*70)
    print("VISUALIZATION: Penalty Analysis")
    print("="*70)
    viz.visualize_penalty_analysis(all_execution_history)
# === 視覺化每個人做四種工作的分數折線圖 ===
    print("\n" + "="*70)
    print("VISUALIZATION: Person-Task Score Line Charts")
    print("="*70)

# 折線圖: 每個人做四種工作的預測分數 (中等難度)
    viz.visualize_person_task_scores(assigner, difficulties=[3, 3, 3, 3])

# 熱力圖: 人員-任務分數矩陣
    print("\n--- Person-Task Score Heatmap ---")
    viz.visualize_person_task_heatmap(assigner, difficulty=3)
# 多難度分析: 每個人在不同難度下的表現
    print("\n--- Multi-Difficulty Analysis ---")
    viz.visualize_person_task_scores_multi_difficulty(assigner)
    
    viz.visualize_assignment_and_execution(assignment_result, execution_results)

def AISim(simulator, assigner, DEVICE, random_tasks):
    # === 新增功能: 100筆隨機任務的自動分配與統計 ===
    print("\n" + "="*60)
    print("AUTOMATED ASSIGNMENT & EXECUTION STATISTICS (100 TASKS)")
    print("="*60)
    
    # 1. 生成 100 筆隨機任務 (Task ID, Difficulty)
    
    # 統計變數
    person_task_counts = {p: 0 for p in CONFIG['people']}
    person_scores = {p: [] for p in CONFIG['people']}
    # 新增: 統計每個人拿到每種工作的數量 {Person: {TaskID: Count}}
    person_task_type_counts = {p: {t: 0 for t in range(1, 5)} for p in CONFIG['people']}
    # 新增: 統計每個人拿到每種工作的總分 {Person: {TaskID: TotalScore}}
    person_task_type_scores = {p: {t: 0.0 for t in range(1, 5)} for p in CONFIG['people']}
    total_score_sum = 0
    
    print("Processing 100 tasks using Trained Assigner (MLP)...")
    
    # 2. 逐一處理每個任務
    for i, (task_id, difficulty) in enumerate(random_tasks):
        
        # 使用 Assigner (MLP) 預測每個人做這個任務的分數
        # 我們利用 preprocessor 建立特徵，並用 assigner.model 進行推論
        
        # 建立特徵: create_inference_features 會針對所有候選人產生特徵
        # 傳入 [(task_id, difficulty)]，它會回傳 4 筆資料 (對應 4 個人)
        features = assigner.preprocessor.create_inference_features([(task_id, difficulty)])
        features_scaled = assigner.preprocessor.transform(features)
        
        assigner.model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(features_scaled).to(DEVICE)
            predictions = assigner.model(x_tensor).cpu().numpy().flatten()
        
        # predictions 是一個長度為 4 的陣列，對應 CONFIG['people'] 的順序
        # 選出分數最高的人的索引
        best_person_idx = np.argmax(predictions)
        best_person = CONFIG['people'][best_person_idx]
        
        # 3. 實際執行 (由最佳人選執行，並更新成長狀態 - 這是 Ground Truth)
        execution_result = simulator.execute_task(best_person, task_id, difficulty, use_model=True)
        real_score = execution_result['final_score']
        
        # 記錄統計
        person_task_counts[best_person] += 1
        person_scores[best_person].append(real_score)
        person_task_type_counts[best_person][task_id] += 1 # 記錄工作類型
        person_task_type_scores[best_person][task_id] += real_score # 記錄工作分數
        total_score_sum += real_score
        
        # Optional: Print progress every 20 tasks
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/100 tasks...")

    # 4. 輸出統計結果
    print("\n--- Statistics Results ---")
    print(f"{'Person':<8} {'Count':<6} {'Total Score':<12} {'Avg Score':<10} {'Task Dist (Count)':<20} {'Avg Score per Task (T1/T2/T3/T4)'}")
    print("-" * 100)
    
    for person in CONFIG['people']:
        count = person_task_counts[person]
        scores = person_scores[person]
        total = sum(scores)
        avg = total / count if count > 0 else 0.0
        
        # 格式化任務分佈字串
        task_dist = person_task_type_counts[person]
        dist_str = f"{task_dist[1]}/{task_dist[2]}/{task_dist[3]}/{task_dist[4]}"
        
        # 計算每種工作的平均分數
        avg_scores_str_list = []
        for t in range(1, 5):
            t_count = person_task_type_counts[person][t]
            t_score = person_task_type_scores[person][t]
            t_avg = t_score / t_count if t_count > 0 else 0.0
            avg_scores_str_list.append(f"{t_avg:.1f}")
        avg_scores_str = "/".join(avg_scores_str_list)
        
        print(f"{person:<8} {count:<6} {total:<12.2f} {avg:<10.2f} {dist_str:<20} {avg_scores_str}")
        
    print("-" * 100)
    print(f"Total Tasks: 100")
    print(f"Grand Total Score: {total_score_sum:.2f}")
    print("="*60)
    
    return {
        'person_task_counts': person_task_counts,
        'person_scores': person_scores,
        'person_task_type_counts': person_task_type_counts,
        'person_task_type_scores': person_task_type_scores,
        'total_score_sum': total_score_sum
    }

def RandomSim(simulator, DEVICE, random_tasks):
    print("\n" + "="*60)
    print("RANDOM ASSIGNMENT & EXECUTION STATISTICS (100 TASKS)")
    print("="*60)
    
    # 統計變數
    person_task_counts = {p: 0 for p in CONFIG['people']}
    person_scores = {p: [] for p in CONFIG['people']}
    person_task_type_counts = {p: {t: 0 for t in range(1, 5)} for p in CONFIG['people']}
    # 新增: 統計每個人拿到每種工作的總分 {Person: {TaskID: TotalScore}}
    person_task_type_scores = {p: {t: 0.0 for t in range(1, 5)} for p in CONFIG['people']}
    total_score_sum = 0
    
    print("Processing 100 tasks using Random Assignment (25 tasks/person)...")
    
    # 準備隨機分配名單 (每人25個，確保平均分配)
    people = CONFIG['people']
    assignments = []
    for p in people:
        assignments.extend([p] * 25)
    np.random.shuffle(assignments)
    
    # 逐一處理每個任務
    for i, ((task_id, difficulty), person) in enumerate(zip(random_tasks, assignments)):
        
        # 實際執行 (使用模型預測分數)
        execution_result = simulator.execute_task(person, task_id, difficulty, use_model=True)
        real_score = execution_result['final_score']
        
        # 記錄統計
        person_task_counts[person] += 1
        person_scores[person].append(real_score)
        person_task_type_counts[person][task_id] += 1
        person_task_type_scores[person][task_id] += real_score # 記錄工作分數
        total_score_sum += real_score
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/100 tasks...")

    # 輸出統計結果
    print("\n--- Statistics Results ---")
    print(f"{'Person':<8} {'Count':<6} {'Total Score':<12} {'Avg Score':<10} {'Task Dist (Count)':<20} {'Avg Score per Task (T1/T2/T3/T4)'}")
    print("-" * 100)
    
    for person in CONFIG['people']:
        count = person_task_counts[person]
        scores = person_scores[person]
        total = sum(scores)
        avg = total / count if count > 0 else 0.0
        
        task_dist = person_task_type_counts[person]
        dist_str = f"{task_dist[1]}/{task_dist[2]}/{task_dist[3]}/{task_dist[4]}"
        
        # 計算每種工作的平均分數
        avg_scores_str_list = []
        for t in range(1, 5):
            t_count = person_task_type_counts[person][t]
            t_score = person_task_type_scores[person][t]
            t_avg = t_score / t_count if t_count > 0 else 0.0
            avg_scores_str_list.append(f"{t_avg:.1f}")
        avg_scores_str = "/".join(avg_scores_str_list)
        
        print(f"{person:<8} {count:<6} {total:<12.2f} {avg:<10.2f} {dist_str:<20} {avg_scores_str}")
        
    print("-" * 100)
    print(f"Total Tasks: 100")
    print(f"Grand Total Score: {total_score_sum:.2f}")
    print("="*60)
    
    return {
        'person_task_counts': person_task_counts,
        'person_scores': person_scores,
        'person_task_type_counts': person_task_type_counts,
        'person_task_type_scores': person_task_type_scores,
        'total_score_sum': total_score_sum
    }
    
    
def main():
    assigner, simulator, final_model, preprocessor, df, DEVICE = initial_train()

# Run multiple rounds to show growth effect
    #demo(assigner, simulator)
    save_model(final_model.state_dict(), preprocessor)
    simulator.reset_growth()

# Create data generator
    data_generator = SyntheticDataGenerator(assigner, simulator, CONFIG)

# Generate synthetic data
    synthetic_df = data_generator.generate_batch(n_rounds=100, use_model=True, verbose=True)

# Show sample of synthetic data
    print("\nSample of synthetic data:")
    print(synthetic_df[['person_id', 'task_id', 'task_difficulty', 'score', 'predicted_score', 'penalty_triggered']].head(10))

    # Save to CSV
    output_csv = 'synthetic_data.csv'
    synthetic_df.to_csv(output_csv, index=False)
    print(f"\nSynthetic data saved to '{output_csv}'")
    

    simulator.reset_growth()
    augmented_df = data_generator.augment_training_data(df, n_rounds=50, use_model=True)

# Reprocess augmented data
    X_aug, y_aug = preprocessor.create_features(augmented_df)
    preprocessor.fit_scaler(X_aug)
    X_aug_scaled = preprocessor.transform(X_aug)
    augmented_dataset = TaskDataset(X_aug_scaled, y_aug)

    # # Retrain primary model
    print("\n" + "="*60)
    print("RETRAINING PRIMARY MODEL WITH AUGMENTED DATA")
    print("="*60)

    trainer_aug = Trainer(CONFIG, DEVICE)
    new_model_state, new_fold_results, _, _ = trainer_aug.cross_validate(
    augmented_dataset, preprocessor
    )

    # # Update primary model
    final_model.load_state_dict(new_model_state)
    assigner = TaskAssigner(final_model, preprocessor, DEVICE)

    print("\nPrimary model retrained with simulator-generated data!")

    # Save the trained model (will also download it)
    save_model(final_model.state_dict(), preprocessor, simulator)
    
    np.random.seed(42)
    random_tasks = [
        (np.random.randint(1, 5), np.random.randint(1, 6)) 
        for _ in range(100)
    ]
    
    simulator.reset_growth()
    ai_stats = AISim(simulator, assigner, DEVICE, random_tasks)
    
    simulator.reset_growth()
    random_stats = RandomSim(simulator, DEVICE, random_tasks)
    
    # Visualize Comparison
    viz.visualize_simulation_comparison(ai_stats, random_stats)
if __name__ == "__main__":
    main()