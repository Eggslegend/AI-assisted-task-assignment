import torch
import numpy as np
import pickle
from config import CONFIG, DEVICE_MODE
import matplotlib.pyplot as plt
import pandas as pd

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    if DEVICE_MODE == 'cpu':
        return torch.device('cpu')
    elif DEVICE_MODE == 'cuda':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_model(model_state, preprocessor, simulator=None, filepath='task_assignment_model.pkl'):
    save_dict = {
        'model_state': model_state,
        'skill_profiles': preprocessor.skill_profiles,
        'scaler_mean': preprocessor.scaler.mean_,
        'scaler_scale': preprocessor.scaler.scale_,
        'person_to_idx': preprocessor.person_to_idx,
        'config': preprocessor.config
    }
    
    # 如果有提供 simulator，也一併儲存
    if simulator is not None:
        save_dict['simulator_state'] = {
            'model_state': simulator.model.state_dict(),
            'scaler_mean': simulator.scaler.mean_ if hasattr(simulator.scaler, 'mean_') else None,
            'scaler_scale': simulator.scaler.scale_ if hasattr(simulator.scaler, 'scale_') else None,
            'sim_config': simulator.sim_config,
            # 儲存成長狀態 (如果需要恢復當前進度)
            'growth_skills': simulator.growth_tracker.current_skills,
            'growth_history': simulator.growth_tracker.task_history,
            'growth_counts': simulator.growth_tracker.total_tasks_completed
        }
        print("Simulator state included in save file.")

    with open(filepath, 'wb') as f:
        pickle.dump(save_dict, f)
    print(f"Model saved to {filepath}")

def load_model(filepath, model_class, preprocessor_class):
    # Note: Requires classes to be passed in to avoid circular imports
    with open(filepath, 'rb') as f:
        save_dict = pickle.load(f)
    
    preprocessor = preprocessor_class(save_dict['config'])
    preprocessor.restore_state(save_dict)
    
    input_dim = len(save_dict['scaler_mean'])
    model = model_class(
        input_dim=input_dim,
        hidden_layers=save_dict['config']['hidden_layers'],
        dropout_rate=save_dict['config']['dropout_rate']
    )
    model.load_state_dict(save_dict['model_state'])
    return model, preprocessor

def get_task_name(task_id):
    """Get task name from task ID."""
    return CONFIG['task_names'].get(task_id, f'Unknown Task {task_id}')


def run_inference(assigner, task_slots=None):
    """
    Run inference with given or random task slots.

    Args:
        assigner: TaskAssigner instance
        task_slots: Optional list of (task_id, difficulty) tuples.
                   Task IDs can repeat! e.g., [(1, 3), (1, 5), (2, 2), (4, 4)]
    """
    if task_slots is None:
        task_slots = [
            (np.random.randint(1, 5), np.random.randint(1, 6))
            for _ in range(4)
        ]

    return assigner.find_optimal_assignment(task_slots)


def print_assignment_result(result):
    """Pretty print assignment results with task names."""
    print("\n" + "="*70)
    print("TASK ASSIGNMENT RESULT")
    print("="*70)

    print("\nInput Task Slots:")
    for i, (task_id, diff) in enumerate(result['task_slots']):
        task_name = get_task_name(task_id)
        print(f"  Slot {i+1}: {task_name} (Difficulty: {diff})")

    print("\nPredicted Score Matrix:")
    # Header with task names (abbreviated for display)
    print("         ", end="")
    for i in range(4):
        task_id, diff = result['task_slots'][i]
        task_name = get_task_name(task_id)[:8]  # Truncate for display
        print(f"  {task_name:>10}(D{diff})", end="")
    print()

    people = ['A', 'B', 'C', 'D']
    for i, person in enumerate(people):
        print(f"Person {person}: ", end="")
        for j in range(4):
            print(f"     {result['score_matrix'][i,j]:6.1f}     ", end="")
        print()

    print("\nOptimal Assignment:")
    for assignment in result['assignments']:
        task_name = get_task_name(assignment['task_id'])
        print(f"  Person {assignment['person']} → Slot {assignment['slot']} "
              f"({task_name}, Difficulty {assignment['difficulty']}) "
              f"| Predicted Score: {assignment['predicted_score']:.1f}")

    print(f"\nTotal Predicted Score: {result['total_score']:.1f}")
    print("="*70)


def plot_results(predictions, targets, fold_results):
    """Create visualization plots."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Actual vs Predicted
    ax1 = axes[0]
    ax1.scatter(targets, predictions, alpha=0.5, edgecolors='black', linewidth=0.5)
    ax1.plot([targets.min(), targets.max()], [targets.min(), targets.max()],
             'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Score')
    ax1.set_ylabel('Predicted Score')
    ax1.set_title('Actual vs Predicted Scores')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Residual plot
    ax2 = axes[1]
    residuals = predictions.flatten() - targets.flatten()
    ax2.scatter(predictions, residuals, alpha=0.5, edgecolors='black', linewidth=0.5)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted Score')
    ax2.set_ylabel('Residual (Predicted - Actual)')
    ax2.set_title('Residual Plot')
    ax2.grid(True, alpha=0.3)

    # 3. CV metrics
    ax3 = axes[2]
    results_df = pd.DataFrame(fold_results)
    x = np.arange(len(fold_results))
    width = 0.35

    ax3.bar(x - width/2, results_df['rmse'], width, label='RMSE', color='steelblue')
    ax3.bar(x + width/2, results_df['mae'], width, label='MAE', color='darkorange')

    ax3.set_xlabel('Fold')
    ax3.set_ylabel('Error')
    ax3.set_title('Cross-Validation Metrics by Fold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'Fold {i+1}' for i in range(len(fold_results))])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()
    
def print_execution_results(execution_results):
    """Pretty print execution results from simulator."""
    print("\n" + "="*70)
    print("TASK EXECUTION RESULTS (Simulator)")
    print("="*70)

    total_actual = 0
    total_predicted = 0

    for result in execution_results:
        penalty_str = " [PENALTY]" if result['penalty_triggered'] else ""
        print(f"\n  Person {result['person']} → {result['task_name']} (Difficulty: {result['difficulty']})")
        print(f"    Relevant Skill: {result['relevant_skill']:.2f} | Experience: {result['experience_factor']:.3f}")
        print(f"    Predicted Score: {result['predicted_score']:.2f}")
        print(f"    Actual Score:    {result['final_score']:.2f}{penalty_str}")
        print(f"    Error:           {result['prediction_error']:+.2f}")

        total_actual += result['final_score']
        total_predicted += result['predicted_score']

    print("\n" + "-"*50)
    print(f"  Total Predicted: {total_predicted:.2f}")
    print(f"  Total Actual:    {total_actual:.2f}")
    print(f"  Total Error:     {total_actual - total_predicted:+.2f}")
    print("="*70)
    