import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from config import CONFIG, SIMULATOR_CONFIG

# Use default font settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

def get_task_name(task_id):
    """從設定檔獲取任務名稱的輔助函式"""
    return CONFIG['task_names'].get(task_id, f'Unknown Task {task_id}')

# ==============================================================================
# 文字報表輸出
# ==============================================================================

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

    people = CONFIG['people']
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

# ==============================================================================
# 圖表繪製功能
# ==============================================================================

def plot_results(predictions, targets, fold_results):
    """Create visualization plots for training metrics."""
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


def visualize_assignment_and_execution(assignment_result, execution_results):
    """
    視覺化任務分配和執行結果的完整圖表。
    """
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Score Matrix Heatmap (左上)
    ax1 = fig.add_subplot(2, 3, 1)
    score_matrix = assignment_result['score_matrix']
    task_slots = assignment_result['task_slots']
    
    im = ax1.imshow(score_matrix, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, ax=ax1, label='Predicted Score')
    
    people = CONFIG['people']
    task_labels = [f"T{t[0]}(D{t[1]})" for t in task_slots]
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(task_labels)
    ax1.set_yticks(range(4))
    ax1.set_yticklabels([f'Person {p}' for p in people])
    ax1.set_title('Predicted Score Matrix', fontsize=12, fontweight='bold')
    
    for i in range(4):
        for j in range(4):
            ax1.text(j, i, f'{score_matrix[i, j]:.1f}',
                    ha='center', va='center', color='black', fontsize=10)
    
    for assignment in assignment_result['assignments']:
        person_idx = people.index(assignment['person'])
        slot_idx = assignment['slot'] - 1
        rect = plt.Rectangle((slot_idx - 0.5, person_idx - 0.5), 1, 1,
                             fill=False, edgecolor='blue', linewidth=3)
        ax1.add_patch(rect)
    
    # 2. 分配連線圖 (中上)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_xlim(-0.5, 2.5)
    ax2.set_ylim(-0.5, 3.5)
    
    for i, person in enumerate(people):
        circle = plt.Circle((0.3, 3 - i), 0.2, color='#3498db', ec='black', lw=2)
        ax2.add_patch(circle)
        ax2.text(0.3, 3 - i, person, ha='center', va='center', 
                fontsize=14, fontweight='bold', color='white')
    
    task_colors = ['#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    for i, (task_id, diff) in enumerate(task_slots):
        task_name = CONFIG['task_names'].get(task_id, f'T{task_id}')[:6]
        circle = plt.Circle((1.7, 3 - i), 0.25, color=task_colors[task_id - 1], 
                            ec='black', lw=2)
        ax2.add_patch(circle)
        ax2.text(1.7, 3 - i, f'{task_name}\nD{diff}', ha='center', va='center', 
                fontsize=8, fontweight='bold', color='white')
    
    for assignment in assignment_result['assignments']:
        person_idx = people.index(assignment['person'])
        slot_idx = assignment['slot'] - 1
        ax2.annotate('', xy=(1.45, 3 - slot_idx), xytext=(0.5, 3 - person_idx),
                    arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
        mid_x = (0.5 + 1.45) / 2
        mid_y = (3 - person_idx + 3 - slot_idx) / 2
        ax2.text(mid_x, mid_y + 0.15, f'{assignment["predicted_score"]:.1f}',
                ha='center', va='center', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('Task Assignment Flow', fontsize=12, fontweight='bold')
    
    # 3. 預測 vs 實際分數比較 (右上)
    ax3 = fig.add_subplot(2, 3, 3)
    persons = [r['person'] for r in execution_results]
    predicted = [r['predicted_score'] for r in execution_results]
    actual = [r['final_score'] for r in execution_results]
    
    x = np.arange(len(persons))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, predicted, width, label='Predicted', color='#3498db', alpha=0.8)
    bars2 = ax3.bar(x + width/2, actual, width, label='Actual', color='#e74c3c', alpha=0.8)
    
    for i, result in enumerate(execution_results):
        if result['penalty_triggered']:
            ax3.annotate('⚠️', xy=(x[i] + width/2, actual[i] + 0.5),
                        fontsize=12, ha='center')
    
    ax3.set_xlabel('Person')
    ax3.set_ylabel('Score')
    ax3.set_title('Predicted vs Actual Scores', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'Person {p}' for p in persons])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 預測誤差分析 (左下)
    ax4 = fig.add_subplot(2, 3, 4)
    errors = [r['prediction_error'] for r in execution_results]
    colors = ['#27ae60' if e >= 0 else '#c0392b' for e in errors]
    
    bars = ax4.bar(x, errors, color=colors, alpha=0.8, edgecolor='black')
    ax4.axhline(y=0, color='black', linestyle='-', lw=1)
    ax4.set_xlabel('Person')
    ax4.set_ylabel('Error (Actual - Predicted)')
    ax4.set_title('Prediction Errors', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'Person {p}' for p in persons])
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, err in zip(bars, errors):
        ypos = bar.get_height() + 0.2 if err >= 0 else bar.get_height() - 0.5
        ax4.text(bar.get_x() + bar.get_width()/2, ypos,
                f'{err:+.2f}', ha='center', va='bottom' if err >= 0 else 'top', fontsize=10)
    
    # 5. 技能 vs 難度雷達圖 (中下)
    ax5 = fig.add_subplot(2, 3, 5, projection='polar')
    categories = ['Skill Level', 'Difficulty', 'Experience', 'Score/10']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    for i, result in enumerate(execution_results):
        values = [
            result['relevant_skill'],
            result['difficulty'],
            result['experience_factor'] * 3,
            result['final_score'] / 10
        ]
        values += values[:1]
        ax5.plot(angles, values, 'o-', linewidth=2, label=f"Person {result['person']}")
        ax5.fill(angles, values, alpha=0.15)
    
    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(categories)
    ax5.set_title('Person Performance Radar', fontsize=12, fontweight='bold', pad=20)
    ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # 6. 執行摘要表格 (右下)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    table_data = []
    headers = ['Person', 'Task', 'Diff', 'Skill', 'Pred', 'Actual', 'Error', 'Penalty']
    
    for r in execution_results:
        penalty_str = '⚠️ Yes' if r['penalty_triggered'] else '✓ No'
        table_data.append([
            r['person'], r['task_name'][:8], r['difficulty'],
            f"{r['relevant_skill']:.2f}", f"{r['predicted_score']:.1f}",
            f"{r['final_score']:.1f}", f"{r['prediction_error']:+.2f}", penalty_str
        ])
    
    total_pred = sum(r['predicted_score'] for r in execution_results)
    total_actual = sum(r['final_score'] for r in execution_results)
    total_error = total_actual - total_pred
    table_data.append(['TOTAL', '-', '-', '-', f'{total_pred:.1f}', f'{total_actual:.1f}', f'{total_error:+.2f}', '-'])
    
    table = ax6.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center', colColours=['#3498db']*8)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    for i in range(len(headers)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    for i in range(len(headers)):
        table[(len(table_data), i)].set_facecolor('#ecf0f1')
        table[(len(table_data), i)].set_text_props(fontweight='bold')
    
    ax6.set_title('Execution Summary', fontsize=12, fontweight='bold', pad=20)
    plt.suptitle('Task Assignment & Execution Visualization', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    return fig

def visualize_growth_over_rounds(growth_history):
    """視覺化多輪執行後的技能成長。"""
    if not growth_history:
        print("No growth history to visualize.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    people = CONFIG['people']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    ax1 = axes[0]
    for i, person in enumerate(people):
        if person in growth_history[0]:
            skills_over_time = [round_data[person]['avg_skill'] for round_data in growth_history]
            ax1.plot(range(len(skills_over_time)), skills_over_time, 
                    'o-', color=colors[i], label=f'Person {person}', linewidth=2)
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Average Skill Level')
    ax1.set_title('Skill Growth Over Rounds', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    for i, person in enumerate(people):
        if person in growth_history[0]:
            cumulative_scores = []
            total = 0
            for round_data in growth_history:
                total += round_data[person].get('score', 0)
                cumulative_scores.append(total)
            ax2.plot(range(len(cumulative_scores)), cumulative_scores, 
                    'o-', color=colors[i], label=f'Person {person}', linewidth=2)
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Cumulative Score')
    ax2.set_title('Cumulative Scores Over Rounds', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    return fig

def visualize_penalty_analysis(execution_history):
    """視覺化 penalty 觸發情況的分析。"""
    all_results = [r for round_results in execution_history for r in round_results]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1 = axes[0]
    penalty_counts = sum(1 for r in all_results if r['penalty_triggered'])
    no_penalty_counts = len(all_results) - penalty_counts
    ax1.pie([no_penalty_counts, penalty_counts], labels=['No Penalty', 'Penalty Triggered'],
           autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], explode=(0, 0.1), shadow=True)
    ax1.set_title('Penalty Trigger Rate', fontsize=12, fontweight='bold')
    
    ax2 = axes[1]
    difficulties = [r['difficulty'] for r in all_results]
    skills = [r['relevant_skill'] for r in all_results]
    penalties = [r['penalty_triggered'] for r in all_results]
    colors = ['#e74c3c' if p else '#2ecc71' for p in penalties]
    ax2.scatter(difficulties, skills, c=colors, alpha=0.7, s=100, edgecolors='black')
    
    ax2.axhline(y=SIMULATOR_CONFIG['low_skill_threshold'], color='orange', linestyle='--', label='Skill Threshold')
    ax2.axvline(x=SIMULATOR_CONFIG['high_difficulty_threshold'], color='purple', linestyle='--', label='Difficulty Threshold')
    ax2.fill_between([SIMULATOR_CONFIG['high_difficulty_threshold'], 6], 0, SIMULATOR_CONFIG['low_skill_threshold'], alpha=0.2, color='red', label='Penalty Zone')
    ax2.set_xlabel('Task Difficulty')
    ax2.set_ylabel('Relevant Skill Level')
    ax2.set_title('Difficulty vs Skill', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[2]
    penalty_scores = [r['final_score'] for r in all_results if r['penalty_triggered']]
    no_penalty_scores = [r['final_score'] for r in all_results if not r['penalty_triggered']]
    
    bp = ax3.boxplot([no_penalty_scores, penalty_scores] if penalty_scores else [no_penalty_scores],
                     labels=['No Penalty', 'With Penalty'][:2 if penalty_scores else 1], patch_artist=True)
    
    colors_box = ['#2ecc71', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors_box[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_ylabel('Final Score')
    ax3.set_title('Score Distribution by Penalty', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    return fig

def visualize_person_task_scores(assigner, difficulties=None):
    """視覺化每個人做四種工作的預測分數折線圖。"""
    if difficulties is None:
        difficulties = [3, 3, 3, 3]
    
    people = CONFIG['people']
    task_ids = [1, 2, 3, 4]
    task_names = [CONFIG['task_names'].get(t, f'Task {t}') for t in task_ids]
    task_slots = [(task_id, difficulties[i]) for i, task_id in enumerate(task_ids)]
    
    score_matrix = assigner.predict_scores(task_slots)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    markers = ['o', 's', '^', 'D']
    
    # 圖1: 每個人的任務分數折線圖
    ax1 = axes[0]
    x = np.arange(len(task_ids))
    for i, person in enumerate(people):
        scores = score_matrix[i, :]
        ax1.plot(x, scores, marker=markers[i], color=colors[i], 
                linewidth=2.5, markersize=10, label=f'Person {person}')
        for j, score in enumerate(scores):
            ax1.annotate(f'{score:.1f}', xy=(x[j], score), xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9, fontweight='bold', color=colors[i])
    ax1.set_xlabel('Task Type')
    ax1.set_ylabel('Predicted Score')
    ax1.set_title(f'Predicted Scores by Person (Diff: {difficulties})', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(task_names, rotation=15, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 圖2: 每個任務的人員分數折線圖
    ax2 = axes[1]
    task_colors = ['#e74c3c', '#9b59b6', '#3498db', '#1abc9c']
    x_people = np.arange(len(people))
    for j, task_id in enumerate(task_ids):
        scores = score_matrix[:, j]
        ax2.plot(x_people, scores, marker='o', color=task_colors[j], 
                linewidth=2.5, markersize=10, label=task_names[j])
        for i, score in enumerate(scores):
            ax2.annotate(f'{score:.1f}', xy=(x_people[i], score), xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9, fontweight='bold', color=task_colors[j])
    ax2.set_xlabel('Person')
    ax2.set_title(f'Predicted Scores by Task (Diff: {difficulties})', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_people)
    ax2.set_xticklabels([f'Person {p}' for p in people])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    return fig, score_matrix

def visualize_person_task_scores_multi_difficulty(assigner):
    """視覺化每個人做四種工作在不同難度下的預測分數。"""
    people = CONFIG['people']
    task_ids = [1, 2, 3, 4]
    difficulty_levels = [1, 2, 3, 4, 5]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    colors = ['#e74c3c', '#9b59b6', '#3498db', '#1abc9c']
    
    for person_idx, person in enumerate(people):
        ax = axes[person_idx]
        for task_idx, task_id in enumerate(task_ids):
            scores_by_difficulty = []
            for diff in difficulty_levels:
                task_slots = [(task_id, diff), (1, 1), (2, 1), (3, 1)]
                score_matrix = assigner.predict_scores(task_slots)
                scores_by_difficulty.append(score_matrix[person_idx, 0])
            
            task_name = CONFIG['task_names'].get(task_id, f'Task {task_id}')
            ax.plot(difficulty_levels, scores_by_difficulty, marker='o', color=colors[task_idx], 
                   linewidth=2, markersize=8, label=task_name)
        
        ax.set_xlabel('Difficulty Level')
        ax.set_ylabel('Predicted Score')
        ax.set_title(f'Person {person}: Score vs Difficulty', fontsize=12, fontweight='bold')
        ax.set_xticks(difficulty_levels)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    
    plt.suptitle('Predicted Scores Across Different Difficulties', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    return fig

def visualize_person_task_heatmap(assigner, difficulty=3):
    """用熱力圖顯示每個人做每個任務的預測分數。"""
    people = CONFIG['people']
    task_ids = [1, 2, 3, 4]
    task_names = [CONFIG['task_names'].get(t, f'Task {t}') for t in task_ids]
    task_slots = [(task_id, difficulty) for task_id in task_ids]
    score_matrix = assigner.predict_scores(task_slots)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(score_matrix, cmap='RdYlGn', aspect='auto')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Predicted Score')
    
    ax.set_xticks(np.arange(len(task_names)))
    ax.set_yticks(np.arange(len(people)))
    ax.set_xticklabels(task_names)
    ax.set_yticklabels([f'Person {p}' for p in people])
    
    for i in range(len(people)):
        for j in range(len(task_ids)):
            score = score_matrix[i, j]
            text_color = 'white' if score < np.mean(score_matrix) else 'black'
            ax.text(j, i, f'{score:.1f}', ha='center', va='center', 
                   fontsize=14, fontweight='bold', color=text_color)
    
    # Highlight best choices
    for i in range(len(people)):
        best_task = np.argmax(score_matrix[i, :])
        rect = plt.Rectangle((best_task - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='blue', linewidth=3)
        ax.add_patch(rect)
    
    ax.set_title(f'Person-Task Score Matrix (Difficulty: {difficulty})', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
    return fig, score_matrix

def visualize_simulation_comparison(ai_stats, random_stats):
    """
    Visualize the comparison between AI Simulation and Random Simulation.
    
    Args:
        ai_stats (dict): Statistics from AISim.
        random_stats (dict): Statistics from RandomSim.
    """
    people = CONFIG['people']
    task_ids = [1, 2, 3, 4]
    task_names = [CONFIG['task_names'].get(t, f'Task {t}') for t in task_ids]
    
    # ==========================================================================
    # 1. Per Person Plots (8 plots: 4 people * 2 types)
    # ==========================================================================
    
    # Plot Type A: Task Counts per Person (AI vs Random)
    fig_counts, axes_counts = plt.subplots(2, 2, figsize=(14, 10))
    axes_counts = axes_counts.flatten()
    
    for idx, person in enumerate(people):
        ax = axes_counts[idx]
        
        ai_counts = [ai_stats['person_task_type_counts'][person][t] for t in task_ids]
        rand_counts = [random_stats['person_task_type_counts'][person][t] for t in task_ids]
        
        x = np.arange(len(task_ids))
        width = 0.35
        
        ax.bar(x - width/2, ai_counts, width, label='AI Sim', color='#3498db', alpha=0.8)
        ax.bar(x + width/2, rand_counts, width, label='Random Sim', color='#e74c3c', alpha=0.8)
        
        ax.set_ylabel('Task Count')
        ax.set_title(f'Person {person}: Task Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(task_names)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
    plt.suptitle('Task Count Distribution per Person (AI vs Random)', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Plot Type B: Average Scores per Task per Person (AI vs Random)
    fig_scores, axes_scores = plt.subplots(2, 2, figsize=(14, 10))
    axes_scores = axes_scores.flatten()
    
    for idx, person in enumerate(people):
        ax = axes_scores[idx]
        
        ai_avg_scores = []
        rand_avg_scores = []
        
        for t in task_ids:
            # AI
            c = ai_stats['person_task_type_counts'][person][t]
            s = ai_stats['person_task_type_scores'][person][t]
            ai_avg_scores.append(s / c if c > 0 else 0)
            
            # Random
            c_r = random_stats['person_task_type_counts'][person][t]
            s_r = random_stats['person_task_type_scores'][person][t]
            rand_avg_scores.append(s_r / c_r if c_r > 0 else 0)
            
        x = np.arange(len(task_ids))
        
        # Using Line Chart as requested ("折線圖")
        ax.plot(x, ai_avg_scores, marker='o', linewidth=2, label='AI Sim', color='#3498db')
        ax.plot(x, rand_avg_scores, marker='s', linewidth=2, label='Random Sim', color='#e74c3c', linestyle='--')
        
        ax.set_ylabel('Average Score')
        ax.set_title(f'Person {person}: Average Score per Task')
        ax.set_xticks(x)
        ax.set_xticklabels(task_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.suptitle('Average Score per Task per Person (AI vs Random)', fontsize=16)
    plt.tight_layout()
    plt.show()

    # ==========================================================================
    # 2. Overall Comparison (4 plots)
    # ==========================================================================
    fig_overall, axes_overall = plt.subplots(2, 2, figsize=(14, 10))
    axes_overall = axes_overall.flatten()
    
    # Plot 1: Total Score Comparison
    ax1 = axes_overall[0]
    sim_types = ['AI Sim', 'Random Sim']
    total_scores = [ai_stats['total_score_sum'], random_stats['total_score_sum']]
    bars = ax1.bar(sim_types, total_scores, color=['#3498db', '#e74c3c'], width=0.5)
    ax1.set_title('Total Score Comparison')
    ax1.set_ylabel('Total Score')
    ax1.bar_label(bars, fmt='%.0f')
    
    # Plot 2: Average Score Comparison (Global)
    ax2 = axes_overall[1]
    # Calculate global average
    ai_global_avg = ai_stats['total_score_sum'] / 100
    rand_global_avg = random_stats['total_score_sum'] / 100
    avgs = [ai_global_avg, rand_global_avg]
    bars2 = ax2.bar(sim_types, avgs, color=['#3498db', '#e74c3c'], width=0.5)
    ax2.set_title('Global Average Score Comparison')
    ax2.set_ylabel('Average Score')
    ax2.bar_label(bars2, fmt='%.2f')
    
    # Plot 4: Total Tasks Assigned per Person
    ax4 = axes_overall[3]
    ai_person_counts = [ai_stats['person_task_counts'][p] for p in people]
    rand_person_counts = [random_stats['person_task_counts'][p] for p in people]
    
    x_p = np.arange(len(people))
    ax4.bar(x_p - width/2, ai_person_counts, width, label='AI Sim', color='#3498db')
    ax4.bar(x_p + width/2, rand_person_counts, width, label='Random Sim', color='#e74c3c')
    ax4.set_title('Total Tasks Assigned per Person')
    ax4.set_xticks(x_p)
    ax4.set_xticklabels([f'P{p}' for p in people])
    ax4.legend()
    
    plt.suptitle('Overall Simulation Comparison', fontsize=16)
    plt.tight_layout()
    plt.show()

    # ==========================================================================
    # 3. Total Score per Person Comparison (1 plot)
    # ==========================================================================
    fig_person_total, ax_pt = plt.subplots(figsize=(10, 6))
    
    ai_person_totals = [sum(ai_stats['person_scores'][p]) for p in people]
    rand_person_totals = [sum(random_stats['person_scores'][p]) for p in people]
    
    x = np.arange(len(people))
    width = 0.35
    
    rects1 = ax_pt.bar(x - width/2, ai_person_totals, width, label='AI Sim', color='#3498db')
    rects2 = ax_pt.bar(x + width/2, rand_person_totals, width, label='Random Sim', color='#e74c3c')
    
    ax_pt.set_ylabel('Total Score')
    ax_pt.set_title('Total Score per Person (AI vs Random)')
    ax_pt.set_xticks(x)
    ax_pt.set_xticklabels([f'Person {p}' for p in people])
    ax_pt.legend()
    
    ax_pt.bar_label(rects1, padding=3, fmt='%.0f')
    ax_pt.bar_label(rects2, padding=3, fmt='%.0f')
    
    plt.tight_layout()
    plt.show()