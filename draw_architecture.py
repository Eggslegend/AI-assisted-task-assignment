import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np

def draw_flowchart():
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # --- Shape Helpers ---

    def draw_text(x, y, w, h, text):
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9, wrap=True)

    def draw_terminal(x, y, w, h, text, color='#eceff1', edge='black'):
        # Stadium shape / Rounded Rectangle
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=2", 
                                     linewidth=1.5, edgecolor=edge, facecolor=color)
        ax.add_patch(box)
        draw_text(x, y, w, h, text)
        return (x + w/2, y, x + w/2, y + h)

    def draw_process(x, y, w, h, text, color='#ffffff', edge='black'):
        # Rectangle
        rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=edge, facecolor=color)
        ax.add_patch(rect)
        draw_text(x, y, w, h, text)
        return (x + w/2, y, x + w/2, y + h)
    
    def draw_predefined_process(x, y, w, h, text, color='#ffffff', edge='black'):
        # Rectangle with double vertical lines
        rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=edge, facecolor=color)
        ax.add_patch(rect)
        # Inner lines
        ax.plot([x + w*0.1, x + w*0.1], [y, y+h], color=edge, linewidth=1.5)
        ax.plot([x + w*0.9, x + w*0.9], [y, y+h], color=edge, linewidth=1.5)
        draw_text(x, y, w, h, text)
        return (x + w/2, y, x + w/2, y + h)

    def draw_io(x, y, w, h, text, color='#ffffff', edge='black'):
        # Parallelogram
        skew = w * 0.15
        verts = [
            (x + skew, y),          # Bottom-left
            (x + w, y),             # Bottom-right
            (x + w - skew, y + h),  # Top-right
            (x, y + h),             # Top-left
            (x + skew, y),          # Close
        ]
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor=color, edgecolor=edge, linewidth=1.5)
        ax.add_patch(patch)
        draw_text(x, y, w, h, text)
        return (x + w/2, y, x + w/2, y + h)

    def draw_database(x, y, w, h, text, color='#ffffff', edge='black'):
        # Cylinder
        # Top ellipse
        ellipse_h = h * 0.2
        
        # Body (Rectangle part)
        rect = patches.Rectangle((x, y + ellipse_h/2), w, h - ellipse_h, linewidth=0, facecolor=color)
        ax.add_patch(rect)
        
        # Bottom ellipse (full)
        bottom_ellipse = patches.Ellipse((x + w/2, y + ellipse_h/2), w, ellipse_h, facecolor=color, edgecolor=edge, linewidth=1.5)
        ax.add_patch(bottom_ellipse)
        
        # Side lines
        ax.plot([x, x], [y + ellipse_h/2, y + h - ellipse_h/2], color=edge, linewidth=1.5)
        ax.plot([x + w, x + w], [y + ellipse_h/2, y + h - ellipse_h/2], color=edge, linewidth=1.5)
        
        # Top ellipse (full)
        top_ellipse = patches.Ellipse((x + w/2, y + h - ellipse_h/2), w, ellipse_h, facecolor=color, edgecolor=edge, linewidth=1.5)
        ax.add_patch(top_ellipse)
        
        draw_text(x, y, w, h, text)
        return (x + w/2, y, x + w/2, y + h)

    def draw_document(x, y, w, h, text, color='#ffffff', edge='black'):
        # Rectangle with wavy bottom
        wave_h = h * 0.1
        
        verts = [
            (x, y + h),             # Top-left
            (x + w, y + h),         # Top-right
            (x + w, y + wave_h),    # Bottom-right start
            # Wave approximation (Bezier)
            (x + w * 0.75, y - wave_h), # Control point 1
            (x + w * 0.5, y + wave_h),  # Mid point
            (x + w * 0.25, y + wave_h*2), # Control point 2
            (x, y + wave_h),        # Bottom-left end
            (x, y + h),             # Close
        ]
        # Simplified wave: Line to bottom-right, then curve to bottom-left
        # Let's use a simpler path for the wave
        verts = [
            (x, y + h),
            (x + w, y + h),
            (x + w, y + wave_h),
            (x + w * 0.66, y - wave_h),
            (x + w * 0.33, y + 2*wave_h),
            (x, y),
            (x, y + h)
        ]
        
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
        
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor=color, edgecolor=edge, linewidth=1.5)
        ax.add_patch(patch)
        draw_text(x, y + wave_h, w, h - wave_h, text) # Adjust text pos
        return (x + w/2, y, x + w/2, y + h)

    def draw_arrow(x1, y1, x2, y2, connectionstyle="arc3,rad=0"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color='black', connectionstyle=connectionstyle))

    # --- Main Execution Flow (Left Column) ---
    
    # Start
    draw_terminal(5, 92, 20, 4, "Start: main()")
    draw_arrow(15, 92, 15, 88)
    
    # Initial Training Phase (Predefined Process as it calls other modules)
    draw_predefined_process(5, 84, 20, 4, "initial_train()")
    draw_arrow(15, 84, 15, 80)

    # 1. Setup
    draw_process(5, 76, 20, 4, "1. Setup & Config")
    draw_arrow(15, 76, 15, 72)

    # 2. Preprocessing
    draw_process(5, 68, 20, 4, "2. Preprocessing")
    draw_arrow(15, 68, 15, 64)

    # 3. Train Main
    draw_process(5, 60, 20, 4, "3. Train Main Model")
    draw_arrow(15, 60, 15, 56)

    # 4. Train Sim
    draw_process(5, 52, 20, 4, "4. Train Simulator")
    draw_arrow(15, 52, 15, 48)

    # Return Models
    draw_process(5, 44, 20, 4, "Return Models")
    draw_arrow(15, 44, 15, 40)

    # Save Initial Model (IO)
    draw_io(5, 36, 20, 4, "Save Initial Model (.pkl)")
    draw_arrow(15, 36, 15, 32)

    # Generate Synthetic Data
    draw_process(5, 28, 20, 4, "Generate Synthetic Data")
    draw_arrow(15, 28, 15, 24)

    # Save CSV (Document)
    draw_document(5, 20, 20, 4, "Save synthetic_data.csv")
    draw_arrow(15, 20, 15, 16)

    # Augment Data
    draw_process(5, 12, 20, 4, "Augment Training Data")
    draw_arrow(15, 12, 15, 8)

    # Retrain
    draw_process(5, 4, 20, 4, "Retrain Primary Model")
    
    # --- Data & Objects (Right Side) ---
    
    # Data Sources (Database)
    draw_database(35, 70, 15, 6, "Training_data.xlsx")
    draw_arrow(35, 73, 25, 70) # Excel -> Preprocessing

    # Models (Process/Component)
    draw_process(35, 60, 15, 4, "ScorePredictor")
    draw_arrow(25, 62, 35, 62) # Train Main -> Predictor

    draw_process(35, 52, 15, 4, "TaskExecutionSimulator")
    draw_arrow(25, 54, 35, 54) # Train Sim -> Simulator

    # Synthetic Data Flow
    draw_process(35, 28, 15, 4, "SyntheticDataGenerator")
    draw_arrow(25, 30, 35, 30) # Gen Syn -> Generator
    
    # Dataframes (IO)
    draw_io(60, 28, 15, 4, "synthetic_df")
    draw_arrow(50, 30, 60, 30) # Generator -> synthetic_df
    draw_arrow(67.5, 28, 25, 22, connectionstyle="arc3,rad=-0.2") # synthetic_df -> Save CSV

    # Augmentation Flow
    draw_io(35, 12, 15, 4, "augmented_df")
    draw_arrow(25, 14, 35, 14) # Augment -> augmented_df
    
    draw_arrow(42.5, 12, 25, 6) # augmented_df -> Retrain

    # Final Save (IO)
    draw_io(35, 4, 15, 4, "Save Retrained Model")
    draw_arrow(25, 6, 35, 6) # Retrain -> Save

    plt.title("Program Structure: Training, Simulation & Retraining Pipeline", fontsize=16)
    plt.tight_layout()
    plt.savefig('program_structure.png', dpi=300, bbox_inches='tight')
    print("Program structure diagram saved to 'program_structure.png'")
    plt.show()

if __name__ == "__main__":
    draw_flowchart()
