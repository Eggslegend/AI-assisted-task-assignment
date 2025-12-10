import pickle
import torch
import numpy as np
import sys
import os

# Add current directory to path so we can import if needed, though for simple inspection we might not need to
sys.path.append(os.getcwd())

def inspect_pkl(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    print(f"Loading {filepath}...\n")
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print("="*50)
        print("PICKLE FILE CONTENT STRUCTURE")
        print("="*50)
        
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"\nKey: '{key}'")
                print(f"Type: {type(value)}")
                
                if key == 'model_state':
                    print("Description: PyTorch Model Weights (State Dict)")
                    print(f"Layers: {list(value.keys())}")
                    # Print shapes of weights
                    for layer_name, tensor in value.items():
                        print(f"  - {layer_name}: {tensor.shape}")
                        
                elif key == 'skill_profiles':
                    print("Description: Person Skill Profiles")
                    print(f"Content: {value}")
                    
                elif key == 'scaler_mean':
                    print("Description: StandardScaler Mean (for normalization)")
                    print(f"Values: {value}")
                    
                elif key == 'scaler_scale':
                    print("Description: StandardScaler Scale (Variance/StdDev)")
                    print(f"Values: {value}")
                    
                elif key == 'person_to_idx':
                    print("Description: Person Name to Index Mapping")
                    print(f"Mapping: {value}")
                    
                elif key == 'config':
                    print("Description: Configuration Dictionary")
                    # Print a few config items
                    print(f"Keys: {list(value.keys())}")
                
                else:
                    print(f"Value: {value}")
        else:
            print(f"Root object type: {type(data)}")
            print(data)
            
    except Exception as e:
        print(f"Error reading pickle file: {e}")

if __name__ == "__main__":
    inspect_pkl('task_assignment_model.pkl')
