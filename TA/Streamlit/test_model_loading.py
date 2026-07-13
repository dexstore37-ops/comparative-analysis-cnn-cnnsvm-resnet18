#!/usr/bin/env python3
"""
Script untuk test apakah semua model dapat di-load dengan benar
"""

import os
import sys
import pickle
import joblib
import tensorflow as tf
from tensorflow import keras

# Model paths
MODEL_PATHS = {
    "CNN": {
        "model": "/workspace/TA/CNN_bestmodel/best_cnn_final.keras"
    },
    "CNN-SVM": {
        "model": "/workspace/TA/CNN-SVM_bestmodel/best_cnn_model.keras",
        "svm": "/workspace/TA/CNN-SVM_bestmodel/best_svm_model.pkl",
        "scaler": "/workspace/TA/CNN-SVM_bestmodel/scaler.pkl"
    },
    "ResNet18": {
        "model": "/workspace/TA/Resnet18_bestmodel/best_resnet18.keras"
    }
}

def test_model_loading():
    """Test loading semua model"""
    print("\n" + "="*70)
    print("Testing Model Loading...")
    print("="*70)
    
    for model_name, paths_dict in MODEL_PATHS.items():
        print(f"\n[{model_name}]")
        print("-" * 50)
        
        try:
            if model_name == "CNN-SVM":
                # Load CNN-SVM
                model_path = paths_dict["model"]
                svm_path = paths_dict.get("svm")
                scaler_path = paths_dict.get("scaler")
                
                print(f"Loading CNN model: {model_path}")
                if os.path.exists(model_path):
                    cnn_model = keras.models.load_model(model_path)
                    print(f"✓ CNN model loaded successfully")
                    print(f"  Input shape: {cnn_model.input_shape}")
                    print(f"  Output shape: {cnn_model.output_shape}")
                else:
                    print(f"✗ CNN model not found: {model_path}")
                    continue
                
                # Load SVM menggunakan joblib
                if svm_path and os.path.exists(svm_path):
                    print(f"Loading SVM model: {svm_path}")
                    try:
                        svm_model = joblib.load(svm_path)
                        print(f"✓ SVM model loaded successfully")
                        print(f"  SVM type: {type(svm_model).__name__}")
                    except Exception as e:
                        print(f"✗ Error loading SVM model: {e}")
                else:
                    print(f"⚠ SVM model not found (optional): {svm_path}")
                
                # Load Scaler menggunakan joblib
                if scaler_path and os.path.exists(scaler_path):
                    print(f"Loading Scaler: {scaler_path}")
                    try:
                        scaler = joblib.load(scaler_path)
                        print(f"✓ Scaler loaded successfully")
                        print(f"  Scaler type: {type(scaler).__name__}")
                    except Exception as e:
                        print(f"✗ Error loading scaler: {e}")
            
            else:
                # Load standard models
                model_path = paths_dict["model"]
                print(f"Loading model: {model_path}")
                if os.path.exists(model_path):
                    model = keras.models.load_model(model_path)
                    print(f"✓ Model loaded successfully")
                    print(f"  Input shape: {model.input_shape}")
                    print(f"  Output shape: {model.output_shape}")
                else:
                    print(f"✗ Model not found: {model_path}")
        
        except Exception as e:
            print(f"✗ Error loading {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("Model loading test completed!")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_model_loading()
