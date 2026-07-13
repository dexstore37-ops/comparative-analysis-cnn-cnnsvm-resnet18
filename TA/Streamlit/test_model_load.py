#!/usr/bin/env python3
"""
Script test untuk memverifikasi model loading tanpa error
"""
import os
import sys
import tensorflow as tf
from tensorflow import keras
import joblib

# Compatibility wrapper untuk GlorotUniform
class CompatGlorotUniform(keras.initializers.GlorotUniform):
    """Wrapper untuk menangani backward compatibility GlorotUniform dengan parameter lama"""
    def __init__(self, seed=None, distribution='truncated_normal', **kwargs):
        super().__init__(seed=seed, distribution=distribution)

    @classmethod
    def from_config(cls, config):
        config_copy = config.copy()
        config_copy.pop('input_axes', None)
        config_copy.pop('output_axes', None)
        return cls(**config_copy)

# Path ke model-model
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

custom_objects = {
    'GlorotUniform': CompatGlorotUniform,
    'CompatGlorotUniform': CompatGlorotUniform
}

print("=" * 80)
print("TEST MODEL LOADING")
print("=" * 80)

for model_name, paths_dict in MODEL_PATHS.items():
    print(f"\n[{model_name}]")
    
    try:
        if model_name == "CNN-SVM":
            model_path = paths_dict["model"]
            print(f"  Model path: {model_path}")
            print(f"  Exists: {os.path.exists(model_path)}")
            
            model = keras.models.load_model(
                model_path,
                compile=False,
                custom_objects=custom_objects,
                safe_mode=False
            )
            print(f"  ✓ CNN model loaded successfully")
            
            svm_path = paths_dict.get("svm")
            if svm_path and os.path.exists(svm_path):
                svm = joblib.load(svm_path)
                print(f"  ✓ SVM model loaded successfully")
            
            scaler_path = paths_dict.get("scaler")
            if scaler_path and os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                print(f"  ✓ Scaler loaded successfully")
        else:
            model_path = paths_dict["model"]
            print(f"  Model path: {model_path}")
            print(f"  Exists: {os.path.exists(model_path)}")
            
            model = keras.models.load_model(
                model_path,
                compile=False,
                custom_objects=custom_objects,
                safe_mode=False
            )
            print(f"  ✓ Model loaded successfully")
    
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
