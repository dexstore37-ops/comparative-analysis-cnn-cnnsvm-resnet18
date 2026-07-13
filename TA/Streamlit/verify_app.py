#!/usr/bin/env python3
"""
Verify Streamlit app dapat berjalan dengan 2 model utama (CNN dan CNN-SVM)
"""
import sys
import os
sys.path.insert(0, '/workspace/TA/Streamlit')

# Set environment untuk Streamlit testing
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'

# Import app module
try:
    import app
    print("✓ app.py imported successfully")
    
    # Test model loading
    print("\nLoading models...")
    models, model_status = app.load_models()
    
    print("\n" + "="*60)
    print("MODEL LOADING STATUS")
    print("="*60)
    
    for model_name, status in model_status.items():
        if status["loaded"]:
            print(f"✓ {model_name}: BERHASIL DIMUAT")
        else:
            print(f"✗ {model_name}: GAGAL - {status['error'][:80]}")
    
    models_available = [name for name, status in model_status.items() if status["loaded"]]
    print(f"\nTotal model berhasil: {len(models_available)}/{len(model_status)}")
    
    if len(models_available) >= 2:
        print("\n✓ SIAP DIJALANKAN: Minimal 2 model berhasil dimuat!")
        sys.exit(0)
    else:
        print("\n✗ ERROR: Hanya 1 model atau lebih sedikit yang berhasil dimuat")
        sys.exit(1)

except Exception as e:
    print(f"✗ Error: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
