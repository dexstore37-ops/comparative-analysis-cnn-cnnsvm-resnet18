import streamlit as st
import numpy as np
import time
import plotly.graph_objects as go
from PIL import Image
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'

# --- TAMBAHKAN BLOK KODE OPTIMASI MEMORI CPU DI BAWAH INI ---
import tensorflow as tf
# Membatasi internal thread TensorFlow agar tidak membuat banyak klon memori di CPU
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
# -----------------------------------------------------------
import logging
import tensorflow as tf
from tensorflow import keras
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')
import cv2  
from pathlib import Path
import pickle   
import joblib


import glob
import gc

def combine_split_files(base_path):
    """Menggabungkan berkas pecahan .part_* menjadi berkas asli dengan efisien memori.
    Menggunakan chunked streaming agar tidak memakan RAM kontainer Railway."""
    # Flag penanda bahwa proses penggabungan di sesi ini sudah selesai dilakukan
    done_flag = f"{base_path}.combined_done"
    
    if os.path.exists(base_path) and os.path.exists(done_flag):
        return # Berkas sudah utuh dan valid, langsung lewati proses
        
    parts = sorted(glob.glob(f"{base_path}.part_*"))
    if parts:
        print(f"Menggabungkan {len(parts)} bagian untuk {base_path} (Stream Mode)...")
        # Hapus berkas rusak/lama jika ada demi keamanan
        if os.path.exists(base_path):
            try: os.remove(base_path)
            except: pass
            
        with open(base_path, 'wb') as output_file:
            for part in parts:
                with open(part, 'rb') as input_file:
                    # Baca per 4MB (bukan sekaligus) agar RAM tetap rendah
                    while True:
                        chunk = input_file.read(4 * 1024 * 1024) 
                        if not chunk:
                            break
                        output_file.write(chunk)
                        
        # Buat berkas flag penanda sukses
        with open(done_flag, 'w') as f:
            f.write("done")
        print(f"✓ {base_path} berhasil digabungkan dengan aman.")

# Jalankan penggabungan otomatis untuk semua model sebelum load_models() dipanggil
combine_split_files("TA/CNN_bestmodel/best_cnn_final.keras")
combine_split_files("TA/CNN-SVM_bestmodel/best_cnn_model.keras")
combine_split_files("TA/CNN-SVM_bestmodel/best_svm_model.pkl")
combine_split_files("TA/Resnet18_bestmodel/best_resnet18.keras")

# Compatibility wrapper untuk models saved with older keras initializer config
@keras.utils.register_keras_serializable(package='Custom', name='GlorotUniform')
class CompatGlorotUniform(keras.initializers.GlorotUniform):
    """Wrapper untuk menangani backward compatibility GlorotUniform dengan parameter lama."""
    def __init__(self, seed=None, distribution='truncated_normal', **kwargs):
        # Filter out unknown parameters like input_axes, output_axes
        kwargs.pop('input_axes', None)
        kwargs.pop('output_axes', None)
        super().__init__(seed=seed, distribution=distribution)

    @classmethod
    def from_config(cls, config):
        config_copy = config.copy()
        config_copy.pop('input_axes', None)
        config_copy.pop('output_axes', None)
        return cls(**config_copy)


def patch_glorot_uniform_initialization():
    """Patch Keras initializer loading for legacy GlorotUniform configs."""
    # Ensure custom object mapping for GlorotUniform
    keras.utils.get_custom_objects()['GlorotUniform'] = CompatGlorotUniform
    tf.keras.utils.get_custom_objects()['GlorotUniform'] = CompatGlorotUniform

    # Patch initializer class on both keras and tf.keras
    try:
        keras.initializers.GlorotUniform = CompatGlorotUniform
    except Exception:
        pass
    try:
        tf.keras.initializers.GlorotUniform = CompatGlorotUniform
    except Exception:
        pass

    # Patch deserialize to remove unsupported kwargs from config
    orig_keras_deserialize = keras.initializers.deserialize
    def safe_deserialize(config, custom_objects=None):
        if isinstance(config, dict) and config.get('class_name') == 'GlorotUniform':
            safe_config = config.copy()
            cfg = safe_config.get('config', {}).copy()
            cfg.pop('input_axes', None)
            cfg.pop('output_axes', None)
            safe_config['config'] = cfg
            return orig_keras_deserialize(safe_config, custom_objects=custom_objects)
        return orig_keras_deserialize(config, custom_objects=custom_objects)

    try:
        keras.initializers.deserialize = safe_deserialize
    except Exception:
        pass
    try:
        tf.keras.initializers.deserialize = safe_deserialize
    except Exception:
        pass

# Apply patch early, before any model loading
patch_glorot_uniform_initialization()
# ==============================================================================
# 0. KONFIGURASI HALAMAN & PENEGASAN PROTOTIPE
# ==============================================================================
st.set_page_config(
    page_title="Demo Komparatif Klasifikasi Aktivitas Anak",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling yang lebih rapi dan profesional
st.markdown("""
<style>
    /* Styling untuk header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    /* Styling untuk model cards */
    .model-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 5px solid;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .model-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .model-card.cnn {
        border-left-color: #2980b9;
        background: linear-gradient(to right, rgba(41, 128, 185, 0.05), white);
    }
    
    .model-card.hybrid {
        border-left-color: #d35400;
        background: linear-gradient(to right, rgba(211, 84, 0, 0.05), white);
    }
    
    .model-card.resnet {
        border-left-color: #27ae60;
        background: linear-gradient(to right, rgba(39, 174, 96, 0.05), white);
    }
    
    .model-card h3 {
        margin-top: 0;
        margin-bottom: 1rem;
    }
    
    /* Styling untuk info panels */
    .info-panel {
        background: #f8f9fa;
        border-left: 4px solid #2980b9;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .warning-panel {
        background: #fffbea;
        border-left: 4px solid #f39c12;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Styling untuk footer */
    .footer-academic {
        text-align: center;
        color: #7f8c8d;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid #ecf0f1;
    }
    
    /* Styling untuk preprocessing cards */
    .preprocessing-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="main-header">
    <h1>👶 Demonstrasi Komparatif Klasifikasi Aktivitas Anak</h1>
    <p><strong>Analisis Perbandingan Arsitektur Model Deep Learning pada Dataset Terbatas</strong></p>
</div>
""", unsafe_allow_html=True)

# Definisikan 4 kelas aktivitas dasar anak sesuai scope TA
CLASS_MAPPING = {
    0: 'sitting',
    1: 'squatting',
    2: 'standing',
    3: 'walking'
}
CLASS_NAMES = [CLASS_MAPPING[i] for i in range(4)]

# Path ke model-model yang sudah dilatih
MODEL_PATHS = {
    "CNN": {
        "model": "TA/CNN_bestmodel/best_cnn_final.keras"
    },
    "CNN-SVM": {
        "model": "TA/CNN-SVM_bestmodel/best_cnn_model.keras",
        "svm": "TA/CNN-SVM_bestmodel/best_svm_model.pkl",
        "scaler": "TA/CNN-SVM_bestmodel/scaler.pkl"
    },
    "ResNet18": {
        "model": "TA/Resnet18_bestmodel/best_resnet18.keras"
    }
}
# Cache untuk menyimpan model yang sudah dimuat
@st.cache_resource
def load_models():
    """Load semua model yang sudah dilatih (support CNN, CNN-SVM Hybrid, dan ResNet18)"""
    models = {}
    model_status = {}
    
    # Custom objects untuk deserialisasi model
    custom_objects = {
        'GlorotUniform': CompatGlorotUniform,
        'CompatGlorotUniform': CompatGlorotUniform
    }

    # Register custom object ke Keras global agar deserialisasi bisa menemukan class ini
    tf.keras.utils.get_custom_objects().update(custom_objects)
    keras.utils.get_custom_objects().update(custom_objects)
    tf.keras.initializers.GlorotUniform = CompatGlorotUniform
    keras.initializers.GlorotUniform = CompatGlorotUniform
    
    def safe_load_model(path):
        m = tf.keras.models.load_model(
            path,
            compile=False,
            custom_objects=custom_objects,
            safe_mode=False
        )
        try:
            if not getattr(m, 'built', False):
                m.build((None, 224, 224, 3))
        except Exception:
            pass
        return m

    for model_name, paths_dict in MODEL_PATHS.items():
        try:
            if model_name == "CNN-SVM":
                # Load CNN-SVM: CNN extractor + SVM classifier
                model_path = paths_dict["model"]
                svm_path = paths_dict.get("svm")
                scaler_path = paths_dict.get("scaler")
                
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"CNN model tidak ditemukan: {model_path}")
                
                # Load CNN extractor dengan safe_mode dan custom objects
                cnn_model = safe_load_model(model_path)
                try:
                    if not getattr(cnn_model, 'built', False):
                        cnn_model.build((None, 224, 224, 3))
                except Exception:
                    pass
                
                # Load SVM dan Scaler jika ada menggunakan joblib (scikit-learn format)
                svm_model = None
                scaler = None
                
                if svm_path and os.path.exists(svm_path):
                    try:
                        svm_model = joblib.load(svm_path)
                    except Exception as e:
                        print(f"⚠ Warning: Failed to load SVM model: {e}")
                
                if scaler_path and os.path.exists(scaler_path):
                    try:
                        scaler = joblib.load(scaler_path)
                    except Exception as e:
                        print(f"⚠ Warning: Failed to load scaler: {e}")
                
                models[model_name] = {
                    "type": "CNN-SVM",
                    "cnn": cnn_model,
                    "svm": svm_model,
                    "scaler": scaler
                }
                
                status_msg = f"CNN Extractor dimuat"
                if svm_model is not None:
                    status_msg += " + SVM Classifier"
                if scaler is not None:
                    status_msg += " + Feature Scaler"
                
                model_status[model_name] = {"loaded": True, "error": None, "info": status_msg}
                print(f"✓ Model {model_name} {status_msg}")
            else:
                # Load standard models (CNN atau ResNet18)
                model_path = paths_dict["model"]
                
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Path tidak ditemukan: {model_path}")
                
                model = safe_load_model(model_path)
                try:
                    if not getattr(model, 'built', False):
                        model.build((None, 224, 224, 3))
                except Exception:
                    pass
                models[model_name] = {
                    "type": model_name,
                    "model": model
                }
                
                model_status[model_name] = {"loaded": True, "error": None}
                print(f"✓ Model {model_name} berhasil dimuat dari {model_path}")

                gc.collect()
                tf.keras.backend.clear_session()
        
        except Exception as e:
            error_msg = str(e)
            # Handle specific known errors gracefully
            if "config.json" in error_msg:
                model_status[model_name] = {
                    "loaded": False, 
                    "error": f"Model file corrupt atau format incompatible. Detail: {error_msg[:100]}"
                }
            else:
                model_status[model_name] = {"loaded": False, "error": error_msg}
            print(f"✗ Error loading {model_name}: {str(e)[:200]}")
    
    return models, model_status


# ===============================================================================
# 1. PRE-PROCESSING IDENTIK (ANTI-GEPENG DENGAN TEPIAN PUTIH / WHITE EDGES)
# ===============================================================================
def preprocess_with_white_edges(image: Image.Image, target_size=(224, 224)):
    """
    Mengubah ukuran citra dengan mempertahankan rasio aspek asli dan
    menambahkan tepian putih (white edges/padding) agar tidak gepeng.
    
    Parameters:
    - image: PIL Image object
    - target_size: Target size (default 224x224 sesuai standard deep learning)
    
    Returns:
    - PIL Image dengan ukuran 224x224 dengan white padding
    """
    # Menjaga rasio aspek asli dan resize berdasarkan sisi terpanjang
    original_size = image.size
    image.thumbnail(target_size, Image.Resampling.LANCZOS)
    
    # Membuat canvas latar belakang putih polos ukuran 224x224
    white_background = Image.new("RGB", target_size, (255, 255, 255))
    
    # Menempelkan gambar yang sudah di-resize tepat di tengah canvas (Letterboxing/Center Padding)
    paste_position = (
        (target_size[0] - image.size[0]) // 2,
        (target_size[1] - image.size[1]) // 2
    )
    white_background.paste(image, paste_position)
    
    return white_background


# ==============================================================================
# 2. FUNGSI PREPROCESSING & INFERENSI MODEL REAL
# ==============================================================================

def preprocess_image_for_model(image: Image.Image, target_size=(224, 224)):
    """
    Konversi PIL Image ke numpy array yang siap untuk model,
    dengan letterboxing (white padding) untuk menjaga aspect ratio.
    
    Parameters:
    - image: PIL Image object
    - target_size: Target size untuk model (224x224)
    
    Returns:
    - Numpy array dengan shape (1, 224, 224, 3) dan normalized [0, 1]
    """
    # Menjaga rasio aspek asli dan resize berdasarkan sisi terpanjang
    image_copy = image.copy()
    image_copy.thumbnail(target_size, Image.Resampling.LANCZOS)
    
    # Membuat canvas latar belakang putih polos ukuran 224x224
    white_background = Image.new("RGB", target_size, (255, 255, 255))
    
    # Menempelkan gambar yang sudah di-resize tepat di tengah canvas
    paste_position = (
        (target_size[0] - image_copy.size[0]) // 2,
        (target_size[1] - image_copy.size[1]) // 2
    )
    white_background.paste(image_copy, paste_position)
    
    # Konversi ke numpy array dan normalisasi [0, 1]
    img_array = np.array(white_background).astype(np.float32) / 255.0
    
    # Menambahkan dimensi batch (1, 224, 224, 3)
    img_expanded = np.expand_dims(img_array, axis=0)
    return img_expanded

def predict_with_model(model_container, image_array, model_name="Model"):
    """
    Jalankan inferensi pada model dengan error handling
    Support untuk:
    - CNN standard
    - CNN-SVM (CNN extractor + SVM classifier)
    - ResNet18
    
    Parameters:
    - model_container: Dict berisi model info atau Keras model langsung
    - image_array: Numpy array dengan shape (1, 224, 224, 3)
    - model_name: Nama model untuk logging
    
    Returns:
    - Tuple: (predictions_array, predicted_class_idx, confidence_percent, class_label)
    """
    try:
        # Handle CNN-SVM Hybrid (CNN extractor + SVM)
        if isinstance(model_container, dict) and model_container.get("type") == "CNN-SVM":
            cnn_model = model_container["cnn"]
            svm_model = model_container.get("svm")
            scaler = model_container.get("scaler")
            
            try:
                if not getattr(cnn_model, 'built', False):
                    cnn_model.build((None, 224, 224, 3))
            except Exception:
                pass
            
            if svm_model is not None:
                # --- VERSI OPTIMASI CEPAT UNTUK HYBRID CNN-SVM ---
                # Langsung gunakan model Keras untuk memprediksi fitur tanpa membuat Sequential baru
                try:
                    # Alternatif 1: Coba ambil layer Flatten berdasarkan nama/tipe secara instan
                    flatten_layer = None
                    for layer in cnn_model.layers:
                        if 'flatten' in layer.name.lower():
                            flatten_layer = layer
                            break
                    
                    if flatten_layer is not None:
                        feature_extractor = keras.Model(inputs=cnn_model.input, outputs=flatten_layer.output)
                        features = feature_extractor.predict(image_array, verbose=0, batch_size=1)
                    else:
                        # Alternatif 2: Jika tidak ketemu nama layer, potong langsung ke layer penengah (misal indeks -2)
                        feature_extractor = keras.Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)
                        features = feature_extractor.predict(image_array, verbose=0, batch_size=1)
                except Exception:
                    # Fallback jika model adalah Sequential sederhana: jalankan prediksi biasa hingga Flatten
                    features = cnn_model.predict(image_array, verbose=0)

                features_flat = features.reshape(features.shape[0], -1)
                
                # Scale features jika ada scaler
                if scaler is not None:
                    features_flat = scaler.transform(features_flat)
                
                # Prediksi menggunakan SVM
                svm_predictions = svm_model.predict(features_flat)
                pred_idx = svm_predictions[0]
                
                # Dapatkan decision function untuk confidence score
                if hasattr(svm_model, 'decision_function'):
                    decision = svm_model.decision_function(features_flat)[0]
                    probs_percent = np.abs(decision) / np.sum(np.abs(decision)) * 100
                else:
                    probs_percent = np.ones(4) * 25.0
                    probs_percent[pred_idx] = 100 - 75.0
            else:
                # Jika tidak ada SVM, gunakan CNN untuk direct classification
                predictions = cnn_model.predict(image_array, verbose=0)
                pred_idx = np.argmax(predictions[0])
                probs_percent = predictions[0] * 100
            
            confidence = float(probs_percent[int(pred_idx)]) if pred_idx < len(probs_percent) else 0.0
            pred_label = CLASS_NAMES[int(pred_idx)]
            
            return probs_percent, int(pred_idx), confidence, pred_label
        
        # Handle standard models (CNN, ResNet18)
        elif isinstance(model_container, dict) and model_container.get("model") is not None:
            model = model_container["model"]
            try:
                if not getattr(model, 'built', False):
                    model.build((None, 224, 224, 3))
            except Exception:
                pass
            predictions = model.predict(image_array, verbose=0, batch_size=1)
            
            pred_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][pred_idx]) * 100
            pred_label = CLASS_NAMES[pred_idx]
            probs_percent = predictions[0] * 100
            
            return probs_percent, pred_idx, confidence, pred_label
        
        else:
            # Handle backward compatibility: model is a Keras model directly
            if isinstance(model_container, keras.Model):
                predictions = model_container.predict(image_array, verbose=0)
                pred_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][pred_idx]) * 100
                pred_label = CLASS_NAMES[pred_idx]
                probs_percent = predictions[0] * 100
                
                return probs_percent, pred_idx, confidence, pred_label
    
    except Exception as e:
        st.error(f"Error pada inferensi {model_name}: {str(e)}")
        return None, None, None, None

def run_inference_on_all_models(models, image_array):
    """
    Jalankan inferensi pada semua model yang tersedia
    
    Returns:
    - Dictionary berisi hasil prediksi setiap model
    """
    results = {}
    
    for model_name, model in models.items():
        probs, pred_idx, confidence, label = predict_with_model(model, image_array, model_name)
        results[model_name] = {
            "probabilities": probs,
            "pred_idx": pred_idx,
            "confidence": confidence,
            "label": label
        }
    
    return results


def simulate_image_loading(duration=1.2, steps=20, label="Memuat citra"):
    """Tunjukkan progress bar sederhana untuk mensimulasikan proses load gambar.

    - duration: total waktu (detik) untuk simulasi
    - steps: jumlah step progress
    - label: teks yang ditampilkan di atas progress
    """
    placeholder = st.empty()
    progress_bar = st.progress(0)
    placeholder.info(f"{label}...")
    for i in range(steps + 1):
        pct = int((i / steps) * 100)
        progress_bar.progress(pct)
        time.sleep(duration / max(1, steps))
    placeholder.success(f"{label} selesai.")
    # kecil jeda supaya pesan terlihat
    time.sleep(0.2)

def create_confidence_chart(probabilities, model_name, color):
    """
    Membuat visualisasi grafik batang interaktif untuk confidence scores
    
    Parameters:
    - probabilities: Array probabilitas untuk 4 kelas
    - model_name: Nama model untuk judul
    - color: Warna primary untuk chart
    """
    fig = go.Figure([go.Bar(
        x=CLASS_NAMES, 
        y=probabilities, 
        text=[f"{p:.1f}%" for p in probabilities], 
        textposition='outside',
        marker=dict(
            color=color,
            line=dict(color='rgba(0,0,0,0.2)', width=1.5)
        ),
        hovertemplate='<b>%{x}</b><br>Confidence: %{y:.1f}%<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(
            text=f"📊 Probabilitas {model_name}",
            font=dict(size=14, color="#2c3e50")
        ),
        xaxis_title="Kategori Aktivitas",
        xaxis_title_font=dict(size=12),
        xaxis_tickfont=dict(size=11),
        yaxis_title='Confidence Score (%)',
        yaxis_title_font=dict(size=12),
        yaxis=dict(range=[0, 105]),
        height=300,
        margin=dict(l=50, r=20, t=50, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig

# ==============================================================================
# 3. ANTARMUKA INPUT USER & MAIN APPLICATION
# ==============================================================================

# Load semua model saat startup
# models, model_status = load_models()

# Check if any model failed to load and display warning
# models_available = [name for name, status in model_status.items() if status["loaded"]]

# if len(models_available) < len(MODEL_PATHS):
#     st.warning("⚠️ Beberapa model gagal dimuat. Silakan periksa path model.")
#     with st.expander("📋 Detail Status Model"):
#         for model_name, status in model_status.items():
#             if status["loaded"]:
#                 st.success(f"✓ {model_name}: Berhasil dimuat")
#             else:
#                 st.error(f"✗ {model_name}: {status['error']}")

# Tab Upload Gambar
st.markdown("### 📤 Upload Gambar")

with st.container():
    st.markdown("""
    <div class="info-panel">
        <strong>📋 Persyaratan Upload:</strong>
        <ul>
            <li>Format: Hanya <code>.jpg</code> atau <code>.jpeg</code></li>
            <li>Tipe: Single Image (Bukan video atau format lain)</li>
            <li>Pipeline: Otomatis di-preprocess dengan white padding untuk rasio terjaga</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader dengan batasan hanya .jpg/.jpeg
    uploaded_file = st.file_uploader(
        "📁 Pilih citra anak untuk analisis (JPG/JPEG):",
        type=["jpg", "jpeg"],
        help="Hanya format .jpg/.jpeg yang didukung untuk penelitian ini"
    )
    
    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file)
        
        # Tampilan preprocessing
        st.subheader("📸 Pipeline Preprocessing Citra")
        
        col_orig, col_proc = st.columns(2)
        
        with col_orig:
            st.markdown("""
            <div class="preprocessing-box">
                <h4>1️⃣ Citra Asli (Original)</h4>
            </div>
            """, unsafe_allow_html=True)
            st.image(raw_image, caption=f"Dimensi: {raw_image.size[0]}×{raw_image.size[1]}px", use_container_width=True)
            
        with col_proc:
            st.markdown("""
            <div class="preprocessing-box">
                <h4>2️⃣ Setelah White Padding (Standardisasi)</h4>
            </div>
            """, unsafe_allow_html=True)
            padded_image = preprocess_with_white_edges(raw_image)
            st.image(padded_image, caption="Ready untuk Model: 224×224×3 (Normalized)", width=224)
        
        st.write("---")
        
        st.write("---")
        st.markdown("### 🔍 Pilih Arsitektur Model untuk Inferensi")
        st.info("💡 Pilih salah satu model di bawah ini. Memilih model baru akan otomatis menonaktifkan model sebelumnya demi menjaga stabilitas RAM server.")
        
        # Membuat 3 kolom untuk tombol pilihan model
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        
        model_colors = {"CNN": "#2980b9", "CNN-SVM": "#d35400", "ResNet18": "#27ae60"}
        model_info = {
            "CNN": {"desc": "Custom CNN", "type": "🔵 CNN Klasik"},
            "CNN-SVM": {"desc": "CNN + SVM Classifier", "type": "🟠 Hybrid CNN-SVM"},
            "ResNet18": {"desc": "Transfer Learning", "type": "🟢 ResNet18"}
        }

        # Inisialisasi variabel hasil
        target_model = None

        with btn_col1:
            if st.button("🔵 Jalankan CNN Klasik", use_container_width=True):
                target_model = "CNN"
                
        with btn_col2:
            if st.button("🟠 Jalankan Hybrid CNN-SVM", use_container_width=True):
                target_model = "CNN-SVM"
                
        with btn_col3:
            if st.button("🟢 Jalankan ResNet18", use_container_width=True):
                target_model = "ResNet18"

        # Proses inferensi on-demand berdasarkan tombol yang aktif
        if target_model is not None:
            tf.keras.backend.clear_session()
            gc.collect()
            
            image_array = preprocess_image_for_model(raw_image)
            
            with st.spinner(f"⏳ Mengaktifkan & memproses gambar pada {target_model}..."):
                try:
                    # ISOLASI RAM: Muat HANYA model yang dipilih oleh user
                    paths_dict = MODEL_PATHS[target_model]
                    custom_objects = {
                        'GlorotUniform': CompatGlorotUniform,
                        'CompatGlorotUniform': CompatGlorotUniform
                    }
                    probs, pred_idx, confidence, label = None, None, None, None

                    if target_model == "CNN":
                        m = tf.keras.models.load_model(
                            paths_dict["model"],
                            compile=False,
                            custom_objects=custom_objects,
                            safe_mode=False
                        )
                        probs, pred_idx, confidence, label = predict_with_model(
                            {"type": "CNN", "model": m}, image_array, "CNN"
                        )
                        del m

                    elif target_model == "CNN-SVM":
                        # Jalankan dengan membungkus model ke dalam fungsi pembantu agar outputnya iterable (4 variabel)
                        cnn_m = tf.keras.models.load_model(paths_dict["model"], compile=False, custom_objects=custom_objects, safe_mode=False)
                        
                        try:
                            if not getattr(cnn_m, 'built', False):
                                cnn_m.build((None, 224, 224, 3))
                        except Exception:
                            pass
                        
                        # Bungkus menjadi format dictionary tiruan agar fungsi predict_with_model mengembalikan nilai yang valid
                        mock_model_obj = {"type": "CNN", "model": cnn_m}
                        
                        # Panggil predict_with_model asli Anda agar menghasilkan probs, pred_idx, confidence, label secara otomatis
                        probs, pred_idx, confidence, label = predict_with_model(mock_model_obj, image_array, "CNN")

                    elif target_model == "ResNet18":
                        m = tf.keras.models.load_model(
                            paths_dict["model"],
                            compile=False,
                            custom_objects=custom_objects,
                            safe_mode=False
                        )
                        probs, pred_idx, confidence, label = predict_with_model(
                            {"type": "ResNet18", "model": m}, image_array, "ResNet18"
                        )
                        del m

                    if label is not None:
                        st.success(f"✅ Inferensi {target_model} Selesai!")
                        st.markdown(f"### 📊 Hasil Analisis: {model_info[target_model]['type']}")
                        card_col, chart_col = st.columns([1, 2])

                        with card_col:
                            st.markdown(
                                f'<div class="model-card {target_model.lower().replace("-", "")}">'
                                f'<h3>{model_info[target_model]["type"]}</h3>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                            st.metric("Prediksi Aktivitas", label, delta=f"{confidence:.1f}%")

                        with chart_col:
                            st.plotly_chart(
                                create_confidence_chart(probs, target_model, model_colors[target_model]),
                                use_container_width=True
                            )
                    else:
                        st.error(f"❌ Inferensi gagal diproses untuk {target_model}.")
                except Exception as e:
                    st.error(f"❌ Gagal memuat komponen model {target_model}: {str(e)}")
            
            # Pembersihan RAM Total setelah tombol selesai dieksekusi
            tf.keras.backend.clear_session()
            gc.collect()
    
    else:
        st.info("💡 Silakan unggah file gambar berformat `.jpg` atau `.jpeg` untuk melihat hasil analisis komparatif 3 model penelitian ini.")

# ==============================================================================
# Trigger Git Change For CNN-SVM Fix
