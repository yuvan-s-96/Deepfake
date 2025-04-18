from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import requests
import os
import io
import librosa.display
import torch
import torchvision.transforms as transforms
import timm
import numpy as np
import cv2
from PIL import Image
import warnings
import librosa
import matplotlib.pyplot as plt
import base64
import logging
import soundfile as sf
import subprocess
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import scipy.stats as stats

# Suppress warnings
warnings.filterwarnings("ignore")

# Import deepfake detection functions from ok.py
from ok import (
    load_xception_models, 
    VideoTransformer,
    detect_deepfake,
    xception_model_paths,
    extract_frames,
    display_frames,
    extract_combined_features,
    visualize_frame_attributions,
    visualize_attention_weights,
    compute_simple_frame_importance
)

app = Flask(__name__)
CORS(app)

# Configuration
OUTPUT_DIR = os.path.join(os.getcwd(), "deepfake_analysis_output")
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create required directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Hugging Face API for audio deepfake detection
API_URL = "https://api-inference.huggingface.co/models/mo-thecreator/Deepfake-audio-detection"
HUGGINGFACE_HEADERS = {"Authorization": "Bearer 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"}

# Video processing configuration
VIDEO_CONFIG = {
    'frame_count': 10,
    'min_frame_interval': 5,
    'target_size': (299, 299)
}

# ✅ Load Pretrained Xception Models for Video Deepfake Detection
xception_models = load_xception_models(xception_model_paths)
if not xception_models:
    raise RuntimeError("Failed to load any Xception models. Check model paths.")

# ✅ Initialize Transformer Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_dim = 2048 * len(xception_models)
transformer_model = VideoTransformer(
    feature_dim=feature_dim,
    num_frames=VIDEO_CONFIG['frame_count'],
    num_blocks=4,
    num_heads=8,
    d_ff=2048,
    dropout=0.1
).to(device)

# Load image detection model
processor = AutoImageProcessor.from_pretrained("Skullly/DeepFake-image-detection-ViT-384")
image_model = AutoModelForImageClassification.from_pretrained("Skullly/DeepFake-image-detection-ViT-384")
image_model.eval()
image_labels = {0: "real", 1: "fake"}

def cleanup_files(*paths):
    """Clean up temporary files"""
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                print(f"Error cleaning up {path}: {e}")

def convert_to_flac(input_path):
    """Convert audio file to FLAC format"""
    filename = os.path.basename(input_path)
    base_filename = os.path.splitext(filename)[0]
    flac_path = os.path.join(UPLOAD_FOLDER, f"{base_filename}.flac")
    
    try:
        subprocess.run([
            'ffmpeg', 
            '-i', input_path, 
            '-c:a', 'flac', 
            '-y',
            flac_path
        ], check=True, capture_output=True)
        return flac_path
    except subprocess.CalledProcessError as e:
        print(f"Conversion error: {e.stderr.decode()}")
        raise RuntimeError(f"Failed to convert audio: {e.stderr.decode()}")

@app.route('/detect-audio', methods=['POST'])
def detect_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    original_filename = request.form.get('original_filename', file.filename)
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    secure_orig_filename = secure_filename(original_filename)
    orig_filepath = os.path.join(UPLOAD_FOLDER, secure_orig_filename)
    file.save(orig_filepath)
    
    filepath = None
    
    try:
        file_ext = os.path.splitext(secure_orig_filename)[1].lower()
        if file_ext not in ['.flac']:
            filepath = convert_to_flac(orig_filepath)
        else:
            filepath = orig_filepath
        
        if filepath != orig_filepath:
            os.remove(orig_filepath)
        
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            raise ValueError("Invalid audio file after conversion")
        
        with open(filepath, "rb") as f:
            response = requests.post(API_URL, headers=HUGGINGFACE_HEADERS, data=f, timeout=30)
        
        response.raise_for_status()
        result = response.json()
        
        if not result or not isinstance(result, list):
            raise ValueError("Invalid response from API")
        
        # Process results
        total_score = sum(item['score'] for item in result)
        normalized_results = [
            {'label': item['label'], 'normalized_score': (item['score'] / total_score) * 100} 
            for item in result
        ]
        
        fake_percentage = next(
            (r['normalized_score'] for r in normalized_results if r['label'].lower() == 'fake'), 
            0
        )
        real_percentage = next(
            (r['normalized_score'] for r in normalized_results if r['label'].lower() == 'real'), 
            0
        )
        
        max_confidence_result = max(result, key=lambda x: x['score'])
        
        # Audio analysis
        y, sr = librosa.load(filepath)
        D = librosa.stft(y)
        spectrogram = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        plt.switch_backend('Agg')
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Audio Spectrogram')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        spectrogram_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # Cleanup
        cleanup_files(filepath)
        
        return jsonify({
            'output': max_confidence_result['label'].upper(),
            'confidence': round(max_confidence_result['score'] * 100, 2),
            'percentages': {
                'fake': fake_percentage,
                'real': real_percentage
            },
            'spectrogram': spectrogram_base64
        })
    
    except Exception as e:
        cleanup_files(orig_filepath, filepath)
        return jsonify({"error": str(e)}), 500

@app.route('/detect-video', methods=['POST'])
def detect_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Extract parameters from request
    frame_count = int(request.form.get('frame_count', VIDEO_CONFIG['frame_count']))
    output_visuals = request.form.get('output_visuals', 'true').lower() == 'true'
    
    # Update video config with user parameters
    custom_video_config = {
        'frame_count': frame_count,
        'min_frame_interval': VIDEO_CONFIG['min_frame_interval'],
        'target_size': VIDEO_CONFIG['target_size']
    }
    
    video_path = os.path.join(UPLOAD_FOLDER, secure_filename(video_file.filename))
    video_file.save(video_path)
    output_dir = os.path.join(OUTPUT_DIR, os.path.splitext(video_file.filename)[0])
    os.makedirs(output_dir, exist_ok=True)

    try:
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        prediction, confidence, frames, frame_importance = detect_deepfake(
            video_path,
            xception_models,
            transformer_model,
            output_dir,
            
        )

        # Convert frame importance to list if it's a tensor
        if isinstance(frame_importance, torch.Tensor):
            frame_importance = frame_importance.tolist()
        elif frame_importance is None:
            frame_importance = []
        # Rest of the code remains the same...
        # Prepare frame data
        frame_data = []
        for i, frame in enumerate(frames):
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            importance = frame_importance[i] if frame_importance else 0.0
            frame_data.append({
                "image": frame_base64,
                "importance": float(importance) if isinstance(importance, (float, int, np.number)) else 0.0
            })

        # Prepare XAI visualizations
        xai_visuals = []
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                if file.startswith("xai_frame_") and file.endswith(".png"):
                    with open(os.path.join(output_dir, file), "rb") as f:
                        xai_visuals.append({
                            "frame": int(file.split("_")[2]),
                            "image": base64.b64encode(f.read()).decode('utf-8'),
                            "model": file.split("_")[-1].split(".")[0]
                        })

        response = {
            "prediction": prediction.upper(),
            "confidence": round(confidence * 100, 2),
            "frames": {
                "count": len(frames),
                "importance_scores": [float(x) for x in frame_importance] if frame_importance else [],
                "images": frame_data
            },
            "xai_visualizations": xai_visuals
        }

        cleanup_files(video_path)
        return jsonify(response)

    except Exception as e:
        cleanup_files(video_path)
        return jsonify({
            "error": str(e),
            "type": type(e).__name__
        }), 500

@app.route('/detect-image', methods=['POST'])
def detect_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
   
    try:
        image = Image.open(filepath).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = image_model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            probabilities = torch.softmax(logits, dim=1).numpy()[0]
        
        cleanup_files(filepath)
        
        return jsonify({
            "prediction": image_labels.get(predicted_class, 'Unknown'),
            "confidence": {
                "real": float(round(probabilities[0] * 100, 2)),
                "fake": float(round(probabilities[1] * 100, 2))
            }
        })

    except Exception as e:
        cleanup_files(filepath)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)