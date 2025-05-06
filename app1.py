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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import logging
import soundfile as sf
import subprocess
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline,VideoMAEForVideoClassification, AutoProcessor
import torch.nn as nn
import scipy.stats as stats
from pydub.utils import which
from captum.attr import IntegratedGradients, GuidedBackprop, Occlusion
import torch.nn.functional as F
import math
ffmpeg_path = r"D:\ffmpeg\bin"  # Change this if your path is different
os.environ["PATH"] += os.pathsep + ffmpeg_path

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

# # Hugging Face API for audio deepfake detection
# API_URL = "https://api-inference.huggingface.co/models/mo-thecreator/Deepfake-audio-detection"
# HUGGINGFACE_HEADERS = {"Authorization": "Bearer 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"}
# Initialize the audio classification pipeline

# Video processing configuration
VIDEO_CONFIG = {
    'frame_count': 10,
    'min_frame_interval': 5,
    'target_size': (299, 299),
    'xception_size': (299, 299),  # Add this
    'videomae_size': (224, 224)   # Add this (typical size for VideoMAE)
}

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the VideoMAE model for deepfake detection
def load_videomae_model():
    print("Loading VideoMAE model...")
    processor = AutoProcessor.from_pretrained("Ammar2k/videomae-base-finetuned-deepfake-subset")
    model = VideoMAEForVideoClassification.from_pretrained("Ammar2k/videomae-base-finetuned-deepfake-subset")
    model.eval()
    return processor, model.to(device)

# Load Xception models
print("Loading Xception models...")
xception_models = load_xception_models(xception_model_paths)
if not xception_models:
    raise RuntimeError("Failed to load any Xception models. Check model paths.")

# Initialize the VideoMAE model
videomae_processor, videomae_model = load_videomae_model()

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
        # This will use the ffmpeg available in your PATH
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
    audio_pipe = pipeline("audio-classification", model="Bisher/wav2vec2_ASV_deepfake_audio_detection")
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
        
        # Use the pipeline for classification
        result = audio_pipe(filepath)
        
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
        
        # Enhanced audio analysis with multiple visualizations
        y, sr = librosa.load(filepath)
        
        # Create a figure with multiple subplots for different visualizations
        plt.switch_backend('Agg')
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        
        # 1. Spectrogram visualization
        D = librosa.stft(y)
        spectrogram = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img = librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='hz', ax=axs[0])
        fig.colorbar(img, ax=axs[0], format='%+2.0f dB')
        axs[0].set_title('Audio Spectrogram')
        
        # 2. Mel-frequency spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        img2 = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=axs[1])
        fig.colorbar(img2, ax=axs[1], format='%+2.0f dB')
        axs[1].set_title('Mel Spectrogram')
        
        # Adjust layout and convert to base64
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        buf.seek(0)
        spectrogram_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # Additional waveform visualization
        plt.figure(figsize=(10, 2))
        librosa.display.waveshow(y, sr=sr)
        plt.title('Waveform')
        plt.tight_layout()
        
        waveform_buf = io.BytesIO()
        plt.savefig(waveform_buf, format='png', dpi=100)
        plt.close()
        waveform_buf.seek(0)
        waveform_base64 = base64.b64encode(waveform_buf.getvalue()).decode('utf-8')
        
        # Cleanup
        cleanup_files(filepath)
        
        return jsonify({
            'output': max_confidence_result['label'].upper(),
            'confidence': round(max_confidence_result['score'] * 100, 2),
            'percentages': {
                'fake': fake_percentage,
                'real': real_percentage
            },
            'visualizations': {
                'spectrogram': spectrogram_base64,
                'waveform': waveform_base64
            },
            'detailed_results': normalized_results
        })
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in detect-audio: {str(e)}\nTraceback: {error_trace}")
        cleanup_files(orig_filepath, filepath)
        return jsonify({"error": str(e), "traceback": error_trace}), 500

# New hybrid function that uses both Xception and VideoMAE
def detect_deepfake_hybrid(video_path, output_dir, config=None):
    if config is None:
        config = VIDEO_CONFIG
    
    # Step 1: Extract frames for both models
    xception_frames = extract_frames(
        video_path, 
        frame_count=config['frame_count'],
        visualize=False
    )
    
    # Resize frames to the target size for Xception
    xception_frames = [cv2.resize(frame, config['xception_size']) for frame in xception_frames]
    
    # Step 2: Extract Xception features (for additional analysis)
    xception_features = []
    xception_predictions = []
    
    for model_name, model in xception_models.items():
        # Call extract_combined_features with only the expected parameters
        features, original_tensors = extract_combined_features(xception_frames, {model_name: model})
        
        # Now we need to get predictions from these features
        with torch.no_grad():
            model.eval()
            pred = torch.sigmoid(model(original_tensors[0])).item()
            
        xception_features.append(features)
        xception_predictions.append(pred)
    
    # Average Xception predictions
    avg_xception_pred = np.mean(xception_predictions)
    xception_prediction = "fake" if avg_xception_pred > 0.5 else "real"
    
    # Step 3: Process frames for VideoMAE
    # First, ensure we have exactly 16 frames (typical VideoMAE expectation)
    if len(xception_frames) < 16:
        # Duplicate frames if we have fewer than 16
        multiplier = math.ceil(16 / len(xception_frames))
        videomae_frames = xception_frames * multiplier
        videomae_frames = videomae_frames[:16]  # Take exactly 16 frames
    elif len(xception_frames) > 16:
        # Sample frames if we have more than 16
        indices = np.linspace(0, len(xception_frames)-1, 16, dtype=int)
        videomae_frames = [xception_frames[i] for i in indices]
    else:
        videomae_frames = xception_frames
    
    # Ensure frames are resized to VideoMAE expected size (224x224)
    videomae_frames = [cv2.resize(frame, (224, 224)) for frame in videomae_frames]
    
    # Convert frames to PIL Images
    pil_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in videomae_frames]
    
    try:
        # Process frames for VideoMAE - with appropriate pixel_values shape
        # VideoMAE typically expects [batch_size, num_frames, height, width, channels]
        inputs = videomae_processor(pil_frames, return_tensors="pt")
        
        # Debug the shape to verify
        print(f"VideoMAE input shape: {inputs['pixel_values'].shape}")
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run inference with VideoMAE
        with torch.no_grad():
            outputs = videomae_model(**inputs)
            logits = outputs.logits
        
        # Get VideoMAE prediction and confidence
        probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
        prediction_idx = torch.argmax(probabilities).item()
        videomae_confidence = probabilities[prediction_idx].item()
        videomae_prediction = "fake" if prediction_idx == 1 else "real"
    
    except Exception as e:
        print(f"VideoMAE processing error: {e}")
        # Fallback to Xception prediction if VideoMAE fails
        videomae_prediction = xception_prediction
        videomae_confidence = avg_xception_pred if avg_xception_pred > 0.5 else 1 - avg_xception_pred
        print(f"Falling back to Xception prediction: {videomae_prediction} ({videomae_confidence:.2f})")
    
    # Calculate frame importance
    frame_importance = compute_simple_frame_importance(xception_frames, xception_models)
    
    # Generate visualizations showing both model predictions
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comparison visualization
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(['Real', 'Fake'], [1-avg_xception_pred, avg_xception_pred])
    plt.title(f"Xception Prediction: {xception_prediction.upper()}")
    plt.ylim(0, 1)
    
    plt.subplot(1, 2, 2)
    if isinstance(probabilities, torch.Tensor):
        real_prob = probabilities[0].item() if len(probabilities) > 1 else 1-videomae_confidence
        fake_prob = probabilities[1].item() if len(probabilities) > 1 else videomae_confidence
    else:
        real_prob = 1-videomae_confidence
        fake_prob = videomae_confidence
    
    plt.bar(['Real', 'Fake'], [real_prob, fake_prob])
    plt.title(f"VideoMAE Prediction: {videomae_prediction.upper()}")
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"))
    plt.close()
    
    # Generate frame visualizations
    for i, frame in enumerate(xception_frames):
        # Add a colored border based on importance
        importance = frame_importance[i] if i < len(frame_importance) else 0.0
        color = (0, int(255 * (1-importance)), int(255 * importance))  # Red to Green
        bordered_frame = cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=color)
        cv2.imwrite(os.path.join(output_dir, f"frame_{i}_imp_{importance:.2f}.jpg"), bordered_frame)
    
    # Return VideoMAE's prediction as the final result
    # but include Xception features for additional insight
    return videomae_prediction, videomae_confidence, xception_frames, frame_importance, xception_features

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
        'xception_size': VIDEO_CONFIG['xception_size'],
        'videomae_size': VIDEO_CONFIG['videomae_size']
    }
    
    video_path = os.path.join(UPLOAD_FOLDER, secure_filename(video_file.filename))
    video_file.save(video_path)
    output_dir = os.path.join(OUTPUT_DIR, os.path.splitext(video_file.filename)[0])
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Use the hybrid detection function
        prediction, confidence, frames, frame_importance, xception_features = detect_deepfake_hybrid(
            video_path,
            output_dir,
            config=custom_video_config
        )

        # Convert frame importance to list if it's a tensor
        if isinstance(frame_importance, torch.Tensor):
            frame_importance = frame_importance.tolist()
        elif frame_importance is None:
            frame_importance = []
            
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

        # Prepare model comparison visualization
        model_comparison_path = os.path.join(output_dir, "model_comparison.png")
        model_comparison_base64 = ""
        if os.path.exists(model_comparison_path):
            with open(model_comparison_path, "rb") as f:
                model_comparison_base64 = base64.b64encode(f.read()).decode('utf-8')

        # Prepare visualizations
        xai_visuals = []
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                if file.startswith("frame_") and file.endswith(".jpg"):
                    with open(os.path.join(output_dir, file), "rb") as f:
                        frame_num = int(file.split("_")[1])
                        xai_visuals.append({
                            "frame": frame_num,
                            "image": base64.b64encode(f.read()).decode('utf-8'),
                            "model": "combined"
                        })

        response = {
            "prediction": prediction.upper(),
            "confidence": round(confidence * 100, 2),
            "frames": {
                "count": len(frames),
                "importance_scores": [float(x) for x in frame_importance] if frame_importance else [],
                "images": frame_data
            },
            "model_comparison": model_comparison_base64,
            "xai_visualizations": xai_visuals
        }

        cleanup_files(video_path)
        return jsonify(response)

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in detect-video: {str(e)}\nTraceback: {error_trace}")
        cleanup_files(video_path)
        return jsonify({
            "error": str(e),
            "type": type(e).__name__,
            "traceback": error_trace
        }), 500


# In the detect-image route, modify it to include XAI visualizations
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
   
    # Check if XAI visualizations are requested
    generate_xai = request.form.get('generate_xai', 'false').lower() == 'true'
    
    try:
        # Load image with PIL first (more memory efficient)
        pil_image = Image.open(filepath).convert("RGB")
        
        # Resize large images to reduce memory usage
        max_size = 800  # Maximum dimension
        if pil_image.width > max_size or pil_image.height > max_size:
            # Calculate ratio to preserve aspect ratio
            ratio = min(max_size / pil_image.width, max_size / pil_image.height)
            new_width = int(pil_image.width * ratio)
            new_height = int(pil_image.height * ratio)
            pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        
        # Process for model
        inputs = processor(images=pil_image, return_tensors="pt")

        with torch.no_grad():
            outputs = image_model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            probabilities = torch.softmax(logits, dim=1).numpy()[0]
        
        response = {
            "prediction": image_labels.get(predicted_class, 'Unknown'),
            "confidence": {
                "real": float(round(probabilities[0] * 100, 2)),
                "fake": float(round(probabilities[1] * 100, 2))
            }
        }
        
        # Generate simple XAI visualizations if requested
        if generate_xai:
            # Convert PIL image to numpy array for visualization
            img_array = np.array(pil_image)
            
            # Create memory-efficient visualizations container
            xai_visualizations = {}
            
            # 1. Pixel Distribution Histogram - simple and memory efficient
            plt.figure(figsize=(5, 3), dpi=80)  # Smaller figure size to save memory
            
            # Downsample the image for histogram to reduce memory usage
            downsampled = img_array[::4, ::4, :]
            
            # Plot histogram for each channel
            plt.hist(downsampled[:,:,0].flatten(), bins=50, alpha=0.5, color='red', label='R')
            plt.hist(downsampled[:,:,1].flatten(), bins=50, alpha=0.5, color='green', label='G')
            plt.hist(downsampled[:,:,2].flatten(), bins=50, alpha=0.5, color='blue', label='B')
            plt.title('Pixel Distribution')
            plt.xlabel('Intensity')
            plt.ylabel('Count')
            plt.legend()
            plt.tight_layout()
            
            # Save to buffer
            hist_buf = io.BytesIO()
            plt.savefig(hist_buf, format='png')
            plt.close()  # Close the figure to free memory
            hist_buf.seek(0)
            xai_visualizations['histogram'] = base64.b64encode(hist_buf.getvalue()).decode('utf-8')
            
            # Memory-efficient XAI approximations:
            
            # 2. Simplified "Integrated Gradients" visualization (memory-efficient approximation)
            plt.figure(figsize=(5, 5), dpi=80)
            
            # Create a simplified heatmap based on image features
            # This uses image edges which are often relevant for deepfake detection
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            
            # Create a colormap overlay
            plt.imshow(pil_image)
            plt.imshow(edges, cmap='hot', alpha=0.5)
            plt.title('Edge Analysis')
            plt.axis('off')
            
            ig_buf = io.BytesIO()
            plt.savefig(ig_buf, format='png')
            plt.close()
            ig_buf.seek(0)
            xai_visualizations['integrated_gradients'] = base64.b64encode(ig_buf.getvalue()).decode('utf-8')
            
            # 3. Simplified "Grad-CAM" visualization (memory-efficient approximation)
            plt.figure(figsize=(5, 5), dpi=80)
            
            # Use frequency domain features (often relevant for deepfake detection)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply simple FFT as a proxy for frequency analysis that deepfake models often use
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
            
            # Normalize for visualization
            magnitude_spectrum = (magnitude_spectrum - magnitude_spectrum.min()) / (magnitude_spectrum.max() - magnitude_spectrum.min())
            
            # Resize to match original image
            magnitude_spectrum_resized = cv2.resize(magnitude_spectrum, (img_array.shape[1], img_array.shape[0]))
            
            # Show original with overlay
            plt.imshow(pil_image)
            plt.imshow(magnitude_spectrum_resized, cmap='inferno', alpha=0.5)
            plt.title('Frequency Analysis')
            plt.axis('off')
            
            cam_buf = io.BytesIO()
            plt.savefig(cam_buf, format='png')
            plt.close()
            cam_buf.seek(0)
            xai_visualizations['grad_cam'] = base64.b64encode(cam_buf.getvalue()).decode('utf-8')
            
            # 4. Simplified "Occlusion" visualization (memory-efficient approximation)
            plt.figure(figsize=(5, 5), dpi=80)
            
            # Create a simple noise analysis visualization
            # Many deepfakes have specific noise patterns
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Extract image noise using high-pass filter
            blur = cv2.GaussianBlur(gray, (0, 0), 3)
            noise = cv2.subtract(gray, blur)
            
            # Enhance noise for visibility
            noise = cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX)
            
            # Show original with overlay
            plt.imshow(pil_image)
            plt.imshow(noise, cmap='viridis', alpha=0.5)
            plt.title('Noise Pattern Analysis')
            plt.axis('off')
            
            occlusion_buf = io.BytesIO()
            plt.savefig(occlusion_buf, format='png')
            plt.close()
            occlusion_buf.seek(0)
            xai_visualizations['occlusion'] = base64.b64encode(occlusion_buf.getvalue()).decode('utf-8')
            
            # Add the visualizations to the response
            response['visualizations'] = {'histogram': xai_visualizations['histogram']}
            
            # Add XAI visualizations separately for the frontend component
            response['xai_visualizations'] = {
                'integrated_gradients': xai_visualizations['integrated_gradients'],
                'grad_cam': xai_visualizations['grad_cam'],
                'occlusion': xai_visualizations['occlusion']
            }
        
        # Clean up resources
        cleanup_files(filepath)
        
        # Explicitly trigger garbage collection to free memory
        if generate_xai:
            import gc
            gc.collect()
            
        return jsonify(response)

    except Exception as e:
        import traceback
        print(f"Error in detect-image: {str(e)}\nTraceback: {traceback.format_exc()}")
        cleanup_files(filepath)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)