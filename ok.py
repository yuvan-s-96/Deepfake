import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os

# ✅ Paths to Pretrained Xception Models
xception_model_paths = {
    "dfdc": r"C:\Users\Yuvan Velkumar\Downloads\drive-download-20250227T132846Z-001\xception_dfdc.pth",
    "dfdc_seed25": r"C:\Users\Yuvan Velkumar\Downloads\drive-download-20250227T132846Z-001\xception_dfdc_seed25.pth",
    "celebdf": r"C:\Users\Yuvan Velkumar\Downloads\drive-download-20250227T132846Z-001\xception_celebdf.pth",
    "celebdf_seed25": r"C:\Users\Yuvan Velkumar\Downloads\drive-download-20250227T132846Z-001\xception_celebdf_seed25.pth",
    "uadfv": r"C:\Users\Yuvan Velkumar\Downloads\drive-download-20250227T132846Z-001\xception_uadfv.pth",
    "uadfv_seed25": r"C:\Users\Yuvan Velkumar\Downloads\drive-download-20250227T132846Z-001\xception_uadfv_seed25.pth",
    "dftimit_lq": r"C:\Users\Yuvan Velkumar\Downloads\drive-download-20250227T132846Z-001\xception_dftimit_lq.pth",
    "dftimit_lq_seed25": r"C:\Users\Yuvan Velkumar\Downloads\drive-download-20250227T132846Z-001\xception_dftimit_lq_seed25.pth",
    "dftimit_hq": r"C:\Users\Yuvan Velkumar\Downloads\drive-download-20250227T132846Z-001\xception_dftimit_hq.pth",
    "dftimit_hq_seed25": r"C:\Users\Yuvan Velkumar\Downloads\drive-download-20250227T132846Z-001\xception_dftimit_hq_seed25.pth"
}

# ✅ PositionalEncoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=100):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

# ✅ Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Store attention weights for XAI
        self.attention_weights = None

    def scaled_dot_product_attention(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        attention = torch.softmax(scores, dim=-1)
        self.attention_weights = attention  # Store for XAI
        return torch.matmul(attention, V)

    def forward(self, query, key, value):
        batch_size = query.size(0)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        output = self.scaled_dot_product_attention(Q, K, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output)

# ✅ Feed Forward
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

# ✅ Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.dropout(self.attention(x, x, x))
        x = self.norm1(x + attn_output)
        ff_output = self.dropout(self.ff(x))
        return self.norm2(x + ff_output)

# ✅ Video Transformer with XAI Support
class VideoTransformer(nn.Module):
    def __init__(self, feature_dim, num_frames, num_blocks=4, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.feature_projection = nn.Linear(feature_dim, 512)
        self.positional_encoding = PositionalEncoding(512, max_seq_length=num_frames)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(512, num_heads, d_ff, dropout) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(512)
        self.fc = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Store intermediate activations for XAI
        self.activations = {}
        self.attention_weights = []
        self.frame_scores = None

    def forward(self, x, return_attention=False):
        x = self.feature_projection(x)
        x = self.positional_encoding(x)
        
        self.attention_weights = []  # Reset weights for each forward pass
        
        for block in self.transformer_blocks:
            x = block(x)
            if hasattr(block.attention, 'attention_weights'):
                self.attention_weights.append(block.attention.attention_weights)
        
        # Global pooling
        pooled = torch.mean(x, dim=1)
        self.activations['pooled'] = pooled
        
        # Calculate frame importance
        if self.attention_weights:
            last_attn = self.attention_weights[-1]
            if last_attn.dim() == 4:  # [batch, heads, seq_len, seq_len]
                self.frame_scores = torch.mean(last_attn, dim=1).squeeze()
        
        # Explicitly get single prediction value
        output = self.fc(self.norm(pooled))
        output = self.sigmoid(output).squeeze()  # Ensure single value output
        
        if return_attention:
            return output, self.attention_weights
        return output
        
    def get_frame_importance(self):
        """Return normalized frame importance scores"""
        if len(self.attention_weights) == 0:
            return None
            
        # Use the last layer's attention for frame importance
        last_attn = self.attention_weights[-1]
        
        # Average attention weights across batch and heads
        # The shape is typically [batch, heads, seq_len, seq_len]
        if last_attn.dim() == 4:
            # Sum of attention directed to each frame (column-wise sum)
            # This indicates how much attention each frame receives
            frame_importance = last_attn.mean(dim=(0, 1)).sum(dim=0)
            
            # Normalize
            if frame_importance.sum() > 0:
                frame_importance = frame_importance / frame_importance.sum()
                
            return frame_importance
        else:
            # Fallback if shape is unexpected
            return torch.ones(x.shape[1], device=last_attn.device) / x.shape[1]

# ✅ XAI Feature Wrapper
class XaiModelWrapper(nn.Module):
    """Wrapper to make a specific feature model compatible with XAI methods"""
    def __init__(self, model_name, model):
        super().__init__()
        self.model_name = model_name
        self.model = model
        
    def forward(self, x):
        # Extract features from the model
        features = self.model.forward_features(x)
        if len(features.shape) > 2:  # If features are in spatial form (B, C, H, W)
            features = torch.mean(features, dim=[2, 3])  # Convert to (B, C)
        return features

# ✅ Load Xception Models
def load_xception_models(model_paths):
    models = {}
    for name, path in model_paths.items():
        try:
            model = timm.create_model('xception', pretrained=False)
            model.fc = nn.Linear(2048, 1)
            state_dict = torch.load(path, map_location="cpu")
            if 'last_linear.weight' in state_dict:
                state_dict['fc.weight'] = state_dict.pop('last_linear.weight')
                state_dict['fc.bias'] = state_dict.pop('last_linear.bias')
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            models[name] = model
        except Exception as e:
            print(f"Error loading model {name}: {e}")
    return models

# ✅ Frame Extraction with Visualization
def extract_frames(video_path, frame_count=10, visualize=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Error opening video file: {video_path}")
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    frame_indices = np.linspace(0, total - 1, frame_count, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    
    if visualize and frames:
        # Visualize extracted frames
        display_frames(frames, frame_indices, title="Extracted Frames")
    
    return frames

# ✅ Display Frames
def display_frames(frames, frame_indices=None, frame_scores=None, title="Frames", save_path=None):
    n_frames = len(frames)
    fig, axes = plt.subplots(1, n_frames, figsize=(n_frames * 3, 3))
    
    if n_frames == 1:
        axes = [axes]
    
    for i, (ax, frame) in enumerate(zip(axes, frames)):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax.imshow(frame_rgb)
        
        caption = f"Frame {i}"
        if frame_indices is not None:
            caption = f"Frame {frame_indices[i]}"
        
        if frame_scores is not None:
            score = frame_scores[i].item() if isinstance(frame_scores[i], torch.Tensor) else frame_scores[i]
            caption += f"\nScore: {score:.3f}"
            
            # Add red border for high importance frames
            if score > np.mean(frame_scores):
                for spine in ax.spines.values():
                    spine.set_color('red')
                    spine.set_linewidth(2)
        
        ax.set_title(caption)
        ax.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    
    plt.show()
    return fig

# ✅ Feature Extraction
def extract_combined_features(frames, models):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model in models.values():
        model.to(device)
    
    # List to store features for each frame
    all_frame_features = []
    original_tensors = []  # Store preprocessed frames for XAI visualization

    for frame in frames:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = transform(img).unsqueeze(0).to(device)
        original_tensors.append(img_tensor)
        
        # Extract and concatenate features from all models for this frame
        frame_features = []
        with torch.no_grad():
            for model in models.values():
                # Get 2048-dim feature vector from each model
                feat = model.forward_features(img_tensor)
                # Global pooling to get a single vector per frame
                if len(feat.shape) > 2:  # If features are in spatial form (B, C, H, W)
                    feat = torch.mean(feat, dim=[2, 3])  # Convert to (B, C)
                frame_features.append(feat.squeeze(0).cpu())
        
        # Concatenate all model features for this frame
        combined_frame_feature = torch.cat(frame_features)
        all_frame_features.append(combined_frame_feature)
    
    # Stack features from all frames to get sequence [frames, features]
    sequence_features = torch.stack(all_frame_features)
    
    # Add batch dimension and move to device
    return sequence_features.unsqueeze(0).to(device), original_tensors

# ✅ XAI Visualization with Grad-CAM
def generate_gradcam(model, input_tensor, target_layer_name='block12'):
    """Generate Grad-CAM visualization for a single frame"""
    try:
        # For PyTorch models without built-in hooks
        activation = {}
        gradients = {}
        
        def save_activation(name):
            def hook(module, input, output):
                activation[name] = output
            return hook
        
        def save_gradient(name):
            def hook(module, grad_input, grad_output):
                gradients[name] = grad_output[0]
            return hook
        
        # Register hooks
        if hasattr(model, target_layer_name):
            target_layer = getattr(model, target_layer_name)
        elif hasattr(model, 'blocks') and len(model.blocks) > 0:
            # Fall back to last block if specified target doesn't exist
            target_layer = model.blocks[-1]
        else:
            # Final fallback
            print("Target layer not found, using last layer available")
            for name, module in reversed(list(model.named_modules())):
                if isinstance(module, nn.Conv2d):
                    target_layer = module
                    break
            
        handle1 = target_layer.register_forward_hook(save_activation('target'))
        handle2 = target_layer.register_backward_hook(save_gradient('target'))
        
        # Forward pass
        model.zero_grad()
        output = model(input_tensor)
        
        # Backward pass
        if isinstance(output, torch.Tensor):
            output_to_backprop = output
        else:
            # If output is not a tensor (e.g., a tuple), take the first element
            output_to_backprop = output[0] if isinstance(output, tuple) else output
            
        # Create a gradient target
        one_hot = torch.zeros_like(output_to_backprop)
        one_hot.fill_(1.0)
        
        # Backward pass
        output_to_backprop.backward(gradient=one_hot, retain_graph=True)
        
        # Remove hooks
        handle1.remove()
        handle2.remove()
        
        # Generate heatmap
        activations = activation['target']
        gradients = gradients['target']
        
        # Handle different activation shapes
        if len(activations.shape) == 4:  # (B, C, H, W)
            # Spatial features from conv layers
            weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
            heatmap = torch.sum(weights * activations, dim=1, keepdim=True)
        else:
            # Non-spatial features
            print("Warning: Activation shape not compatible with Grad-CAM")
            return None
            
        heatmap = torch.relu(heatmap)
        
        # Normalize heatmap
        heatmap = heatmap - heatmap.min()
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        return None

def visualize_frame_attributions(frames, model, model_name, save_dir=None):
    """Generate visual explanations for why a frame was classified as fake/real"""
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    for i, frame in enumerate(frames):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        try:
            # Generate Grad-CAM
            heatmap = generate_gradcam(model, img_tensor)
            
            if heatmap is not None:
                # Resize heatmap to match original image
                heatmap = heatmap.cpu().squeeze().numpy()
                heatmap = cv2.resize(heatmap, (299, 299))
                
                # Convert heatmap to RGB
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                
                # Convert original frame to RGB and resize
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (299, 299))
                
                # Superimpose heatmap on original image
                superimposed = cv2.addWeighted(frame_resized, 0.6, heatmap, 0.4, 0)
                
                # Display
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(frame_rgb)
                plt.title(f"Original Frame {i}")
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(superimposed)
                plt.title(f"XAI Heatmap - {model_name}")
                plt.axis('off')
                
                if save_dir:
                    plt.savefig(os.path.join(save_dir, f"xai_frame_{i}_{model_name}.png"))
                
                plt.show()
        except Exception as e:
            print(f"Error visualizing frame {i}: {e}")

# ✅ Interpret Transformer Attention
def visualize_attention_weights(transformer_model, frames, frame_importance, save_path=None):
    """Visualize which frames the transformer pays most attention to"""
    if frame_importance is None:
        print("No attention weights available for visualization")
        return
    
    # Convert to numpy if needed
    if isinstance(frame_importance, torch.Tensor):
        frame_importance = frame_importance.cpu().numpy()
    
    # Create visualization
    display_frames(frames, frame_scores=frame_importance, 
                   title="Frame Importance in Deepfake Detection", 
                   save_path=save_path)
    
    # Plot frame importance as a bar chart
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(frame_importance)), frame_importance)
    plt.xlabel("Frame Index")
    plt.ylabel("Importance Score")
    plt.title("Frame Importance Scores")
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path.replace('.png', '_barchart.png'))
    
    plt.show()

# ✅ Simple attention-based frame importance
def compute_simple_frame_importance(frames, models):
    """Compute frame importance based on individual model predictions"""
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Store predictions for each frame
    frame_scores = []
    
    # Get predictions for each frame
    for frame in frames:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Get average prediction across models
        frame_preds = []
        with torch.no_grad():
            for model in models.values():
                model.to(device)
                model.eval()
                # Get prediction from the model
                try:
                    pred = torch.sigmoid(model(img_tensor)).item()
                    frame_preds.append(pred)
                except Exception as e:
                    print(f"Error getting prediction: {e}")
                    continue
        
        # Store average prediction as frame importance
        if frame_preds:
            avg_pred = np.mean(frame_preds) 
            # How far from 0.5 (neutral) - farther is more important
            importance = abs(avg_pred - 0.5) * 2  # Scale to 0-1
            frame_scores.append(importance)
        else:
            frame_scores.append(0.0)
    
    # Normalize importance scores
    if frame_scores:
        total = sum(frame_scores)
        if total > 0:
            frame_scores = [score/total for score in frame_scores]
    
    return frame_scores

# ✅ Run Inference with XAI
def detect_deepfake(video_path, models, transformer, output_dir=None):
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Extract frames with visualization
    frames = extract_frames(video_path, frame_count=10, visualize=True)
    
    # Extract features
    features, original_tensors = extract_combined_features(frames, models)
    
    # Get prediction from transformer
    with torch.no_grad():
        output = transformer(features)
    
    # Ensure we have a single scalar value
    if isinstance(output, torch.Tensor):
        result = output.item()
    else:
        result = output  # In case it's already a float
    
    prediction = "Fake" if result > 0.5 else "Real"
    confidence = result if result > 0.5 else 1 - result
    
    print(f"Prediction: {result:.4f} → {prediction} (Confidence: {confidence:.2%})")
    
    # Get frame importance
    frame_importance = transformer.get_frame_importance()
    
    # If transformer attention failed, fall back to simpler method
    if frame_importance is None:
        print("Using fallback method for frame importance")
        frame_importance = compute_simple_frame_importance(frames, models)
    
    # Visualize attention weights
    print("\nVisualizing frame importance...")
    save_path = os.path.join(output_dir, "frame_importance.png") if output_dir else None
    visualize_attention_weights(transformer, frames, frame_importance, save_path)
    
    # Generate XAI visualizations for key frames
    if output_dir:
        # Use the most important model for visualization
        key_model_name = "celebdf"  # You can change this or make it dynamic
        if key_model_name in models:
            print(f"\nGenerating XAI visualizations using {key_model_name} model...")
            visualize_frame_attributions(frames, models[key_model_name], key_model_name, output_dir)
    
    return prediction, confidence, frames, frame_importance

# ✅ Create XAI Model Visualization
def visualize_model_contributions(video_path, models, output_dir=None):
    """Visualize how different models contribute to the final prediction"""
    frames = extract_frames(video_path, frame_count=5, visualize=False)
    
    # Create a dictionary to store individual model predictions
    model_predictions = {}
    
    # Preprocessing transform
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Process each frame with each model
    for model_name, model in models.items():
        model.to(device)
        model.eval()
        
        frame_preds = []
        for frame in frames:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            try:
                with torch.no_grad():
                    # Get prediction from the model
                    pred = torch.sigmoid(model(img_tensor)).item()
                    frame_preds.append(pred)
            except Exception as e:
                print(f"Error with model {model_name}: {e}")
                continue
        
        # Store average prediction for this model
        if frame_preds:
            avg_pred = np.mean(frame_preds)
            model_predictions[model_name] = avg_pred
    
    # Visualize model contributions if we have data
    if model_predictions:
        plt.figure(figsize=(12, 6))
        models_list = list(model_predictions.keys())
        preds_list = [model_predictions[m] for m in models_list]
        
        bars = plt.bar(models_list, preds_list, color=['blue' if p < 0.5 else 'red' for p in preds_list])
        
        plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.7)
        plt.xlabel('Models')
        plt.ylabel('Fake Probability')
        plt.title('Model Contributions to Deepfake Detection')
        plt.xticks(rotation=45, ha='right')
        
        # Add prediction labels on top of bars
        for bar, pred in zip(bars, preds_list):
            label = f"{pred:.2f}"
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, 
                    label, ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, "model_contributions.png"))
        
        plt.show()
    
    return model_predictions

# ✅ Run
if __name__ == "__main__":
    # Create output directory for visualizations
    output_dir = "deepfake_analysis_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    print("Loading models...")
    models = load_xception_models(xception_model_paths)
    
    # Initialize transformer
    transformer = VideoTransformer(feature_dim=2048 * len(models), num_frames=10).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Example video
    video_path = r"C:\Users\Yuvan Velkumar\Downloads\Celeb-DF-v2\Celeb-synthesis\id9_id4_0006.mp4"
    
    # Run deepfake detection with XAI
    print("\nDetecting deepfakes with explanations...")
    prediction, confidence, frames, frame_importance = detect_deepfake(video_path, models, transformer, output_dir)
    
    # Visualize model contributions
    print("\nAnalyzing individual model contributions...")
    model_contributions = visualize_model_contributions(video_path, models, output_dir)
    
    print(f"\nFinal prediction: {prediction} with {confidence:.2%} confidence")
    print(f"Visualization outputs saved to: {output_dir}")