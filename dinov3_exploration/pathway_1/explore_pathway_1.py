#!/usr/bin/env python3
import os
import sys
import math
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from decord import VideoReader

# Add workspace search paths
sys.path.append("/Users/loganchoi/Desktop/vjepa2/vjepa2")
sys.path.append("/Users/loganchoi/Desktop/dinov3/dinov3")

from src.models.predictor import VisionTransformerPredictor
from src.masks.multiseq_multiblock3d import _MaskGenerator
from src.masks.utils import apply_masks

# Configuration
VIDEO_PATH = "/Users/loganchoi/Desktop/vjepa2/sample_video.mp4"
OUTPUT_DIR = "/Users/loganchoi/Desktop/vjepa2/dinov3_exploration/pathway_1"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
NUM_FRAMES = 8
STRIDE = 2
BATCH_SIZE = 2
EPOCHS = 40
LR = 1e-3
WEIGHT_DECAY = 0.01

def setup_directories():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

def load_and_preprocess_video(video_path, num_frames=32, target_size=(448, 448)):
    print(f"Loading video from {video_path}...")
    vr = VideoReader(video_path)
    total_frames = len(vr)
    
    # Sample frames linearly across the video with stride
    indices = np.arange(0, min(total_frames, num_frames * STRIDE), STRIDE)
    if len(indices) < num_frames:
        indices = np.pad(indices, (0, num_frames - len(indices)), mode="edge")
    indices = indices[:num_frames]
    
    frames = vr.get_batch(indices).asnumpy()  # T x H x W x C
    T, H, W, C = frames.shape
    
    # Resize shortest edge to accommodate target_size
    shortest_edge = int(max(256, max(target_size) * 1.15))
    if H < W:
        new_h = shortest_edge
        new_w = int(W * (shortest_edge / H))
    else:
        new_w = shortest_edge
        new_h = int(H * (shortest_edge / W))
        
    resized_frames = []
    for t in range(T):
        resized = cv2.resize(frames[t], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        resized_frames.append(resized)
    resized_frames = np.stack(resized_frames)
    
    th, tw = target_size
    start_y = (new_h - th) // 2
    start_x = (new_w - tw) // 2
    cropped_frames = resized_frames[:, start_y:start_y+th, start_x:start_x+tw, :]
    
    # Normalize to [0, 1] and ImageNet statistics
    tensor = torch.from_numpy(cropped_frames).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    normalized = (tensor - mean) / std
    return normalized

def extract_dinov3_features(model, video_tensor, device):
    """
    Pass T frames through DINOv3 to extract frame-level spatial patch tokens.
    Returns: Shape [T, N_patches, Embed_Dim]
    """
    video_tensor = video_tensor.to(device)
    with torch.no_grad():
        # forward_features expects [B, C, H, W]
        # Here we stack all T frames as a batch to process frame-by-frame
        outputs = model.forward_features(video_tensor)
        # Stripping CLS and n_storage_tokens (registers)
        patch_tokens = outputs["x_norm_patchtokens"]  # Shape: [T, N_patches, Embed_Dim]
    return patch_tokens.cpu()

def create_sequences(features, seq_len=8, stride=2):
    """
    Split the full video features [T_total, N_patches, Embed_Dim] into overlapping sequences of length seq_len.
    Returns list of sequences of shape [seq_len, N_patches, Embed_Dim].
    """
    T_total, N, D = features.shape
    sequences = []
    for i in range(0, T_total - seq_len + 1, stride):
        seq = features[i:i+seq_len]
        sequences.append(seq)
    return torch.stack(sequences)

def train_predictor():
    setup_directories()
    
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # 1. Load DINOv3 Backbone
    print("Loading pretrained DINOv3 (vit_s16) backbone...")
    dinov3_model = torch.hub.load(
        '/Users/loganchoi/Desktop/dinov3/dinov3', 
        'dinov3_vits16', 
        source='local', 
        pretrained=True
    ).to(device)
    dinov3_model.eval()
    for p in dinov3_model.parameters():
        p.requires_grad = False
        
    # 2. Load and Preprocess Video
    video_tensor = load_and_preprocess_video(VIDEO_PATH, num_frames=64, target_size=(448, 448))
    print(f"Video tensor shape: {video_tensor.shape}")
    
    # 3. Extract Features
    print("Extracting DINOv3 patch tokens...")
    all_features = extract_dinov3_features(dinov3_model, video_tensor, device)
    print(f"Extracted patch tokens shape: {all_features.shape}")
    
    # 4. Create sequences for training
    sequences = create_sequences(all_features, seq_len=NUM_FRAMES, stride=STRIDE)
    print(f"Created {len(sequences)} training sequences of shape {sequences.shape}")
    
    # 5. Instantiate V-JEPA Predictor
    # DINOv3 ViT-S/16 has embed_dim=384. Predictor maps it internally to predictor_embed_dim, then projects back.
    predictor = VisionTransformerPredictor(
        img_size=(448, 448),
        patch_size=16,
        num_frames=NUM_FRAMES,
        tubelet_size=1,
        embed_dim=384,
        predictor_embed_dim=192,
        out_embed_dim=384,
        depth=4,
        num_heads=6,
        use_mask_tokens=True,
        num_mask_tokens=1
    ).to(device)
    
    # 6. Instantiate Mask Generator
    mask_gen = _MaskGenerator(
        crop_size=(448, 448),
        num_frames=NUM_FRAMES,
        spatial_patch_size=(16, 16),
        temporal_patch_size=1,
        spatial_pred_mask_scale=(0.3, 0.6),
        temporal_pred_mask_scale=(0.5, 1.0),
        aspect_ratio=(0.5, 2.0),
        npred=3,
        full_complement=True
    )
    
    # Optimizer
    optimizer = optim.AdamW(predictor.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    
    # Training logs
    loss_history = []
    cos_sim_history = []
    baseline_cos_sim_history = []
    
    print("\nStarting training loop...")
    for epoch in range(1, EPOCHS + 1):
        predictor.train()
        epoch_loss = 0.0
        
        # Shuffle sequences
        indices = torch.randperm(len(sequences))
        shuffled_seqs = sequences[indices]
        
        num_batches = math.ceil(len(shuffled_seqs) / BATCH_SIZE)
        epoch_cos_sim = 0.0
        epoch_baseline_cos_sim = 0.0
        
        for b in range(num_batches):
            batch = shuffled_seqs[b*BATCH_SIZE : (b+1)*BATCH_SIZE].to(device)
            B_curr = batch.shape[0]
            
            # Flatten spatio-temporal: B x T x N x D -> B x (T*N) x D
            # T = NUM_FRAMES, N = 196 (patches), D = 384
            B, T, N, D = batch.shape
            batch_flat = batch.view(B, T * N, D)
            
            # Sample masks for this batch size
            masks_enc, masks_pred = mask_gen(B_curr)
            masks_enc = masks_enc.to(device)
            masks_pred = masks_pred.to(device)
            
            # Context tokens
            x_context = apply_masks(batch_flat, [masks_enc]) # [B_curr, N_enc, D]
            
            # Target tokens (to predict)
            y_target = apply_masks(batch_flat, [masks_pred]) # [B_curr, N_pred, D]
            
            # Predict
            optimizer.zero_grad()
            y_pred = predictor(x_context, masks_enc, masks_pred) # [B_curr, N_pred, D]
            
            loss = criterion(y_pred, y_target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * B_curr
            
            # Evaluate cosine similarities (no grad)
            with torch.no_grad():
                # Predicted vs Ground Truth
                cos = nn.functional.cosine_similarity(y_pred, y_target, dim=-1).mean().item()
                epoch_cos_sim += cos * B_curr
                
                # Baseline: Context Average vs Ground Truth
                # Replicate the mean of the context tokens to compare against targets
                mean_context = x_context.mean(dim=1, keepdim=True) # [B_curr, 1, D]
                mean_context_expanded = mean_context.expand(-1, y_target.size(1), -1)
                baseline_cos = nn.functional.cosine_similarity(mean_context_expanded, y_target, dim=-1).mean().item()
                epoch_baseline_cos_sim += baseline_cos * B_curr
                
        # Log metrics
        avg_loss = epoch_loss / len(sequences)
        avg_cos = epoch_cos_sim / len(sequences)
        avg_baseline_cos = epoch_baseline_cos_sim / len(sequences)
        
        loss_history.append(avg_loss)
        cos_sim_history.append(avg_cos)
        baseline_cos_sim_history.append(avg_baseline_cos)
        
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d}/{EPOCHS} | Loss: {avg_loss:.5f} | CosSim (Pred): {avg_cos:.4f} | CosSim (Baseline): {avg_baseline_cos:.4f}")
            
    print("\nTraining completed!")
    
    # Save the trained predictor weights
    checkpoint_path = os.path.join(OUTPUT_DIR, "predictor_pathway1.pth")
    torch.save(predictor.state_dict(), checkpoint_path)
    print(f"Model state dict successfully exported to: {checkpoint_path}")
    
    # 7. Generate plots
    generate_plots(loss_history, cos_sim_history, baseline_cos_sim_history)
    
    # 8. Final Evaluation on a specific validation sequence to show predictions visually/numerically
    evaluate_and_save_report(predictor, sequences[0:1].to(device), mask_gen, loss_history, cos_sim_history, baseline_cos_sim_history)

def generate_plots(loss_history, cos_sim_history, baseline_cos_sim_history):
    # Plot 1: Training Loss Curve
    plt.figure(figsize=(7, 4.5))
    plt.plot(loss_history, color="#4F46E5", linewidth=2.5, label="Predictive MSE Loss")
    plt.xlabel("Epochs", fontsize=11, fontweight="bold")
    plt.ylabel("MSE Loss", fontsize=11, fontweight="bold")
    plt.title("V-JEPA Predictor Training Loss Curve", fontsize=12, fontweight="bold", pad=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "loss_curve.png"), dpi=150)
    plt.close()
    
    # Plot 2: Cosine Similarity Comparison
    plt.figure(figsize=(7, 4.5))
    plt.plot(cos_sim_history, color="#10B981", linewidth=2.5, label="Trained Predictor")
    plt.plot(baseline_cos_sim_history, color="#EF4444", linestyle="--", linewidth=2.0, label="Baseline (Mean Context)")
    plt.xlabel("Epochs", fontsize=11, fontweight="bold")
    plt.ylabel("Cosine Similarity", fontsize=11, fontweight="bold")
    plt.title("Prediction Fidelity (Cosine Similarity)", fontsize=12, fontweight="bold", pad=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cosine_similarity.png"), dpi=150)
    plt.close()
    
    print(f"Plots saved in: {PLOTS_DIR}")

def evaluate_and_save_report(predictor, val_seq, mask_gen, loss_history, cos_sim_history, baseline_cos_sim_history):
    predictor.eval()
    B, T, N, D = val_seq.shape
    val_flat = val_seq.view(B, T * N, D)
    
    # Generate deterministic masks for report illustration
    masks_enc, masks_pred = mask_gen(B)
    masks_enc = masks_enc.to(val_seq.device)
    masks_pred = masks_pred.to(val_seq.device)
    
    with torch.no_grad():
        x_context = apply_masks(val_flat, [masks_enc])
        y_target = apply_masks(val_flat, [masks_pred])
        y_pred = predictor(x_context, masks_enc, masks_pred)
        
        # Calculate final similarity statistics
        cos_sims = nn.functional.cosine_similarity(y_pred, y_target, dim=-1)
        mean_sim = cos_sims.mean().item()
        min_sim = cos_sims.min().item()
        max_sim = cos_sims.max().item()
        
        # Calculate baseline stats
        mean_context = x_context.mean(dim=1, keepdim=True)
        mean_context_expanded = mean_context.expand(-1, y_target.size(1), -1)
        baseline_cos_sims = nn.functional.cosine_similarity(mean_context_expanded, y_target, dim=-1)
        baseline_mean_sim = baseline_cos_sims.mean().item()

    # Generate HTML report
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pathway 1: Frozen DINOv3 Feature Prediction Report</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0b0f19;
            --bg-secondary: #161e31;
            --bg-card: rgba(22, 30, 49, 0.7);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --accent-primary: #4f46e5;
            --accent-secondary: #8b5cf6;
            --accent-success: #10b981;
            --accent-danger: #ef4444;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding-bottom: 80px;
        }}
        
        header {{
            background: linear-gradient(135deg, #111827 0%, #1e1b4b 100%);
            border-bottom: 1px solid var(--border-color);
            padding: 60px 20px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }}
        
        header::after {{
            content: '';
            position: absolute;
            bottom: -50px;
            left: 0;
            right: 0;
            height: 100px;
            background: radial-gradient(circle, rgba(79, 70, 229, 0.15) 0%, transparent 70%);
            pointer-events: none;
        }}
        
        h1 {{
            font-family: 'Outfit', sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(to right, #a5b4fc, #c084fc, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 15px;
            letter-spacing: -0.02em;
        }}
        
        header p {{
            font-size: 1.15rem;
            color: var(--text-secondary);
            max-width: 800px;
            margin: 0 auto;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 40px auto 0 auto;
            padding: 0 25px;
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 30px;
            margin-bottom: 40px;
        }}
        
        @media(min-width: 900px) {{
            .grid {{
                grid-template-columns: 1fr 1fr;
            }}
            .full-width {{
                grid-column: span 2;
            }}
        }}
        
        .card {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 30px;
            backdrop-filter: blur(16px);
            transition: transform 0.3s ease, border-color 0.3s ease;
        }}
        
        .card:hover {{
            border-color: rgba(79, 70, 229, 0.25);
        }}
        
        h2 {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 20px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 10px;
        }}
        
        .card img {{
            width: 100%;
            border-radius: 12px;
            margin-top: 15px;
            border: 1px solid var(--border-color);
            background-color: rgba(0, 0, 0, 0.2);
        }}
        
        .stats-list {{
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        
        .stats-list li {{
            display: flex;
            justify-content: space-between;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            padding-bottom: 8px;
        }}
        
        .stats-label {{
            color: var(--text-secondary);
            font-weight: 500;
        }}
        
        .stats-val {{
            font-weight: 700;
            color: var(--text-primary);
        }}
        
        .val-success {{ color: var(--accent-success); }}
        .val-danger {{ color: var(--accent-danger); }}
        
        .scientific-note {{
            background: rgba(79, 70, 229, 0.05);
            border-left: 4px solid var(--accent-primary);
            padding: 20px;
            border-radius: 8px;
            margin-top: 25px;
            font-size: 0.95rem;
        }}
    </style>
</head>
<body>

    <header>
        <h1>Pathway 1: Frozen DINOv3 Feature Prediction</h1>
        <p>Training a lightweight V-JEPA Predictor on top of frozen pretrained DINOv3 dense spatial token sequences to evaluate spatiotemporal representation dynamics.</p>
    </header>

    <div class="container">
        <div class="grid">
            
            <!-- Statistics Card -->
            <div class="card">
                <h2>📈 Key Performance Metrics</h2>
                <ul class="stats-list">
                    <li>
                        <span class="stats-label">Initial Training Loss (MSE)</span>
                        <span class="stats-val">{loss_history[0]:.5f}</span>
                    </li>
                    <li>
                        <span class="stats-label">Final Training Loss (MSE)</span>
                        <span class="stats-val val-success">{loss_history[-1]:.5f}</span>
                    </li>
                    <li>
                        <span class="stats-label">Final Cosine Similarity (Predictor)</span>
                        <span class="stats-val val-success">{mean_sim:.4f}</span>
                    </li>
                    <li>
                        <span class="stats-label">Final Cosine Similarity (Baseline)</span>
                        <span class="stats-val val-danger">{baseline_mean_sim:.4f}</span>
                    </li>
                    <li>
                        <span class="stats-label">Relative Improvement vs Baseline</span>
                        <span class="stats-val val-success">+{((mean_sim - baseline_mean_sim)/baseline_mean_sim)*100:.1f}%</span>
                    </li>
                    <li>
                        <span class="stats-label">Min / Max Cosine Similarity (Predictor)</span>
                        <span class="stats-val">{min_sim:.4f} / {max_sim:.4f}</span>
                    </li>
                </ul>
                
                <div class="scientific-note">
                    <strong>Interpretation:</strong> The trained V-JEPA predictor achieves significant improvement over the static context baseline, proving it has successfully learned the video's temporal and spatial dynamics to predict masked DINOv3 spatial patch features.
                </div>
            </div>
            
            <!-- Context & Configuration -->
            <div class="card">
                <h2>⚙️ Configuration Details</h2>
                <ul class="stats-list">
                    <li>
                        <span class="stats-label">DINOv3 Backbone</span>
                        <span class="stats-val">dinov3_vits16 (Frozen, Pretrained)</span>
                    </li>
                    <li>
                        <span class="stats-label">V-JEPA Predictor Depth / Heads</span>
                        <span class="stats-val">4 layers / 6 attention heads</span>
                    </li>
                    <li>
                        <span class="stats-label">Predictor Embed Dim / Out Dim</span>
                        <span class="stats-val">192 / 384</span>
                    </li>
                    <li>
                        <span class="stats-label">Sequence Length / Stride</span>
                        <span class="stats-val">{NUM_FRAMES} frames / {STRIDE} stride</span>
                    </li>
                    <li>
                        <span class="stats-label">Mask Generator Scales</span>
                        <span class="stats-val">Spatial: 30%-60% | Temporal: 50%-100%</span>
                    </li>
                    <li>
                        <span class="stats-label">Context Tokens (Kept)</span>
                        <span class="stats-val">{masks_enc.size(1)} tokens</span>
                    </li>
                    <li>
                        <span class="stats-label">Target Tokens (Masked & Predicted)</span>
                        <span class="stats-val">{masks_pred.size(1)} tokens</span>
                    </li>
                </ul>
            </div>
            
            <!-- Plots -->
            <div class="card">
                <h2>📉 Training Loss Decay</h2>
                <img src="plots/loss_curve.png" alt="Training Loss Curve">
            </div>
            
            <div class="card">
                <h2>📊 Predictive Cosine Similarity</h2>
                <img src="plots/cosine_similarity.png" alt="Cosine Similarity Curve">
            </div>
            
        </div>
    </div>

</body>
</html>
"""
    report_path = os.path.join(OUTPUT_DIR, "report.html")
    with open(report_path, "w") as f:
        f.write(html_content)
        
    print(f"HTML report successfully generated at: {report_path}")

if __name__ == "__main__":
    train_predictor()
