#!/usr/bin/env python3
"""
Option 3: Hierarchical Layer Representation Analysis for V-JEPA 2.1.
This script extracts hidden states from all 24 Transformer layers of V-JEPA 2.1,
trains linear classification probes for spatial structure and motion magnitude,
performs Representational Similarity Analysis (RSA) to measure context leakage,
and creates layer-wise PCA visual comparison overlays.
"""

import os
import argparse
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from decord import VideoReader
from transformers import AutoModel, AutoVideoProcessor

SAMPLE_VIDEO_URL = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/bowling/-WH-lxmGJVY_000005_000015.mp4"
SAMPLE_VIDEO_PATH = "sample_video.mp4"
HF_MODEL_NAME = "facebook/vjepa2-vitl-fpc64-256"


def download_sample_video(url, path):
    if not os.path.exists(path):
        print(f"Downloading sample video from: {url}")
        urllib.request.urlretrieve(url, path)
        print(f"Downloaded sample video to: {path}")
    else:
        print(f"Sample video already exists at: {path}")


def load_video_frames(video_path, num_frames=64, stride=2):
    print(f"Loading video from: {video_path}")
    vr = VideoReader(video_path)
    total_frames = len(vr)
    print(f"Total video frames: {total_frames}, FPS: {vr.get_avg_fps():.2f}")

    # Calculate frame indices (sample num_frames frames with specified stride)
    max_index = min(total_frames, num_frames * stride)
    frame_idx = np.arange(0, max_index, stride)
    
    # Pad index if video is too short
    if len(frame_idx) < num_frames:
        frame_idx = np.pad(frame_idx, (0, num_frames - len(frame_idx)), mode="edge")

    # Keep only the requested number of frames
    frame_idx = frame_idx[:num_frames]

    print(f"Sampling frame indices: {frame_idx}")
    video_data = vr.get_batch(frame_idx).asnumpy()  # T x H x W x C
    return video_data


def resize_and_center_crop(frames, target_size=(256, 256), shortest_edge=292):
    """
    Center crops and resizes the original video frames so they align with model tokens.
    """
    T, H, W, C = frames.shape
    
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
    
    return cropped_frames


def compute_motion_magnitude(video_tensor, tubelet_size=2, patch_size=16):
    """
    Computes motion magnitude for each spatial-temporal tubelet token.
    video_tensor shape: [T, C, H, W]
    Returns a tensor of shape [T_tokens, H_tokens, W_tokens]
    """
    print("Computing dense motion maps...")
    # Compute absolute frame difference over the channel dimension
    video_tensor_float = video_tensor.float()
    diff = torch.abs(video_tensor_float[1:] - video_tensor_float[:-1]).mean(dim=1)  # [T-1, H, W]
    diff = torch.cat([diff, diff[-1:]], dim=0)  # [T, H, W] to match original frames length
    
    # 3D average pool difference maps to match token grid resolution
    # Input to avg_pool3d: [B, C, T, H, W]
    motion_tokens = F.avg_pool3d(
        diff.unsqueeze(0).unsqueeze(0),
        kernel_size=(tubelet_size, patch_size, patch_size),
        stride=(tubelet_size, patch_size, patch_size)
    ).squeeze(0).squeeze(0)  # [T_tokens, H_tokens, W_tokens]
    
    return motion_tokens


class LinearProbes(nn.Module):
    def __init__(self, embed_dim=1024):
        super().__init__()
        # Predict row coordinate (0 to 15)
        self.row_pred = nn.Linear(embed_dim, 16)
        # Predict column coordinate (0 to 15)
        self.col_pred = nn.Linear(embed_dim, 16)
        # Predict motion magnitude bin (0 to 4)
        self.motion_pred = nn.Linear(embed_dim, 5)

    def forward(self, x):
        return self.row_pred(x), self.col_pred(x), self.motion_pred(x)


def train_and_eval_probes(hidden_states, row_targets, col_targets, motion_targets, train_idx, val_idx, epochs=60, lr=0.005, device="cpu"):
    """
    Trains linear probes on hidden state representations for structure and motion classification.
    """
    X = hidden_states.to(device)
    y_row = row_targets.to(device)
    y_col = col_targets.to(device)
    y_mot = motion_targets.to(device)
    
    # Convert inputs to float32
    X = X.float()
    
    probes = LinearProbes(embed_dim=X.shape[-1]).to(device)
    optimizer = torch.optim.Adam(probes.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Split
    X_train, X_val = X[train_idx], X[val_idx]
    y_row_train, y_row_val = y_row[train_idx], y_row[val_idx]
    y_col_train, y_col_val = y_col[train_idx], y_col[val_idx]
    y_mot_train, y_mot_val = y_mot[train_idx], y_mot[val_idx]
    
    # Training Loop
    probes.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out_row, out_col, out_mot = probes(X_train)
        loss = criterion(out_row, y_row_train) + criterion(out_col, y_col_train) + criterion(out_mot, y_mot_train)
        loss.backward()
        optimizer.step()
        
    # Validation Eval
    probes.eval()
    with torch.no_grad():
        out_row, out_col, out_mot = probes(X_val)
        
        # Structure accuracies
        pred_row = out_row.argmax(dim=-1)
        pred_col = out_col.argmax(dim=-1)
        row_acc = (pred_row == y_row_val).float().mean().item()
        col_acc = (pred_col == y_col_val).float().mean().item()
        structure_acc = (row_acc + col_acc) / 2.0
        
        # Motion accuracy
        pred_mot = out_mot.argmax(dim=-1)
        motion_acc = (pred_mot == y_mot_val).float().mean().item()
        
    return structure_acc, motion_acc


def compute_rsa_leakage(hidden_states, motion_flat, fg_threshold_pct=0.15, bg_threshold_pct=0.30):
    """
    Computes context leakage via Representational Similarity Analysis (RSA).
    Measures the average cosine similarity between high-motion foreground and static background tokens.
    """
    # L2 normalized features
    feats_norm = F.normalize(hidden_states.float(), p=2, dim=-1)
    
    # Identify foreground/background indices
    n_tokens = len(motion_flat)
    sorted_idx = torch.argsort(motion_flat)
    
    # Bottom bg_threshold_pct are background
    bg_count = int(n_tokens * bg_threshold_pct)
    bg_idx = sorted_idx[:bg_count]
    
    # Top fg_threshold_pct are foreground
    fg_count = int(n_tokens * fg_threshold_pct)
    fg_idx = sorted_idx[-fg_count:]
    
    # Slices
    feats_fg = feats_norm[fg_idx]  # [N_fg, D]
    feats_bg = feats_norm[bg_idx]  # [N_bg, D]
    
    # Cosine similarity matrix between fg and bg
    sim_matrix = torch.matmul(feats_fg, feats_bg.T)  # [N_fg, N_bg]
    mean_leakage = sim_matrix.mean().item()
    
    # Intra-class similarities
    fg_coherence = torch.matmul(feats_fg, feats_fg.T).mean().item()
    bg_coherence = torch.matmul(feats_bg, feats_bg.T).mean().item()
    
    return mean_leakage, fg_coherence, bg_coherence


def compute_layer_pca(features, t_p, h_p, w_p):
    """
    Computes PCA (top 3 components) globally across all tokens in a layer.
    Returns projected components reshaped to (T_p, H_p, W_p, 3) and normalized to [0, 1].
    """
    features_float = features.float()
    mean = features_float.mean(dim=0, keepdim=True)
    centered = features_float - mean
    
    # SVD approximation
    U, S, V = torch.pca_lowrank(centered, q=3, center=False)
    projected = torch.matmul(centered, V[:, :3])  # (T_p * H_p * W_p, 3)
    projected = projected.view(t_p, h_p, w_p, 3)
    
    # Normalize with robust quantiles
    normalized = torch.zeros_like(projected)
    for i in range(3):
        comp = projected[..., i]
        q_low = torch.quantile(comp, 0.01)
        q_high = torch.quantile(comp, 0.99)
        denom = q_high - q_low if q_high > q_low else 1e-8
        clamped = torch.clamp(comp, min=q_low, max=q_high)
        normalized[..., i] = (clamped - q_low) / denom
        
    return normalized


def interpolate_and_overlay(projected_pca, original_frames, alpha=0.5):
    """
    Interpolates PCA feature maps (T_p, H_p, W_p, 3) trilinearly to match original frames (T, H, W, 3).
    Saves the alpha-blended overlay video frames.
    """
    T, H, W, C = original_frames.shape
    
    # Permute to (1, 3, T_p, H_p, W_p)
    pca_tensor = projected_pca.permute(3, 0, 1, 2).unsqueeze(0)
    interpolated = F.interpolate(
        pca_tensor,
        size=(T, H, W),
        mode='trilinear',
        align_corners=False
    )
    
    # Convert back to (T, H, W, 3) numpy
    pca_numpy = interpolated.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
    pca_numpy = np.clip(pca_numpy * 255.0, 0, 255).astype(np.uint8)
    
    # Blend overlay
    overlayed = (original_frames * (1 - alpha) + pca_numpy * alpha).astype(np.uint8)
    return overlayed, pca_numpy


def generate_grid_comparison_video(original, overlays_dict, output_path, fps=15.0):
    """
    Creates a 2x3 video grid:
    Row 0: Original, Layer 2 Overlay, Layer 8 Overlay
    Row 1: Layer 14 Overlay, Layer 20 Overlay, Layer 24 Overlay
    """
    T, H, W, C = original.shape
    grid_h, grid_w = H * 2, W * 3
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (grid_w, grid_h))
    
    for t in range(T):
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        # Helper to add layer name label text
        def draw_label(img, text):
            labeled = img.copy()
            cv2.putText(labeled, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            return labeled
            
        # Top Row
        grid[0:H, 0:W] = draw_label(original[t], "Original")
        grid[0:H, W:2*W] = draw_label(overlays_dict[2][t], "Layer 2 PCA")
        grid[0:H, 2*W:3*W] = draw_label(overlays_dict[8][t], "Layer 8 PCA")
        
        # Bottom Row
        grid[H:2*H, 0:W] = draw_label(overlays_dict[14][t], "Layer 14 PCA")
        grid[H:2*H, W:2*W] = draw_label(overlays_dict[20][t], "Layer 20 PCA")
        grid[H:2*H, 2*W:3*W] = draw_label(overlays_dict[24][t], "Layer 24 PCA")
        
        # Convert RGB to BGR for OpenCV
        grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
        out.write(grid_bgr)
        
    out.release()
    print(f"Saved comparison grid video to: {output_path}")


def save_keyframe_layers_grid(original, overlays_dict, output_path, num_keyframes=5):
    """
    Saves a static comparison grid showing layers side-by-side for keyframe indexes.
    """
    T, H, W, C = original.shape
    indices = np.linspace(0, T - 1, num_keyframes, dtype=int)
    layers = [2, 8, 14, 20, 24]
    
    # 6 rows (Original + 5 layers), num_keyframes columns
    fig, axes = plt.subplots(6, num_keyframes, figsize=(2.4 * num_keyframes, 13.5))
    
    for idx, frame_idx in enumerate(indices):
        # Row 0: Original
        axes[0, idx].imshow(original[frame_idx])
        axes[0, idx].axis('off')
        axes[0, idx].set_title(f"Frame {frame_idx}", fontsize=10)
        if idx == 0:
            axes[0, idx].set_ylabel("Original", fontsize=11, fontweight='bold', labelpad=12)
            axes[0, idx].axis('on')
            axes[0, idx].set_xticks([])
            axes[0, idx].set_yticks([])
            
        for l_idx, layer in enumerate(layers):
            ax = axes[l_idx + 1, idx]
            ax.imshow(overlays_dict[layer][frame_idx])
            ax.axis('off')
            if idx == 0:
                ax.set_ylabel(f"Layer {layer} PCA", fontsize=11, fontweight='bold', labelpad=12)
                ax.axis('on')
                ax.set_xticks([])
                ax.set_yticks([])
                
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved keyframe layers grid image to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Option 3: V-JEPA Hierarchical Layer Representation Analysis")
    parser.add_argument("--video_path", type=str, default=SAMPLE_VIDEO_PATH, help="Path to input video")
    parser.add_argument("--model_name", type=str, default=HF_MODEL_NAME, help="Hugging Face model ID")
    parser.add_argument("--output_dir", type=str, default="option3/visualizations", help="Output directory")
    parser.add_argument("--num_frames", type=int, default=64, help="Number of frames to sample")
    parser.add_argument("--stride", type=int, default=2, help="Sampling stride")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha blending weight for PCA overlay")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Download sample video if needed
    download_sample_video(SAMPLE_VIDEO_URL, args.video_path)
    
    # 2. Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS device.")
    else:
        device = torch.device("cpu")
        print("Using CPU device.")

    # 3. Load processor and model
    print(f"Loading processor: {args.model_name}...")
    processor = AutoVideoProcessor.from_pretrained(args.model_name)
    print(f"Loading model: {args.model_name}...")
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()
    
    # 4. Load & Preprocess original frames for visualization overlays
    raw_frames = load_video_frames(args.video_path, num_frames=args.num_frames, stride=args.stride)
    cropped_frames = resize_and_center_crop(raw_frames, target_size=(256, 256))
    
    # 5. Preprocess frames for model inputs
    # processor expects T x C x H x W
    video_tensor = torch.from_numpy(cropped_frames).permute(0, 3, 1, 2)  # [T, C, H, W]
    inputs = processor(video_tensor, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 6. Model forward pass, capturing all hidden states
    print("Running model inference to extract intermediate representations...")
    try:
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "mps" in str(e).lower():
            print(f"\n[Warning] GPU/MPS inference failed due to memory limit: {e}")
            print("Falling back to CPU device for inference...")
            model = model.to("cpu")
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
        else:
            raise e
        
    hidden_states_all = [h.cpu() for h in outputs.hidden_states]  # List of 25 layers: [0] = patch embed, [1-24] = layer outputs
    print(f"Retrieved {len(hidden_states_all)} representation layers.")
    
    # 7. Compute ground truth target targets for probes
    # A. Structure: row and column coordinates
    T_p = args.num_frames // model.config.tubelet_size
    H_p = 256 // model.config.patch_size
    W_p = 256 // model.config.patch_size
    n_tokens = T_p * H_p * W_p
    
    row_targets = torch.zeros(n_tokens, dtype=torch.long)
    col_targets = torch.zeros(n_tokens, dtype=torch.long)
    for i in range(n_tokens):
        # spatial patch coordinates
        y = (i % (H_p * W_p)) // W_p
        x = i % W_p
        row_targets[i] = y
        col_targets[i] = x
        
    # B. Motion: binned motion magnitude
    motion_tokens = compute_motion_magnitude(video_tensor, tubelet_size=model.config.tubelet_size, patch_size=model.config.patch_size)
    motion_flat = motion_tokens.flatten().to(device)
    
    # Bin motion into 5 categories using quantiles
    motion_np = motion_flat.cpu().numpy()
    quantiles = np.percentile(motion_np, [20, 40, 60, 80])
    motion_targets = torch.zeros_like(motion_flat, dtype=torch.long)
    for i, q in enumerate(quantiles):
        motion_targets[motion_flat > q] = i + 1
        
    # Split train/val indices (80% train, 20% validation)
    indices = np.arange(n_tokens)
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(0.8 * n_tokens)
    train_idx = indices[:split]
    val_idx = indices[split:]
    
    # 8. Loop over layers to train probes & compute RSA leakage
    probing_layers = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    
    structure_accs = []
    motion_accs = []
    leakage_scores = []
    fg_coherences = []
    bg_coherences = []
    
    print("\n--- Training Linear Probes and Computing RSA ---")
    for layer in probing_layers:
        # hidden_states_all[layer] shape: [1, N_tokens, 1024]
        layer_feats = hidden_states_all[layer].squeeze(0).cpu()
        
        # Train & Eval Probes
        struct_acc, mot_acc = train_and_eval_probes(
            layer_feats, row_targets, col_targets, motion_targets, 
            train_idx, val_idx, epochs=50, lr=0.01, device="cpu"
        )
        structure_accs.append(struct_acc)
        motion_accs.append(mot_acc)
        
        # Compute RSA Leakage
        leakage, fg_coh, bg_coh = compute_rsa_leakage(layer_feats, motion_flat.cpu())
        leakage_scores.append(leakage)
        fg_coherences.append(fg_coh)
        bg_coherences.append(bg_coh)
        
        print(f"Layer {layer:02d}: Structure Probe Acc={struct_acc:.4f} | Motion Probe Acc={mot_acc:.4f} | Foreground-Background Leakage={leakage:.4f}")
        
    # Save Probing Line Chart
    plt.figure(figsize=(8, 5))
    plt.plot(probing_layers, structure_accs, 'o-', color='#1f77b4', linewidth=2, label="Structure (Spatial Location)")
    plt.plot(probing_layers, motion_accs, 's-', color='#ff7f0e', linewidth=2, label="Motion Magnitude")
    plt.axhline(y=1/16, color='#1f77b4', linestyle='--', alpha=0.5, label="Structure Chance Line")
    plt.axhline(y=1/5, color='#ff7f0e', linestyle='--', alpha=0.5, label="Motion Chance Line")
    plt.xlabel("Layer Index (0=Embeddings, 24=Final Output)", fontsize=11)
    plt.ylabel("Validation Classification Accuracy", fontsize=11)
    plt.title("Probing Layer Abstraction: Structure vs. Motion", fontsize=12, fontweight='bold', pad=12)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    metrics_plot_path = os.path.join(args.output_dir, "layer_probing_metrics.png")
    plt.savefig(metrics_plot_path, dpi=150)
    plt.close()
    print(f"Saved probing line plot to: {metrics_plot_path}")
    
    # Save RSA Leakage Line Chart
    plt.figure(figsize=(8, 5))
    plt.plot(probing_layers, leakage_scores, 'o-', color='#2ca02c', linewidth=2, label="Cross-Similarity (Foreground-Background)")
    plt.plot(probing_layers, fg_coherences, '^-', color='#d62728', linewidth=1.5, alpha=0.7, label="Foreground Coherence")
    plt.plot(probing_layers, bg_coherences, 'v-', color='#9467bd', linewidth=1.5, alpha=0.7, label="Background Coherence")
    plt.xlabel("Layer Index (0=Embeddings, 24=Final Output)", fontsize=11)
    plt.ylabel("Average Cosine Similarity", fontsize=11)
    plt.title("Representational Similarity Analysis (RSA): Context Leakage", fontsize=12, fontweight='bold', pad=12)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    rsa_plot_path = os.path.join(args.output_dir, "context_leakage_rsa.png")
    plt.savefig(rsa_plot_path, dpi=150)
    plt.close()
    print(f"Saved RSA leakage plot to: {rsa_plot_path}")

    # 9. Perform layer-wise PCA and overlays for representative layers
    pca_layers = [2, 8, 14, 20, 24]
    overlays = {}
    
    print("\n--- Computing Layer-wise PCA and Rendering Overlays ---")
    for layer in pca_layers:
        layer_feats = hidden_states_all[layer].squeeze(0)  # [N_tokens, 1024]
        
        # Global PCA SVD over tokens
        projected_pca = compute_layer_pca(layer_feats, T_p, H_p, W_p)  # [T_p, H_p, W_p, 3]
        
        # Interpolate and generate overlay
        overlay_frames, raw_pca_numpy = interpolate_and_overlay(projected_pca, cropped_frames, alpha=args.alpha)
        overlays[layer] = overlay_frames
        
        # Save individual layer overlay video
        lay_video_path = os.path.join(args.output_dir, f"layer_{layer}_pca.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(lay_video_path, fourcc, 15.0, (256, 256))
        for t in range(args.num_frames):
            frame_bgr = cv2.cvtColor(overlay_frames[t], cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()
        print(f"Saved Layer {layer} PCA overlay video to: {lay_video_path}")
        
    # 10. Generate compilation outputs
    grid_video_path = os.path.join(args.output_dir, "layer_pca_comparison.mp4")
    generate_grid_comparison_video(cropped_frames, overlays, grid_video_path)
    
    keyframe_grid_path = os.path.join(args.output_dir, "keyframe_layers_grid.png")
    save_keyframe_layers_grid(cropped_frames, overlays, keyframe_grid_path)
    
    print("\nOption 3 pipeline execution complete! All visualizations generated successfully.")


if __name__ == "__main__":
    main()
