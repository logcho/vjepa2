#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Add workspace search paths
sys.path.append("/Users/loganchoi/Desktop/vjepa2/vjepa2")
sys.path.append("/Users/loganchoi/Desktop/dinov3/dinov3")

from src.models.predictor import VisionTransformerPredictor
from src.masks.utils import apply_masks

# Redefine the probe models to load weights or train on the fly
class MLPProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        return self.mlp(x)

def generate_video_frames_for_sample(coords, frame_size=448, circle_radius=30):
    """
    Renders the 8 frames of a trajectory.
    """
    frames = []
    for cx, cy in coords:
        frame = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)
        # White circle
        cv2.circle(frame, (int(cx), int(cy)), circle_radius, (255, 255, 255), -1)
        # Add a subtle grid/coordinate indicator so direction is obvious visually
        cv2.circle(frame, (int(cx), int(cy)), 3, (255, 0, 0), -1) # Blue center dot
        frames.append(frame)
    return np.stack(frames)

def generate_sample_trajectory_coords(label, T=8, frame_size=448, circle_radius=30):
    """
    Generates a single trajectory coordinates for a specific class label.
    """
    coords = []
    # Use deterministic offsets to make the visualization clean and archetypal
    if label == 0:  # Horizontal Left-to-Right
        x, y = 60, 224
        dx, dy = 46, 0
        for t in range(T): coords.append((x + t*dx, y + t*dy))
    elif label == 1:  # Horizontal Right-to-Left
        x, y = 388, 224
        dx, dy = -46, 0
        for t in range(T): coords.append((x + t*dx, y + t*dy))
    elif label == 2:  # Vertical Bottom-to-Top
        x, y = 224, 388
        dx, dy = 0, -46
        for t in range(T): coords.append((x + t*dx, y + t*dy))
    elif label == 3:  # Vertical Top-to-Bottom
        x, y = 224, 60
        dx, dy = 0, 46
        for t in range(T): coords.append((x + t*dx, y + t*dy))
    elif label == 4 or label == 5:  # Circles (4: CW, 5: CCW)
        cx, cy = 224, 224
        r = 120
        start_angle = 0
        step = -2*np.pi / 8 if label == 4 else 2*np.pi / 8
        for t in range(T):
            angle = start_angle + t * step
            coords.append((cx + r*np.cos(angle), cy + r*np.sin(angle)))
    elif label == 6:  # Diagonal Top-Left to Bottom-Right
        x, y = 60, 60
        dx, dy = 46, 46
        for t in range(T): coords.append((x + t*dx, y + t*dy))
    elif label == 7:  # Diagonal Bottom-Left to Top-Right
        x, y = 60, 388
        dx, dy = 46, -46
        for t in range(T): coords.append((x + t*dx, y + t*dy))
        
    coords = [(np.clip(x, circle_radius, frame_size - circle_radius),
               np.clip(y, circle_radius, frame_size - circle_radius))
              for x, y in coords]
    return coords

def main():
    print("=== Visualizing Video Classification Predictions ===")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    data_dir = "/Users/loganchoi/Desktop/vjepa2/dinov3_exploration/pathway_1/video_classification"
    npz_path = os.path.join(data_dir, "synthetic_data.npz")
    if not os.path.exists(npz_path):
        print(f"Dataset not found at {npz_path}. Run generate_data.py first.")
        sys.exit(1)
        
    # Load dataset
    data = np.load(npz_path)
    features_np = data["features"]  # [200, 8, 784, 384]
    labels_np = data["labels"]      # [200]
    classes_desc = data["classes"]  # 8 class descriptions
    
    num_videos, T, N_patches, D = features_np.shape
    
    # Train the MLP classifier on-the-fly in 1 second to make this visualization script standalone
    print("Training MLP classifier on pre-extracted features...")
    train_idx, val_idx = [], []
    for c in range(8):
        start = c * 25
        train_idx.extend(range(start, start + 20))
        val_idx.extend(range(start + 20, start + 25))
        
    features = torch.from_numpy(features_np).float()
    labels = torch.from_numpy(labels_np).long()
    
    # Load Predictor
    predictor = VisionTransformerPredictor(
        img_size=(448, 448),
        patch_size=16,
        num_frames=T,
        tubelet_size=1,
        embed_dim=384,
        predictor_embed_dim=192,
        out_embed_dim=384,
        depth=4,
        num_heads=6,
        use_mask_tokens=True,
        num_mask_tokens=1
    ).to(device)
    predictor.load_state_dict(torch.load("/Users/loganchoi/Desktop/vjepa2/dinov3_exploration/pathway_1/predictor_pathway1.pth", map_location=device))
    predictor.eval()
    
    # Causal feature extractor
    def extract_causal_reprs(feats_tensor):
        B = len(feats_tensor)
        pooled_reprs = []
        batch_size = 4
        for idx in range(0, B, batch_size):
            batch_feats = feats_tensor[idx:idx+batch_size]
            B_curr = len(batch_feats)
            feats_flat = batch_feats.view(B_curr, T * N_patches, D).to(device)
            masks_enc = torch.arange(0, 4 * N_patches, device=device).unsqueeze(0).expand(B_curr, -1)
            masks_pred = torch.arange(4 * N_patches, T * N_patches, device=device).unsqueeze(0).expand(B_curr, -1)
            with torch.no_grad():
                x_context = apply_masks(feats_flat, [masks_enc])
                y_pred = predictor(x_context, masks_enc, masks_pred)
                pooled = y_pred.mean(dim=1)
                pooled_reprs.append(pooled.cpu())
        return torch.cat(pooled_reprs, dim=0)
        
    train_reprs = extract_causal_reprs(features[train_idx]).to(device)
    val_reprs = extract_causal_reprs(features[val_idx]).to(device)
    
    # Train the MLP probe
    mlp_model = MLPProbe(input_dim=D).to(device)
    optimizer = optim.AdamW(mlp_model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    train_labels_dev = labels[train_idx].to(device)
    for epoch in range(120):
        mlp_model.train()
        optimizer.zero_grad()
        logits = mlp_model(train_reprs)
        loss = criterion(logits, train_labels_dev)
        loss.backward()
        optimizer.step()
        
    mlp_model.eval()
    
    # Now generate the large figure
    print("Generating spatiotemporal visualization figure...")
    fig = plt.figure(figsize=(19, 15))
    
    # Pre-extract DINOv3 model for rendering fresh samples if needed, but since we already have the coordinates,
    # we can generate them deterministically and run the pre-extracted features for predictions!
    # For each class, let's take a validation sample from the dataset to make it authentic
    for c in range(8):
        # Find a val sample index for class c (e.g. index 0 of the val samples for this class)
        val_sample_idx = val_idx[c * 5] # first val sample of class c
        
        sample_feats = features[val_sample_idx].unsqueeze(0) # [1, T, N, D]
        sample_label = labels[val_sample_idx].item()
        
        # Get predictor representations and classify
        sample_repr = extract_causal_reprs(sample_feats).to(device)
        with torch.no_grad():
            logits = mlp_model(sample_repr)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred_class = probs.argmax()
            pred_conf = probs[pred_class]
            
        # Draw frame sequence
        # We can construct coordinates for this validation sample from the saved state array in the dataset!
        coords = data["states"][val_sample_idx] # [T, 2]
        # Denormalize coordinates back to pixel space
        coords_pixels = [(int((cx + 1) * 224), int((cy + 1) * 224)) for cx, cy in coords]
        
        frames = generate_video_frames_for_sample(coords_pixels)
        
        # Plot the 8 frames in the row
        for t in range(T):
            ax = plt.subplot2grid((8, 12), (c, t))
            ax.imshow(frames[t])
            ax.axis("off")
            # Highlight context vs target (first 4 frames are context, last 4 are predicted target)
            if t < 4:
                ax.set_title(f"t={t}\n[Ctxt]", color="#10B981", fontsize=8, pad=2)
                # Draw a green border around the subplot
                for spine in ax.spines.values():
                    spine.set_color("#10B981")
                    spine.set_linewidth(1.5)
            else:
                ax.set_title(f"t={t}\n[Pred]", color="#3B82F6", fontsize=8, pad=2)
                for spine in ax.spines.values():
                    spine.set_color("#3B82F6")
                    spine.set_linewidth(1.5)
                    
            if t == 0:
                # Add class label text on the far left
                ax.text(-80, 224, f"Class {c}\n{classes_desc[c]}", 
                        fontsize=9, fontweight="bold", verticalalignment='center', horizontalalignment='right')
                
        # Plot the probability bar chart in the remaining columns
        ax_bar = plt.subplot2grid((8, 12), (c, 8), colspan=4)
        colors = ["#EF4444"] * 8
        colors[sample_label] = "#10B981" # Green for ground truth
        if pred_class != sample_label:
            colors[pred_class] = "#F59E0B" # Orange for wrong prediction
        else:
            colors[pred_class] = "#4F46E5" # Indigo/Blue-purple for correct prediction
            
        bars = ax_bar.barh(np.arange(8), probs, color=colors, height=0.6, alpha=0.85)
        ax_bar.set_yticks(np.arange(8))
        ax_bar.set_yticklabels([f"C{i}" for i in range(8)], fontsize=8)
        ax_bar.set_xlim(0, 1.05)
        ax_bar.grid(True, axis='x', linestyle="--", alpha=0.5)
        
        # Add labels to the bars
        for bar in bars:
            width = bar.get_width()
            if width > 0.05:
                ax_bar.text(width + 0.01, bar.get_y() + bar.get_height()/2, f"{width*100:.1f}%", 
                            fontsize=7, fontweight='bold', va='center')
                
        is_correct = "CORRECT" if pred_class == sample_label else "INCORRECT"
        ax_bar.set_title(f"True: {classes_desc[sample_label]} | Pred: C{pred_class} ({is_correct}, {pred_conf*100:.1f}%)", 
                         fontsize=8.5, fontweight="bold", color="#1E293B")
        
    plt.suptitle("Spatiotemporal Motion Sequences & V-JEPA Classification Probabilities", 
                 fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()
    
    plot_path = os.path.join(data_dir, "visualizations", "classification_visuals.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved spatiotemporal classification visuals to: {plot_path}")
    
    # Also copy it to the artifact folder so the user can inspect it
    dest_path = "/Users/loganchoi/.gemini/antigravity-ide/brain/6fc1cc9f-7e2c-4efd-a855-cb25c10b268a/visualizations/classification_visuals.png"
    import shutil
    shutil.copy(plot_path, dest_path)
    print(f"Copied visualization to artifact path: {dest_path}")

if __name__ == "__main__":
    main()
