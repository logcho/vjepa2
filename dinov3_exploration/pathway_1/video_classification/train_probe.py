#!/usr/bin/env python3
import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Add search paths
sys.path.append("/Users/loganchoi/Desktop/vjepa2/vjepa2")
sys.path.append("/Users/loganchoi/Desktop/dinov3/dinov3")

from src.models.predictor import VisionTransformerPredictor
from src.masks.multiseq_multiblock3d import _MaskGenerator
from src.masks.utils import apply_masks

# Linear Probe Model
class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes=8):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)

# MLP Probe Model
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

def train_eval_model(model, train_x, train_y, val_x, val_y, epochs=100, lr=1e-3, weight_decay=1e-4):
    device = train_x.device
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    train_acc_history = []
    val_acc_history = []
    
    batch_size = 16
    num_train = len(train_x)
    
    for epoch in range(1, epochs + 1):
        model.train()
        # Shuffle train data
        indices = torch.randperm(num_train)
        shuffled_x = train_x[indices]
        shuffled_y = train_y[indices]
        
        num_batches = math.ceil(num_train / batch_size)
        epoch_loss = 0.0
        correct = 0
        
        for b in range(num_batches):
            bx = shuffled_x[b*batch_size : (b+1)*batch_size]
            by = shuffled_y[b*batch_size : (b+1)*batch_size]
            
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(bx)
            preds = logits.argmax(dim=-1)
            correct += (preds == by).sum().item()
            
        train_acc = correct / num_train
        train_acc_history.append(train_acc)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_preds = val_logits.argmax(dim=-1)
            val_acc = (val_preds == val_y).sum().item() / len(val_y)
            val_acc_history.append(val_acc)
            
    return train_acc_history, val_acc_history

def main():
    print("=== Training Action Recognition Probes ===")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    data_dir = "/Users/loganchoi/Desktop/vjepa2/dinov3_exploration/pathway_1/video_classification"
    npz_path = os.path.join(data_dir, "synthetic_data.npz")
    if not os.path.exists(npz_path):
        print(f"Dataset not found at {npz_path}. Run generate_data.py first.")
        sys.exit(1)
        
    # Load synthetic dataset
    data = np.load(npz_path)
    features_np = data["features"]  # [200, 8, 784, 384]
    labels_np = data["labels"]      # [200]
    classes_desc = data["classes"]  # 8 class descriptions
    
    num_videos, T, N_patches, D = features_np.shape
    print(f"Loaded {num_videos} videos. Length={T}, Patches={N_patches}, Dim={D}")
    
    # 2. Split dataset (first 20 per class = train, last 5 = val)
    train_idx = []
    val_idx = []
    samples_per_class = 25
    for c in range(8):
        start = c * samples_per_class
        train_idx.extend(range(start, start + 20))
        val_idx.extend(range(start + 20, start + 25))
        
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    
    features = torch.from_numpy(features_np).float()
    labels = torch.from_numpy(labels_np).long()
    
    train_feats = features[train_idx]
    train_labels = labels[train_idx]
    val_feats = features[val_idx]
    val_labels = labels[val_idx]
    
    print(f"Train split size: {len(train_feats)}")
    print(f"Val split size: {len(val_feats)}")
    
    # 3. Baseline A: DINOv3-Mean (Spatially & Temporally Pooled)
    # Average across all patches and all frames -> [B, D]
    train_dinov3_mean = train_feats.mean(dim=(1, 2)).to(device)
    val_dinov3_mean = val_feats.mean(dim=(1, 2)).to(device)
    
    print("\nTraining Baseline A (DINOv3-Mean Linear Probe)...")
    baseline_a_model = LinearProbe(input_dim=D)
    ta_a, va_a = train_eval_model(baseline_a_model, train_dinov3_mean, train_labels.to(device), val_dinov3_mean, val_labels.to(device), epochs=100)
    print(f"Baseline A Final Accuracy - Train: {ta_a[-1]:.4f} | Val: {va_a[-1]:.4f}")
    
    # 4. Baseline B: DINOv3-Concat (Spatially Pooled, Temporally Flattened)
    # Average across patches, flatten temporal dimension -> [B, T * D]
    train_dinov3_concat = train_feats.mean(dim=2).flatten(1, 2).to(device)
    val_dinov3_concat = val_feats.mean(dim=2).flatten(1, 2).to(device)
    
    print("\nTraining Baseline B (DINOv3-Concat Linear Probe)...")
    baseline_b_model = LinearProbe(input_dim=T * D)
    ta_b, va_b = train_eval_model(baseline_b_model, train_dinov3_concat, train_labels.to(device), val_dinov3_concat, val_labels.to(device), epochs=100)
    print(f"Baseline B Final Accuracy - Train: {ta_b[-1]:.4f} | Val: {va_b[-1]:.4f}")
    
    # 5. V-JEPA Predictor Feature Extraction
    print("\nLoading pretrained V-JEPA Predictor...")
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
    
    model_path = "/Users/loganchoi/Desktop/vjepa2/dinov3_exploration/pathway_1/predictor_pathway1.pth"
    predictor.load_state_dict(torch.load(model_path, map_location=device))
    predictor.eval()
    
    # Instantiate deterministic/reproducible mask generator for validation
    mask_gen = _MaskGenerator(
        crop_size=(448, 448),
        num_frames=T,
        spatial_patch_size=(16, 16),
        temporal_patch_size=1,
        spatial_pred_mask_scale=(0.4, 0.5),
        temporal_pred_mask_scale=(0.6, 0.8),
        aspect_ratio=(0.8, 1.25),
        npred=3,
        full_complement=True
    )
    
    print("Extracting V-JEPA predictor spatiotemporal tokens...")
    # Process all samples to get predictor predictions
    # We will run them in a batch to get spatiotemporal representations
    def extract_predictor_reprs(feats_tensor):
        B = len(feats_tensor)
        pooled_reprs = []
        batch_size = 4
        
        # Seed for reproducibility of masks across runs
        torch.manual_seed(42)
        
        for idx in range(0, B, batch_size):
            batch_feats = feats_tensor[idx:idx+batch_size]
            B_curr = len(batch_feats)
            feats_flat = batch_feats.view(B_curr, T * N_patches, D).to(device)
            
            masks_enc, masks_pred = mask_gen(B_curr)
            masks_enc = masks_enc.to(device)
            masks_pred = masks_pred.to(device)
            
            with torch.no_grad():
                x_context = apply_masks(feats_flat, [masks_enc])
                y_pred = predictor(x_context, masks_enc, masks_pred) # [B_curr, N_pred, D]
                pooled_repr = y_pred.mean(dim=1) # [B_curr, D]
                pooled_reprs.append(pooled_repr.cpu())
                
        return torch.cat(pooled_reprs, dim=0)
        
    train_vjepa_reprs = extract_predictor_reprs(train_feats).to(device)
    val_vjepa_reprs = extract_predictor_reprs(val_feats).to(device)
    
    # 5b. V-JEPA Predictor Causal Future Feature Extraction
    # Context = frames 0..3, Target = frames 4..7
    def extract_predictor_reprs_causal(feats_tensor):
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
                y_pred = predictor(x_context, masks_enc, masks_pred) # [B_curr, N_pred, D]
                pooled_repr = y_pred.mean(dim=1) # [B_curr, D]
                pooled_reprs.append(pooled_repr.cpu())
                
        return torch.cat(pooled_reprs, dim=0)
        
    train_vjepa_reprs_causal = extract_predictor_reprs_causal(train_feats).to(device)
    val_vjepa_reprs_causal = extract_predictor_reprs_causal(val_feats).to(device)
    
    # 6. Train V-JEPA Linear Probe (Random Mask)
    print("\nTraining V-JEPA Predictor (Random Mask, Linear Probe)...")
    vjepa_linear_model = LinearProbe(input_dim=D)
    ta_vj_lin, va_vj_lin = train_eval_model(vjepa_linear_model, train_vjepa_reprs, train_labels.to(device), val_vjepa_reprs, val_labels.to(device), epochs=100)
    print(f"V-JEPA (Random Mask) Linear Probe Final Accuracy - Train: {ta_vj_lin[-1]:.4f} | Val: {va_vj_lin[-1]:.4f}")
    
    # 7. Train V-JEPA MLP Probe (Random Mask)
    print("\nTraining V-JEPA Predictor (Random Mask, MLP Probe)...")
    vjepa_mlp_model = MLPProbe(input_dim=D)
    ta_vj_mlp, va_vj_mlp = train_eval_model(vjepa_mlp_model, train_vjepa_reprs, train_labels.to(device), val_vjepa_reprs, val_labels.to(device), epochs=100)
    print(f"V-JEPA (Random Mask) MLP Probe Final Accuracy - Train: {ta_vj_mlp[-1]:.4f} | Val: {va_vj_mlp[-1]:.4f}")

    # 7b. Train V-JEPA Linear Probe (Causal Future)
    print("\nTraining V-JEPA Predictor (Causal Future, Linear Probe)...")
    vjepa_linear_model_causal = LinearProbe(input_dim=D)
    ta_vj_lin_c, va_vj_lin_c = train_eval_model(vjepa_linear_model_causal, train_vjepa_reprs_causal, train_labels.to(device), val_vjepa_reprs_causal, val_labels.to(device), epochs=100)
    print(f"V-JEPA (Causal Future) Linear Probe Final Accuracy - Train: {ta_vj_lin_c[-1]:.4f} | Val: {va_vj_lin_c[-1]:.4f}")
    
    # 7c. Train V-JEPA MLP Probe (Causal Future)
    print("\nTraining V-JEPA Predictor (Causal Future, MLP Probe)...")
    vjepa_mlp_model_causal = MLPProbe(input_dim=D)
    ta_vj_mlp_c, va_vj_mlp_c = train_eval_model(vjepa_mlp_model_causal, train_vjepa_reprs_causal, train_labels.to(device), val_vjepa_reprs_causal, val_labels.to(device), epochs=100)
    print(f"V-JEPA (Causal Future) MLP Probe Final Accuracy - Train: {ta_vj_mlp_c[-1]:.4f} | Val: {va_vj_mlp_c[-1]:.4f}")
    
    # 8. Plot Accuracy Curves
    plt.figure(figsize=(10, 6))
    plt.plot(va_a, color="#EF4444", linewidth=2, linestyle=":", label="Baseline A: DINOv3-Mean (Val)")
    plt.plot(va_b, color="#F59E0B", linewidth=2, linestyle="--", label="Baseline B: DINOv3-Concat (Val)")
    plt.plot(va_vj_lin, color="#10B981", linewidth=1.5, linestyle=":", label="V-JEPA (Random Mask, Linear Probe) (Val)")
    plt.plot(va_vj_mlp, color="#10B981", linewidth=2.0, linestyle="--", label="V-JEPA (Random Mask, MLP Probe) (Val)")
    plt.plot(va_vj_lin_c, color="#3B82F6", linewidth=2.0, linestyle="-.", label="V-JEPA (Causal Future, Linear Probe) (Val)")
    plt.plot(va_vj_mlp_c, color="#4F46E5", linewidth=2.5, label="V-JEPA (Causal Future, MLP Probe) (Val)")
    
    plt.xlabel("Training Epochs", fontsize=11, fontweight="bold")
    plt.ylabel("Validation Accuracy", fontsize=11, fontweight="bold")
    plt.title("Action Recognition Downstream Probe Performance Comparison", fontsize=12, fontweight="bold", pad=12)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    plot_path = os.path.join(data_dir, "visualizations", "probe_performance.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved performance plots to: {plot_path}")
    
    # Generate per-class evaluation stats for V-JEPA MLP Causal Probe
    vjepa_mlp_model_causal.eval()
    with torch.no_grad():
        val_logits = vjepa_mlp_model_causal(val_vjepa_reprs_causal)
        val_preds = val_logits.argmax(dim=-1).cpu().numpy()
        val_true = val_labels.numpy()
        
    print("\nClassification Report (V-JEPA Causal Future MLP Probe):")
    print(f"{'Class Description':<36} | {'Accuracy':<10}")
    print("-" * 50)
    for c in range(8):
        c_indices = np.where(val_true == c)[0]
        c_acc = np.mean(val_preds[c_indices] == val_true[c_indices])
        print(f"{classes_desc[c]:<36} | {c_acc * 100:.1f}%")
        
if __name__ == "__main__":
    main()
