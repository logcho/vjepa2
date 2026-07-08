#!/usr/bin/env python3
import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Add workspace search paths
sys.path.append("/Users/loganchoi/Desktop/vjepa2/vjepa2")
sys.path.append("/Users/loganchoi/Desktop/dinov3/dinov3")

from src.models.ac_predictor import VisionTransformerPredictorAC

def main():
    print("=== Training Action-Conditioned V-JEPA Predictor ===")
    
    # 1. Device Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # 2. Load Dataset
    data_dir = "/Users/loganchoi/Desktop/vjepa2/dinov3_exploration/pathway_1/planning_rl"
    npz_path = os.path.join(data_dir, "synthetic_data.npz")
    if not os.path.exists(npz_path):
        print(f"Dataset not found at {npz_path}. Run generate_data.py first.")
        sys.exit(1)
        
    data = np.load(npz_path)
    obs = data["observations"]  # [120, 8, 224, 224, 3]
    states = data["states"]      # [120, 8, 7]
    actions = data["actions"]    # [120, 7, 7]
    
    num_trajs, T, H, W, C = obs.shape
    print(f"Loaded {num_trajs} trajectories of length {T} with frame size {H}x{W}")

    # 3. Load DINOv3 Backbone
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

    # 4. Extract DINOv3 Features
    print("Extracting DINOv3 features for all frames...")
    extracted_features = []
    
    # Process in batches of trajectories to avoid memory issues
    batch_size = 10
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    
    with torch.no_grad():
        for i in range(0, num_trajs, batch_size):
            traj_batch = obs[i:i+batch_size]  # [batch, T, H, W, 3]
            B_curr = len(traj_batch)
            
            # Reshape to [B_curr * T, H, W, 3] and preprocess
            flat_frames = traj_batch.reshape(-1, H, W, 3)
            tensor = torch.from_numpy(flat_frames).permute(0, 3, 1, 2).float().to(device) / 255.0
            normalized = (tensor - mean) / std
            
            # Extract features
            outputs = dinov3_model.forward_features(normalized)
            patch_tokens = outputs["x_norm_patchtokens"].cpu()  # [B_curr * T, N_patches, Embed_Dim]
            
            # Reshape back to [B_curr, T, N_patches, Embed_Dim]
            N_patches = patch_tokens.shape[1]
            D = patch_tokens.shape[2]
            features_batch = patch_tokens.view(B_curr, T, N_patches, D)
            extracted_features.append(features_batch)
            
    extracted_features = torch.cat(extracted_features, dim=0)
    print(f"Extracted features shape: {extracted_features.shape}") # [120, 8, 196, 384]

    # 5. Split Train and Validation
    train_split = 100
    
    train_features = extracted_features[:train_split]
    train_states = torch.tensor(states[:train_split], dtype=torch.float32)
    train_actions = torch.tensor(actions[:train_split], dtype=torch.float32)
    
    val_features = extracted_features[train_split:]
    val_states = torch.tensor(states[train_split:], dtype=torch.float32)
    val_actions = torch.tensor(actions[train_split:], dtype=torch.float32)

    # 6. Instantiate Action-Conditioned Predictor
    predictor = VisionTransformerPredictorAC(
        img_size=(H, W),
        patch_size=16,
        num_frames=T - 1,   # 7 frame transitions
        tubelet_size=1,
        embed_dim=384,
        predictor_embed_dim=192,
        depth=4,
        num_heads=6,
        action_embed_dim=7
    ).to(device)

    # 7. Optimizer & Criterion
    LR = 1e-3
    WEIGHT_DECAY = 0.01
    EPOCHS = 50
    TRAIN_BATCH_SIZE = 8
    
    optimizer = optim.AdamW(predictor.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    
    # 8. Training Loop
    loss_history = []
    val_loss_history = []
    
    print("\nStarting training loop...")
    for epoch in range(1, EPOCHS + 1):
        predictor.train()
        epoch_loss = 0.0
        
        # Shuffle train data
        indices = torch.randperm(train_split)
        shuffled_feats = train_features[indices]
        shuffled_states = train_states[indices]
        shuffled_actions = train_actions[indices]
        
        num_batches = math.ceil(train_split / TRAIN_BATCH_SIZE)
        for b in range(num_batches):
            idx_start = b * TRAIN_BATCH_SIZE
            idx_end = min((b + 1) * TRAIN_BATCH_SIZE, train_split)
            B_curr = idx_end - idx_start
            
            # Load batch to device
            batch_feats = shuffled_feats[idx_start:idx_end].to(device)      # [B, 8, 196, 384]
            batch_states = shuffled_states[idx_start:idx_end].to(device)    # [B, 8, 7]
            batch_actions = shuffled_actions[idx_start:idx_end].to(device)  # [B, 7, 7]
            
            # Context input is frames 0..6: shape [B, 7 * 196, 384]
            x_context = batch_feats[:, :7].flatten(1, 2)
            
            # Actions and states are steps 0..6
            actions_in = batch_actions[:, :7]
            states_in = batch_states[:, :7]
            
            # Target is frames 1..7: shape [B, 7 * 196, 384]
            y_target = batch_feats[:, 1:8].flatten(1, 2)
            
            # Forward pass
            optimizer.zero_grad()
            y_pred = predictor(x_context, actions_in, states_in)
            
            loss = criterion(y_pred, y_target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * B_curr
            
        avg_loss = epoch_loss / train_split
        loss_history.append(avg_loss)
        
        # Validation evaluation
        predictor.eval()
        with torch.no_grad():
            val_feats_dev = val_features.to(device)
            val_states_dev = val_states.to(device)
            val_actions_dev = val_actions.to(device)
            
            x_val_context = val_feats_dev[:, :7].flatten(1, 2)
            y_val_target = val_feats_dev[:, 1:8].flatten(1, 2)
            
            y_val_pred = predictor(x_val_context, val_actions_dev[:, :7], val_states_dev[:, :7])
            val_loss = criterion(y_val_pred, y_val_target).item()
            val_loss_history.append(val_loss)
            
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {avg_loss:.5f} | Val Loss: {val_loss:.5f}")
            
    print("\nTraining completed!")
    
    # 9. Save Weights
    model_save_path = os.path.join(data_dir, "predictor_ac_pathway1.pth")
    torch.save(predictor.state_dict(), model_save_path)
    print(f"Saved predictor weights to: {model_save_path}")
    
    # 10. Generate Training Plots
    plt.figure(figsize=(7, 4.5))
    plt.plot(loss_history, color="#4F46E5", linewidth=2.5, label="Train Loss")
    plt.plot(val_loss_history, color="#10B981", linewidth=2.0, linestyle="--", label="Val Loss")
    plt.xlabel("Epochs", fontsize=11, fontweight="bold")
    plt.ylabel("MSE Loss", fontsize=11, fontweight="bold")
    plt.title("Action-Conditioned Predictor Loss Curve", fontsize=12, fontweight="bold", pad=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(data_dir, "visualizations", "loss_curve.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved training plot to: {plot_path}")

if __name__ == "__main__":
    main()
