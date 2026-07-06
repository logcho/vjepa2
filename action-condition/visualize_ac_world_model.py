#!/usr/bin/env python3
"""
Option 4: Action-Conditioned World Modeling for V-JEPA 2.
This script:
1. Loads the pre-trained V-JEPA 2 Action-Conditioned (ViT-Giant) model from Torch Hub.
2. Loads a sample Franka robot trajectory containing video observations and states.
3. Extracts latent representations using the V-JEPA 2 encoder.
4. Performs forward transitions using the Action-Conditioned Predictor.
5. Computes and plots the Action Energy Landscape (2D prediction loss heatmap) over spatial control offsets.
6. Performs MPC action inference via the Cross-Entropy Method (CEM).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Resolve workspace paths and add vjepa2 modules to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, os.path.join(workspace_dir, "vjepa2"))
sys.path.insert(0, os.path.join(workspace_dir, "vjepa2", "notebooks"))

from app.vjepa_droid.transforms import make_transforms
from utils.mpc_utils import poses_to_diff, compute_new_pose
from utils.world_model_wrapper import WorldModel

def main():
    print("=== V-JEPA 2 Action-Conditioned World Modeling ===")

    # 1. Device Setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS device.")
    else:
        device = torch.device("cpu")
        print("Using CPU device.")

    # 2. Load Model
    print("\nLoading pre-trained vjepa2_ac_vit_giant model from local sources...")
    try:
        from src.hub.backbones import vjepa2_ac_vit_giant
        encoder, predictor = vjepa2_ac_vit_giant(pretrained=True)
        encoder = encoder.to(device).eval()
        predictor = predictor.to(device).eval()
        print("Successfully loaded model.")
    except Exception as e:
        print(f"Failed to load pre-trained model: {e}")
        print("Falling back to local initialization without pre-trained weights for demo purposes...")
        from src.hub.backbones import vjepa2_ac_vit_giant
        encoder, predictor = vjepa2_ac_vit_giant(pretrained=False)
        encoder = encoder.to(device).eval()
        predictor = predictor.to(device).eval()

    # 3. Load Trajectory Data
    traj_path = os.path.join(workspace_dir, "vjepa2", "notebooks", "franka_example_traj.npz")
    print(f"\nLoading trajectory from: {traj_path}")
    trajectory = np.load(traj_path)
    np_clips = trajectory["observations"]  # [1, T, 256, 256, 3]
    np_states = trajectory["states"]        # [1, T, 7]
    print(f"Clips shape: {np_clips.shape}")
    print(f"States shape: {np_states.shape}")

    # 4. Preprocess Frames and States
    crop_size = 256
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop_size,
    )
    
    # transform expects T x H x W x C and returns C x T x H x W
    clips = transform(np_clips[0]).unsqueeze(0).to(device)  # [1, 3, T, H, W]
    states = torch.tensor(np_states, device=device, dtype=torch.float32)

    # Compute ground truth action (delta pose from frame 0 to frame 1)
    np_actions = np.expand_dims(poses_to_diff(np_states[0, 0], np_states[0, 1]), axis=(0, 1))
    actions = torch.tensor(np_actions, device=device, dtype=torch.float32)
    
    gt_action = actions[0, 0].cpu().numpy()
    print(f"\nGround-Truth Action (translation & gripper delta):")
    print(f"  dx: {gt_action[0]:.4f}")
    print(f"  dy: {gt_action[1]:.4f}")
    print(f"  dz: {gt_action[2]:.4f}")
    print(f"  dgripper: {gt_action[6]:.4f}")

    # 5. Feature Extraction via target encoder
    tokens_per_frame = int((crop_size // encoder.patch_size) ** 2)
    
    def forward_target(c, normalize_reps=True):
        B, C, T, H, W = c.size()
        c = c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        h = encoder(c)
        h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
        if normalize_reps:
            h = F.layer_norm(h, (h.size(-1),))
        return h

    print("\nRunning feature extraction on video frames...")
    with torch.no_grad():
        h = forward_target(clips)  # [1, T * tokens_per_frame, D]
    print(f"Extracted representations shape: {h.shape} (Batch x Total Tokens x Embed Dim)")

    z_0 = h[:, :tokens_per_frame]      # Frame 0 tokens
    z_1_gt = h[:, tokens_per_frame:]  # Frame 1 tokens (target)

    # 6. Action-Conditioned Future Representation Prediction
    print("\nRunning future state prediction...")
    with torch.no_grad():
        # Predict representation of Frame 1 using Frame 0 tokens + true action + true state
        z_1_tf = predictor(z_0, actions, states[:, :1])[:, -tokens_per_frame:]
        z_1_tf = F.layer_norm(z_1_tf, (z_1_tf.size(-1),))

    # Evaluate representation forecasting accuracy
    prediction_loss = torch.mean(torch.abs(z_1_tf - z_1_gt)).item()
    print(f"Forecasting L1 representation error: {prediction_loss:.6f}")

    # 7. Compute Action Energy Landscape (2D Grid Heatmap over X-Z translation offsets)
    print("\nComputing Action Energy Landscape...")
    nsamples = 19
    grid_range = 0.15  # range of action offsets (meters)
    
    da_vals = np.linspace(-grid_range, grid_range, nsamples)
    dc_vals = np.linspace(-grid_range, grid_range, nsamples)
    
    action_candidates = []
    for db in dc_vals:  # z-translation (vertical axis in plot)
        for da in da_vals:  # x-translation (horizontal axis in plot)
            # Create action: [dx, dy, dz, rx, ry, rz, gripper]
            # Keeping dy (y-translation) and rotations zero
            action_candidates.append([da, 0.0, db, 0.0, 0.0, 0.0, 0.0])
            
    action_candidates = torch.tensor(action_candidates, device=device, dtype=torch.float32).unsqueeze(1)  # [S, 1, 7]
    S = len(action_candidates)

    # Repeat context state and frame tokens for parallel prediction
    z_0_repeated = z_0.repeat(S, 1, 1)
    states_0_repeated = states[:, :1].repeat(S, 1, 1)
    z_1_gt_repeated = z_1_gt.repeat(S, 1, 1)

    print(f"Evaluating predictor on {S} candidate actions...")
    with torch.no_grad():
        z_1_candidates = predictor(z_0_repeated, action_candidates, states_0_repeated)[:, -tokens_per_frame:]
        z_1_candidates = F.layer_norm(z_1_candidates, (z_1_candidates.size(-1),))
        
        # Calculate prediction errors (L1 distance in representation space)
        losses = torch.mean(torch.abs(z_1_candidates - z_1_gt_repeated), dim=[1, 2]).cpu().numpy()

    loss_grid = losses.reshape(nsamples, nsamples)

    # Create visualization folder if it doesn't exist
    os.makedirs(os.path.join(current_dir, "visualizations"), exist_ok=True)

    # Plot Heatmap of the Energy Landscape
    plt.figure(figsize=(8, 6.5))
    heatmap = plt.imshow(
        loss_grid, 
        extent=[-grid_range, grid_range, -grid_range, grid_range], 
        origin='lower', 
        cmap='viridis',
        aspect='auto'
    )
    plt.colorbar(heatmap, label="Representation Prediction L1 Loss")
    
    # Plot Ground Truth action
    plt.scatter(gt_action[0], gt_action[2], color='red', marker='*', s=150, edgecolors='white', label='Ground-Truth Action')
    
    plt.xlabel("Action Delta X (m)", fontsize=11)
    plt.ylabel("Action Delta Z (m)", fontsize=11)
    plt.title("Action-Conditioned V-JEPA Energy Landscape", fontsize=12, fontweight='bold', pad=12)
    plt.legend(loc='upper right')
    
    # Highlight the minimum energy area
    min_idx = np.argmin(losses)
    min_x, min_z = action_candidates[min_idx, 0, 0].item(), action_candidates[min_idx, 0, 2].item()
    plt.scatter(min_x, min_z, color='cyan', marker='o', s=80, edgecolors='black', label='Minimum Energy Action')
    plt.legend(loc='upper right')

    landscape_plot_path = os.path.join(current_dir, "visualizations", "energy_landscape.png")
    plt.savefig(landscape_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Energy Landscape heatmap to: {landscape_plot_path}")

    # Plot Prediction Error Comparison (Teacher Forcing vs Base Frame Loss)
    plt.figure(figsize=(7, 5))
    categories = ["No-Action Baseline (Frame 0)", "Action-Conditioned prediction"]
    baseline_loss = torch.mean(torch.abs(z_0 - z_1_gt)).item()
    errors = [baseline_loss, prediction_loss]
    
    bars = plt.bar(categories, errors, color=['#d62728', '#1f77b4'], width=0.5)
    plt.ylabel("L1 Distance in Representation Space", fontsize=11)
    plt.title("Action-Conditioned Representation Forecasting Accuracy", fontsize=12, fontweight='bold', pad=12)
    plt.grid(True, axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.5f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
                    
    error_plot_path = os.path.join(current_dir, "visualizations", "prediction_error.png")
    plt.savefig(error_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved representation error plot to: {error_plot_path}")

    # 8. CEM Planning Demonstration
    print("\n--- CEM Action Planning Demonstration ---")
    
    # Initialize MPC / WorldModel CEM wrapper
    world_model_cem = WorldModel(
        encoder=encoder,
        predictor=predictor,
        tokens_per_frame=tokens_per_frame,
        transform=transform,
        mpc_args={
            "rollout": 1,
            "samples": 500,
            "topk": 15,
            "cem_steps": 6,
            "momentum_mean": 0.15,
            "momentum_mean_gripper": 0.15,
            "momentum_std": 0.75,
            "momentum_std_gripper": 0.15,
            "maxnorm": 0.15,
            "verbose": False
        },
        normalize_reps=True,
        device=device
    )

    with torch.no_grad():
        planned_action = world_model_cem.infer_next_action(z_0, states[:, :1], z_1_gt).cpu().numpy()[0]
        
    print(f"Planned Action (x, y, z):")
    print(f"  dx: {planned_action[0]:.4f} (GT: {gt_action[0]:.4f})")
    print(f"  dy: {planned_action[1]:.4f} (GT: {gt_action[1]:.4f})")
    print(f"  dz: {planned_action[2]:.4f} (GT: {gt_action[2]:.4f})")
    print(f"  dgripper: {planned_action[6]:.4f} (GT: {gt_action[6]:.4f})")
    
    l1_diff = np.mean(np.abs(planned_action[:3] - gt_action[:3]))
    print(f"\nMean Absolute Error (Translation dx, dy, dz): {l1_diff:.4f} meters")
    print("Action-Conditioned World Modeling evaluation completed successfully!")

if __name__ == "__main__":
    main()
