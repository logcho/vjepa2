#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add workspace search paths
sys.path.append("/Users/loganchoi/Desktop/vjepa2/vjepa2")
sys.path.append("/Users/loganchoi/Desktop/dinov3/dinov3")

from src.models.ac_predictor import VisionTransformerPredictorAC

def draw_frame(x, y, frame_size=224, radius=15):
    frame = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)
    cv2.circle(frame, (int(x), int(y)), radius, (255, 255, 255), -1)
    return frame

def extract_features(model, frame, device):
    # frame: [H, W, 3] np.uint8
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    
    tensor = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    normalized = (tensor - mean) / std
    
    with torch.no_grad():
        outputs = model.forward_features(normalized)
        patch_tokens = outputs["x_norm_patchtokens"] # [1, 196, 384]
    return patch_tokens

def cem_planner(
    predictor, 
    z_start, 
    state_start, 
    z_goal, 
    rollout=7, 
    cem_steps=10, 
    samples=600, 
    topk=20, 
    maxnorm=0.16, # approx 18 pixels in normalized coords
    device="cpu"
):
    # z_start: [1, 196, 384]
    # state_start: [1, 7]
    # z_goal: [1, 196, 384]
    
    # Initialize mean and std of action distribution
    # mean: [rollout, 2] (for dx, dy)
    # std: [rollout, 2]
    mean = torch.zeros((rollout, 2), device=device)
    std = torch.ones((rollout, 2), device=device) * (maxnorm / 2.0)
    
    momentum_mean = 0.15
    momentum_std = 0.75
    
    for step in range(cem_steps):
        # 1. Sample actions: [samples, rollout, 2]
        eps = torch.randn(samples, rollout, 2, device=device)
        action_samples = eps * std.unsqueeze(0) + mean.unsqueeze(0)
        action_samples = torch.clamp(action_samples, -maxnorm, maxnorm)
        
        # Pad to 7D
        actions_7d = torch.zeros(samples, rollout, 7, device=device)
        actions_7d[:, :, :2] = action_samples
        
        # 2. Rollout future latents
        z_curr = z_start.repeat(samples, 1, 1)                  # [samples, 196, 384]
        state_curr = state_start.repeat(samples, 1).unsqueeze(1) # [samples, 1, 7]
        
        z_seq = z_curr
        states_seq = state_curr
        actions_seq = None
        
        for h in range(rollout):
            act_h = actions_7d[:, h:h+1] # [samples, 1, 7]
            if actions_seq is None:
                actions_seq = act_h
            else:
                actions_seq = torch.cat([actions_seq, act_h], dim=1) # [samples, h+1, 7]
                
            with torch.no_grad():
                y_pred = predictor(z_seq, actions_seq, states_seq)
                z_next = y_pred[:, -196:] # last step's frame prediction
                
            state_next = states_seq[:, -1:].clone()
            state_next[:, 0, :2] = torch.clamp(state_next[:, 0, :2] + act_h[:, 0, :2], -1.0, 1.0)
            
            # Append next step to history
            z_seq = torch.cat([z_seq, z_next], dim=1)
            states_seq = torch.cat([states_seq, state_next], dim=1)
            
        # 3. Evaluate final state representation distance to goal
        z_final = z_seq[:, -196:]
        loss = torch.mean(torch.abs(z_final - z_goal.repeat(samples, 1, 1)), dim=[1, 2])
        
        # Select top-k samples
        values, indices = torch.topk(loss, topk, largest=False)
        selected_actions = action_samples[indices]
        
        # Update mean and std
        new_mean = selected_actions.mean(dim=0)
        new_std = selected_actions.std(dim=0)
        
        mean = new_mean * (1.0 - momentum_mean) + mean * momentum_mean
        std = new_std * (1.0 - momentum_std) + std * momentum_std
        std = torch.clamp(std, min=1e-3)
        
    return mean

def main():
    print("=== Latent State Planning using CEM ===")
    
    # 1. Device Setup
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # 2. Load Models
    print("Loading pretrained DINOv3 backbone...")
    dinov3_model = torch.hub.load(
        '/Users/loganchoi/Desktop/dinov3/dinov3', 
        'dinov3_vits16', 
        source='local', 
        pretrained=True
    ).to(device)
    dinov3_model.eval()

    print("Loading trained Action-Conditioned Predictor...")
    predictor = VisionTransformerPredictorAC(
        img_size=(224, 224),
        patch_size=16,
        num_frames=7,
        tubelet_size=1,
        embed_dim=384,
        predictor_embed_dim=192,
        depth=4,
        num_heads=6,
        action_embed_dim=7
    ).to(device)
    
    weights_path = "/Users/loganchoi/Desktop/vjepa2/dinov3_exploration/pathway_1/planning_rl/predictor_ac_pathway1.pth"
    predictor.load_state_dict(torch.load(weights_path, map_location=device))
    predictor.eval()
    print("Models loaded successfully.")

    # 3. Setup Start and Goal Configurations
    frame_size = 224
    radius = 15
    
    start_x, start_y = 50.0, 50.0
    goal_x, goal_y = 170.0, 170.0
    
    start_frame = draw_frame(start_x, start_y, frame_size, radius)
    goal_frame = draw_frame(goal_x, goal_y, frame_size, radius)
    
    # Normalize states to [-1, 1]
    state_start = np.zeros((1, 7), dtype=np.float32)
    state_start[0, 0] = (start_x - frame_size / 2) / (frame_size / 2)
    state_start[0, 1] = (start_y - frame_size / 2) / (frame_size / 2)
    state_start[0, 2] = radius / frame_size
    state_start_tensor = torch.tensor(state_start, device=device)
    
    z_start = extract_features(dinov3_model, start_frame, device)
    z_goal = extract_features(dinov3_model, goal_frame, device)
    
    print(f"Planning task: Start ({start_x:.1f}, {start_y:.1f}) -> Goal ({goal_x:.1f}, {goal_y:.1f})")

    # 4. Run CEM planning
    rollout_steps = 7
    planned_actions = cem_planner(
        predictor=predictor,
        z_start=z_start,
        state_start=state_start_tensor,
        z_goal=z_goal,
        rollout=rollout_steps,
        cem_steps=15,
        samples=150,
        topk=25,
        maxnorm=0.16, # max step ~18 pixels
        device=device
    )
    
    actions_np = planned_actions.cpu().numpy()
    print("\nPlanned action sequence (normalized offsets dx, dy):")
    for step in range(rollout_steps):
        dx, dy = actions_np[step]
        dx_pixels = dx * (frame_size / 2)
        dy_pixels = dy * (frame_size / 2)
        print(f"  Step {step}: dx = {dx:.4f} ({dx_pixels:+.1f} px), dy = {dy:.4f} ({dy_pixels:+.1f} px)")

    # 5. Execute planned actions and generate sequence of frames
    x_curr, y_curr = start_x, start_y
    trajectory_frames = [start_frame]
    actual_positions = [(x_curr, y_curr)]
    
    for step in range(rollout_steps):
        dx, dy = actions_np[step]
        dx_pixels = dx * (frame_size / 2)
        dy_pixels = dy * (frame_size / 2)
        
        # Apply physics transition in environment
        new_x = np.clip(x_curr + dx_pixels, radius + 5, frame_size - radius - 5)
        new_y = np.clip(y_curr + dy_pixels, radius + 5, frame_size - radius - 5)
        
        x_curr, y_curr = new_x, new_y
        actual_positions.append((x_curr, y_curr))
        
        frame = draw_frame(x_curr, y_curr, frame_size, radius)
        trajectory_frames.append(frame)

    # 6. Visualize planned trajectory
    fig, axes = plt.subplots(1, rollout_steps + 2, figsize=(18, 3.5))
    
    # Start Frame
    axes[0].imshow(start_frame)
    axes[0].set_title("Start Frame (0)", color="#4F46E5", fontweight="bold")
    axes[0].axis("off")
    
    # Trajectory Frames
    for step in range(rollout_steps):
        axes[step + 1].imshow(trajectory_frames[step + 1])
        axes[step + 1].set_title(f"Step {step + 1}", color="#10B981")
        axes[step + 1].axis("off")
        
    # Goal Frame
    axes[rollout_steps + 1].imshow(goal_frame)
    axes[rollout_steps + 1].set_title("Goal Frame", color="#EF4444", fontweight="bold")
    axes[rollout_steps + 1].axis("off")
    
    plt.tight_layout()
    viz_dir = "/Users/loganchoi/Desktop/vjepa2/dinov3_exploration/pathway_1/planning_rl/visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    traj_path = os.path.join(viz_dir, "planning_trajectory.png")
    plt.savefig(traj_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved planning trajectory visualization to: {traj_path}")

    # Generate smooth planning trajectory video
    print("Generating smooth trajectory video...")
    all_frames = []
    fps = 30
    frames_per_step = 15
    for i in range(len(actual_positions) - 1):
        p_start = actual_positions[i]
        p_end = actual_positions[i+1]
        for f in range(frames_per_step):
            alpha = f / frames_per_step
            curr_x = p_start[0] + alpha * (p_end[0] - p_start[0])
            curr_y = p_start[1] + alpha * (p_end[1] - p_start[1])
            
            frame = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)
            
            # Goal target indicator (faded green)
            cv2.circle(frame, (int(goal_x), int(goal_y)), radius, (0, 100, 0), 2)
            cv2.putText(frame, "G", (int(goal_x) - 5, int(goal_y) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
            
            # Path history (cyan line)
            for idx in range(i):
                cv2.line(frame, (int(actual_positions[idx][0]), int(actual_positions[idx][1])), 
                         (int(actual_positions[idx+1][0]), int(actual_positions[idx+1][1])), (200, 200, 0), 1)
            cv2.line(frame, (int(actual_positions[i][0]), int(actual_positions[i][1])), 
                     (int(curr_x), int(curr_y)), (200, 200, 0), 1)
                     
            # Circle object (white)
            cv2.circle(frame, (int(curr_x), int(curr_y)), radius, (255, 255, 255), -1)
            
            # HUD Overlay
            cv2.putText(frame, f"Step: {i} Frame: {f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(frame, f"Pos: ({curr_x:.1f}, {curr_y:.1f})", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(frame, f"Goal: ({goal_x:.1f}, {goal_y:.1f})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
            
            all_frames.append(frame)
            
    # Final state static padding
    final_frame = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)
    cv2.circle(final_frame, (int(goal_x), int(goal_y)), radius, (0, 255, 0), 2)
    cv2.circle(final_frame, (int(actual_positions[-1][0]), int(actual_positions[-1][1])), radius, (255, 255, 255), -1)
    for idx in range(len(actual_positions) - 1):
        cv2.line(final_frame, (int(actual_positions[idx][0]), int(actual_positions[idx][1])), 
                 (int(actual_positions[idx+1][0]), int(actual_positions[idx+1][1])), (200, 200, 0), 1)
    cv2.putText(final_frame, "Step: 7 (Goal Reached)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.putText(final_frame, f"Pos: ({actual_positions[-1][0]:.1f}, {actual_positions[-1][1]:.1f})", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    for _ in range(30):
        all_frames.append(final_frame)
        
    # Write to file
    video_path = os.path.join(viz_dir, "planning_trajectory.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (frame_size, frame_size))
    for f in all_frames:
        out.write(f)
    out.release()
    print(f"Saved planning trajectory video to: {video_path}")

    # Check goal reaching accuracy
    final_pos = actual_positions[-1]
    dist_to_goal = np.sqrt((final_pos[0] - goal_x)**2 + (final_pos[1] - goal_y)**2)
    print(f"Final reached position: ({final_pos[0]:.2f}, {final_pos[1]:.2f})")
    print(f"Target goal position: ({goal_x:.2f}, {goal_y:.2f})")
    print(f"Goal reaching distance error: {dist_to_goal:.2f} pixels (approx. {dist_to_goal/frame_size*100:.1f}% of frame size)")

    # 7. Compute 2D Action Energy Landscape
    print("\nComputing Action Energy Landscape (dx vs dy offsets)...")
    grid_res = 21
    grid_range = 0.20 # range of action offsets (normalized)
    
    dx_vals = np.linspace(-grid_range, grid_range, grid_res)
    dy_vals = np.linspace(-grid_range, grid_range, grid_res)
    
    action_candidates = []
    for dy in dy_vals:
        for dx in dx_vals:
            action_candidates.append([dx, dy, 0.0, 0.0, 0.0, 0.0, 0.0])
            
    action_candidates = torch.tensor(action_candidates, device=device, dtype=torch.float32).unsqueeze(1) # [S, 1, 7]
    S = len(action_candidates)
    
    # Replicate context state and frame tokens for parallel prediction
    z_0_repeated = z_start.repeat(S, 1, 1)
    states_0_repeated = state_start_tensor.repeat(S, 1, 1)
    z_goal_repeated = z_goal.repeat(S, 1, 1)
    
    with torch.no_grad():
        # Predict representation of next frame
        y_pred_candidates = predictor(z_0_repeated, action_candidates, states_0_repeated)
        z_1_candidates = y_pred_candidates[:, -196:]
        
        # Calculate prediction errors to the GOAL frame
        # Lower loss indicates the action moves us closer to the goal
        losses = torch.mean(torch.abs(z_1_candidates - z_goal_repeated), dim=[1, 2]).cpu().numpy()
        
    loss_grid = losses.reshape(grid_res, grid_res)
    
    # Find minimum energy action direction
    min_idx = np.argmin(losses)
    min_dx = action_candidates[min_idx, 0, 0].item()
    min_dy = action_candidates[min_idx, 0, 1].item()
    
    # Ground-truth direction towards goal
    gt_dir_x = (goal_x - start_x)
    gt_dir_y = (goal_y - start_y)
    gt_dir_norm = np.sqrt(gt_dir_x**2 + gt_dir_y**2)
    # Scale ground truth direction for visual overlay
    gt_dx_norm = (gt_dir_x / gt_dir_norm) * 0.12
    gt_dy_norm = (gt_dir_y / gt_dir_norm) * 0.12
    
    # Plot Energy Landscape Heatmap
    plt.figure(figsize=(8, 6.5))
    heatmap = plt.imshow(
        loss_grid, 
        extent=[-grid_range, grid_range, -grid_range, grid_range], 
        origin='lower', 
        cmap='viridis',
        aspect='auto'
    )
    plt.colorbar(heatmap, label="Representation distance to Goal Frame")
    
    # Ground Truth target direction arrow/star
    plt.scatter(gt_dx_norm, gt_dy_norm, color='red', marker='*', s=150, edgecolors='white', label='Direct Goal Direction')
    # Minimum Energy point predicted
    plt.scatter(min_dx, min_dy, color='cyan', marker='o', s=80, edgecolors='black', label='Minimum Energy Action')
    
    plt.xlabel("Action Offset DX (normalized)", fontsize=11)
    plt.ylabel("Action Offset DY (normalized)", fontsize=11)
    plt.title("Action-Conditioned V-JEPA Energy Landscape to Goal", fontsize=12, fontweight='bold', pad=12)
    plt.legend(loc='upper right')
    
    landscape_path = os.path.join(viz_dir, "planning_energy_landscape.png")
    plt.savefig(landscape_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Action Energy Landscape heatmap to: {landscape_path}")

if __name__ == "__main__":
    main()
