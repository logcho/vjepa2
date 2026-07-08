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

def generate_trajectories(num_samples_per_class, T=8, frame_size=448, circle_radius=30):
    """
    Generates trajectories for 8 different action classes.
    Returns list of trajectories (each is a list of T (x, y) coordinates) and labels.
    """
    trajectories = []
    labels = []
    classes_desc = [
        "Horizontal Left-to-Right",
        "Horizontal Right-to-Left",
        "Vertical Bottom-to-Top",
        "Vertical Top-to-Bottom",
        "Clockwise Circle",
        "Counter-Clockwise Circle",
        "Diagonal Top-Left to Bottom-Right",
        "Diagonal Bottom-Left to Top-Right"
    ]

    for label in range(8):
        for _ in range(num_samples_per_class):
            coords = []
            
            # Base start position, step size, and angle variables with noise
            if label == 0:  # Horizontal Left-to-Right
                x = np.random.uniform(40, 80)
                y = np.random.uniform(200, 248)
                dx = np.random.uniform(40, 50)
                dy = np.random.uniform(-4, 4)
                for t in range(T):
                    coords.append((x + t*dx, y + t*dy))
                    
            elif label == 1:  # Horizontal Right-to-Left
                x = np.random.uniform(368, 408)
                y = np.random.uniform(200, 248)
                dx = np.random.uniform(-50, -40)
                dy = np.random.uniform(-4, 4)
                for t in range(T):
                    coords.append((x + t*dx, y + t*dy))
                    
            elif label == 2:  # Vertical Bottom-to-Top
                x = np.random.uniform(200, 248)
                y = np.random.uniform(368, 408)
                dx = np.random.uniform(-4, 4)
                dy = np.random.uniform(-50, -40)
                for t in range(T):
                    coords.append((x + t*dx, y + t*dy))
                    
            elif label == 3:  # Vertical Top-to-Bottom
                x = np.random.uniform(200, 248)
                y = np.random.uniform(40, 80)
                dx = np.random.uniform(-4, 4)
                dy = np.random.uniform(40, 50)
                for t in range(T):
                    coords.append((x + t*dx, y + t*dy))
                    
            elif label == 4 or label == 5:  # Circles (4: CW, 5: CCW)
                cx = np.random.uniform(210, 238)
                cy = np.random.uniform(210, 238)
                r = np.random.uniform(100, 130)
                start_angle = np.random.uniform(0, 2*np.pi)
                #CW uses negative step, CCW uses positive step
                step = -2*np.pi / 8.5 if label == 4 else 2*np.pi / 8.5
                for t in range(T):
                    angle = start_angle + t * step
                    coords.append((cx + r*np.cos(angle), cy + r*np.sin(angle)))
                    
            elif label == 6:  # Diagonal Top-Left to Bottom-Right
                x = np.random.uniform(40, 80)
                y = np.random.uniform(40, 80)
                dx = np.random.uniform(40, 50)
                dy = np.random.uniform(40, 50)
                for t in range(T):
                    coords.append((x + t*dx, y + t*dy))
                    
            elif label == 7:  # Diagonal Bottom-Left to Top-Right
                x = np.random.uniform(40, 80)
                y = np.random.uniform(368, 408)
                dx = np.random.uniform(40, 50)
                dy = np.random.uniform(-50, -40)
                for t in range(T):
                    coords.append((x + t*dx, y + t*dy))
            
            # Clip coords to boundaries
            coords = [(np.clip(x, circle_radius, frame_size - circle_radius),
                       np.clip(y, circle_radius, frame_size - circle_radius))
                      for x, y in coords]
            trajectories.append(coords)
            labels.append(label)

    return trajectories, labels, classes_desc

def main():
    print("=== Generating Synthetic Video Classification Dataset ===")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    output_dir = "/Users/loganchoi/Desktop/vjepa2/dinov3_exploration/pathway_1/video_classification"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
    # 25 samples per class (20 train, 5 validation)
    num_samples_per_class = 25
    T = 8
    frame_size = 448
    circle_radius = 30
    
    trajectories, labels, classes_desc = generate_trajectories(num_samples_per_class, T, frame_size, circle_radius)
    num_videos = len(trajectories)
    print(f"Generated {num_videos} trajectories across 8 classes.")
    
    # Load DINOv3 model
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
        
    print("Extracting DINOv3 features for all videos...")
    
    # Preprocessing constants
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    
    extracted_features = []
    # Save coordinate states as well for completeness/debugging
    states = []
    
    with torch.no_grad():
        for i, (coords, label) in enumerate(zip(trajectories, labels)):
            if (i + 1) % 20 == 0 or i == 0:
                print(f"Processing video {i + 1}/{num_videos}...")
            
            # Generate frames
            frames = []
            state = []
            for t in range(T):
                cx, cy = coords[t]
                # Draw black frame
                frame = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)
                # Draw white circle
                cv2.circle(frame, (int(cx), int(cy)), circle_radius, (255, 255, 255), -1)
                frames.append(frame)
                
                # State coordinate representation normalized between -1 and 1
                state.append([(cx - frame_size/2) / (frame_size/2), (cy - frame_size/2) / (frame_size/2)])
                
            states.append(np.array(state, dtype=np.float32))
            
            # Preprocess and pass to DINOv3
            frames_np = np.stack(frames) # [T, H, W, C]
            tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float().to(device) / 255.0
            normalized = (tensor - mean) / std
            
            outputs = dinov3_model.forward_features(normalized)
            patch_tokens = outputs["x_norm_patchtokens"].cpu()  # [T, N_patches, Embed_Dim]
            
            extracted_features.append(patch_tokens)
            
    extracted_features = torch.stack(extracted_features).numpy()  # [N_videos, T, N_patches, Embed_Dim]
    labels = np.array(labels, dtype=np.int64)
    states = np.stack(states) # [N_videos, T, 2]
    
    print(f"Extracted features shape: {extracted_features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"States shape: {states.shape}")
    
    npz_path = os.path.join(output_dir, "synthetic_data.npz")
    np.savez_compressed(npz_path, features=extracted_features, labels=labels, states=states, classes=classes_desc)
    print(f"Saved dataset to: {npz_path}")
    
    # Save a trajectory visualization for each class to inspect the generator
    plt.figure(figsize=(15, 8))
    for c_idx in range(8):
        # find first sample of class c_idx
        idx = c_idx * num_samples_per_class
        coords = trajectories[idx]
        
        plt.subplot(2, 4, c_idx + 1)
        # Plot circle centers with lines
        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]
        plt.plot(xs, ys, 'o-', linewidth=2, label="Centers")
        plt.scatter([xs[0]], [ys[0]], color='green', s=100, label="Start", zorder=5) # Start green
        plt.scatter([xs[-1]], [ys[-1]], color='red', s=100, label="End", zorder=5)   # End red
        plt.xlim(0, frame_size)
        plt.ylim(frame_size, 0) # Flip y-axis to match image space coordinate system
        plt.title(classes_desc[c_idx], fontsize=10, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.5)
        if c_idx == 0:
            plt.legend(prop={'size': 7})
            
    plt.suptitle("Generated Trajectory Coordinate Paths", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    viz_path = os.path.join(output_dir, "visualizations", "trajectory_paths.png")
    plt.savefig(viz_path, dpi=150)
    plt.close()
    print(f"Saved trajectory paths visualization to: {viz_path}")
    
    # Let's save a single video's actual frame sequence as well
    fig, axes = plt.subplots(1, T, figsize=(15, 2.5))
    first_idx = 4 * num_samples_per_class # Class 4: Clockwise Circle
    coords = trajectories[first_idx]
    for t in range(T):
        cx, cy = coords[t]
        frame = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)
        cv2.circle(frame, (int(cx), int(cy)), circle_radius, (255, 255, 255), -1)
        axes[t].imshow(frame)
        axes[t].axis("off")
        axes[t].set_title(f"t={t}")
    plt.suptitle(f"Sample Frame Sequence: {classes_desc[4]}", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    seq_path = os.path.join(output_dir, "visualizations", "sample_sequence.png")
    plt.savefig(seq_path, dpi=150)
    plt.close()
    print(f"Saved sample frame sequence image to: {seq_path}")

if __name__ == "__main__":
    main()
