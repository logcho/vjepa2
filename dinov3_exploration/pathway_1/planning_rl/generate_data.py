#!/usr/bin/env python3
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_data(output_dir, num_trajectories=100, T=8, frame_size=224, radius=15):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)

    frames_all = []
    actions_all = []
    states_all = []

    # Max step size per frame transition in pixels
    max_step = 18

    print(f"Generating {num_trajectories} synthetic trajectories...")
    for traj_idx in range(num_trajectories):
        frames = []
        states = []
        actions = []

        # Start at a random position away from borders
        x = np.random.uniform(radius + 20, frame_size - radius - 20)
        y = np.random.uniform(radius + 20, frame_size - radius - 20)

        for t in range(T):
            # Record state (padded to 7D)
            # We store state as: [x, y, radius, 0, 0, 0, 0] normalized by frame size
            state = np.zeros(7, dtype=np.float32)
            state[0] = (x - frame_size / 2) / (frame_size / 2)
            state[1] = (y - frame_size / 2) / (frame_size / 2)
            state[2] = radius / frame_size
            states.append(state)

            # Generate frame
            frame = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)
            # Draw a white circle
            cv2.circle(frame, (int(x), int(y)), radius, (255, 255, 255), -1)
            frames.append(frame)

            if t < T - 1:
                # Generate a random action/displacement
                dx = np.random.uniform(-max_step, max_step)
                dy = np.random.uniform(-max_step, max_step)

                # Compute target position
                new_x = np.clip(x + dx, radius + 5, frame_size - radius - 5)
                new_y = np.clip(y + dy, radius + 5, frame_size - radius - 5)

                # Actual executed displacement (action) after boundary check
                actual_dx = new_x - x
                actual_dy = new_y - y

                action = np.zeros(7, dtype=np.float32)
                action[0] = actual_dx / (frame_size / 2)  # Normalized action
                action[1] = actual_dy / (frame_size / 2)
                actions.append(action)

                # Update position
                x, y = new_x, new_y

        frames_all.append(np.stack(frames))
        states_all.append(np.stack(states))
        actions_all.append(np.stack(actions))

    frames_all = np.stack(frames_all)      # [B, T, H, W, C]
    states_all = np.stack(states_all)      # [B, T, 7]
    actions_all = np.stack(actions_all)    # [B, T-1, 7]

    print(f"Data generation complete!")
    print(f"Frames shape: {frames_all.shape}")
    print(f"States shape: {states_all.shape}")
    print(f"Actions shape: {actions_all.shape}")

    npz_path = os.path.join(output_dir, "synthetic_data.npz")
    np.savez_compressed(npz_path, observations=frames_all, states=states_all, actions=actions_all)
    print(f"Saved dataset to: {npz_path}")

    # Visualize a sample trajectory
    sample_traj = frames_all[0]
    fig, axes = plt.subplots(1, T, figsize=(15, 3))
    for t in range(T):
        axes[t].imshow(sample_traj[t])
        axes[t].axis("off")
        axes[t].set_title(f"t={t}")
    plt.tight_layout()
    viz_path = os.path.join(output_dir, "visualizations", "sample_trajectory.png")
    plt.savefig(viz_path, dpi=150)
    plt.close()
    print(f"Saved sample trajectory visualization to: {viz_path}")

if __name__ == "__main__":
    generate_synthetic_data(
        output_dir="/Users/loganchoi/Desktop/vjepa2/dinov3_exploration/pathway_1/planning_rl",
        num_trajectories=120, # 100 train, 20 val
        T=8,
        frame_size=224,
        radius=15
    )
