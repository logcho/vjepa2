#!/usr/bin/env python3
import os
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import cv2
from decord import VideoReader

SAMPLE_VIDEO_URL = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/bowling/-WH-lxmGJVY_000005_000015.mp4"
SAMPLE_VIDEO_PATH = "sample_video.mp4"
OUTPUT_DIR = "tubelet_exploration"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

def download_sample_video(url, path):
    if not os.path.exists(path):
        print(f"Downloading sample video from: {url}")
        urllib.request.urlretrieve(url, path)
        print(f"Downloaded sample video to: {path}")
    else:
        print(f"Sample video already exists at: {path}")

def load_video_frames(video_path, num_frames=16, stride=4):
    print(f"Loading video from: {video_path}")
    vr = VideoReader(video_path)
    total_frames = len(vr)
    max_index = min(total_frames, num_frames * stride)
    frame_idx = np.arange(0, max_index, stride)
    
    if len(frame_idx) < num_frames:
        frame_idx = np.pad(frame_idx, (0, num_frames - len(frame_idx)), mode="edge")
        
    frame_idx = frame_idx[:num_frames]
    video_data = vr.get_batch(frame_idx).asnumpy()  # T x H x W x C
    return video_data

def resize_and_center_crop(frames, target_size=(256, 256)):
    T, H, W, C = frames.shape
    shortest_edge = 292
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

def extract_tubelets(frames, tubelet_size=2, patch_size=16):
    """
    Decomposes video frames (T, H, W, C) into tubelet tokens.
    Returns:
      tubelets: numpy array of shape [T_p, H_p, W_p, tubelet_size, patch_size, patch_size, C]
    """
    T, H, W, C = frames.shape
    T_p = T // tubelet_size
    H_p = H // patch_size
    W_p = W // patch_size
    
    # Initialize tubelet container
    tubelets = np.zeros((T_p, H_p, W_p, tubelet_size, patch_size, patch_size, C), dtype=frames.dtype)
    
    for t in range(T_p):
        for y in range(H_p):
            for x in range(W_p):
                t_start = t * tubelet_size
                y_start = y * patch_size
                x_start = x * patch_size
                
                tubelets[t, y, x] = frames[
                    t_start:t_start+tubelet_size,
                    y_start:y_start+patch_size,
                    x_start:x_start+patch_size,
                    :
                ]
                
    return tubelets

def analyze_tubelets():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    download_sample_video(SAMPLE_VIDEO_URL, SAMPLE_VIDEO_PATH)
    
    # 1. Load and process video
    raw_frames = load_video_frames(SAMPLE_VIDEO_PATH, num_frames=16, stride=4)
    cropped_frames = resize_and_center_crop(raw_frames, target_size=(256, 256))
    
    # 2. Extract tubelets
    tubelet_size = 2
    patch_size = 16
    tubelets = extract_tubelets(cropped_frames, tubelet_size, patch_size)
    T_p, H_p, W_p, _, _, _, C = tubelets.shape
    
    # 3. Compute metrics for each tubelet
    # We will compute:
    # A. Spatial Variance (amount of texture inside the tubelet)
    # B. Temporal Difference (motion inside the tubelet)
    n_tubelets = T_p * H_p * W_p
    tubelet_list = []
    
    for t in range(T_p):
        for y in range(H_p):
            for x in range(W_p):
                tubelet = tubelets[t, y, x]  # shape: [2, 16, 16, 3]
                
                # Spatial variance: average variance of pixels inside the patches
                spatial_var = np.var(tubelet)
                
                # Temporal difference (motion): mean absolute difference between the 2 frames
                temp_diff = np.mean(np.abs(tubelet[1].astype(float) - tubelet[0].astype(float)))
                
                # Flattened index matching V-JEPA2's ordering
                flat_idx = t * (H_p * W_p) + y * W_p + x
                
                tubelet_list.append({
                    "t": t, "y": y, "x": x,
                    "flat_idx": flat_idx,
                    "spatial_var": spatial_var,
                    "motion": temp_diff,
                    "data": tubelet
                })
                
    # Sort tubelets by motion magnitude
    sorted_by_motion = sorted(tubelet_list, key=lambda x: x["motion"], reverse=True)
    
    # Plot top 8 high-motion tubelets vs top 8 low-motion tubelets
    fig, axes = plt.subplots(4, 8, figsize=(14, 8))
    
    # High-motion tubelets
    for col in range(8):
        tubelet_info = sorted_by_motion[col]
        # Frame 0
        axes[0, col].imshow(tubelet_info["data"][0])
        axes[0, col].axis('off')
        if col == 0:
            axes[0, col].set_title("Frame 0", fontsize=9, fontweight='semibold')
        # Frame 1
        axes[1, col].imshow(tubelet_info["data"][1])
        axes[1, col].axis('off')
        if col == 0:
            axes[1, col].set_title("Frame 1", fontsize=9, fontweight='semibold')
        # Metadata
        axes[1, col].text(8, 20, f"M: {tubelet_info['motion']:.1f}\nCoord: ({tubelet_info['t']},{tubelet_info['y']},{tubelet_info['x']})", 
                         color='yellow', fontsize=8, ha='center', fontweight='bold',
                         bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))
        
    # Low-motion (static) tubelets
    sorted_by_static = sorted(tubelet_list, key=lambda x: x["motion"])
    for col in range(8):
        tubelet_info = sorted_by_static[col]
        # Frame 0
        axes[2, col].imshow(tubelet_info["data"][0])
        axes[2, col].axis('off')
        if col == 0:
            axes[2, col].set_title("Frame 0 (Static)", fontsize=9, fontweight='semibold')
        # Frame 1
        axes[3, col].imshow(tubelet_info["data"][1])
        axes[3, col].axis('off')
        if col == 0:
            axes[3, col].set_title("Frame 1 (Static)", fontsize=9, fontweight='semibold')
        # Metadata
        axes[3, col].text(8, 20, f"M: {tubelet_info['motion']:.1f}\nCoord: ({tubelet_info['t']},{tubelet_info['y']},{tubelet_info['x']})", 
                         color='cyan', fontsize=8, ha='center', fontweight='bold',
                         bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))
        
    plt.suptitle("Tubelet Segmentation Analysis: Highly Dynamic vs Static Tokens\n(V-JEPA 2.1 Tubelet Tokenizer Grid Representation)", 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "tubelet_motion_analysis.png"), dpi=150)
    plt.close()
    
    # Save statistics and return
    avg_motion = np.mean([x["motion"] for x in tubelet_list])
    max_motion = np.max([x["motion"] for x in tubelet_list])
    avg_var = np.mean([x["spatial_var"] for x in tubelet_list])
    
    print("Tubelet decomposition metrics:")
    print(f"  Total Tubelets Extracted: {n_tubelets}")
    print(f"  Average Tubelet Motion (pixel diff): {avg_motion:.4f}")
    print(f"  Maximum Tubelet Motion: {max_motion:.4f}")
    print(f"  Average Spatial Variance: {avg_var:.4f}")
    print(f"Saved tubelet analysis plot to: {os.path.join(PLOTS_DIR, 'tubelet_motion_analysis.png')}")

if __name__ == "__main__":
    analyze_tubelets()
