#!/usr/bin/env python3
"""
Option 1: Dense Feature Visualization and PCA-based Temporal Tracking for V-JEPA 2.1.
This script extracts dense representations from V-JEPA 2.1, reduces their dimensions via PCA,
and generates visualization overlays to track object parts and shapes across motions.
"""

import os
import argparse
import urllib.request
import numpy as np
import torch
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
    Simulates the Hugging Face VJEPA2VideoProcessor geometry preprocessing
    so that visual overlays align perfectly.
    """
    T, H, W, C = frames.shape
    print(f"Original frames shape for visualization: {frames.shape}")
    
    # 1. Resize shortest edge to `shortest_edge`
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
    
    # 2. Center crop to target_size
    th, tw = target_size
    start_y = (new_h - th) // 2
    start_x = (new_w - tw) // 2
    cropped_frames = resized_frames[:, start_y:start_y+th, start_x:start_x+tw, :]
    print(f"Cropped frames shape for visualization: {cropped_frames.shape}")
    
    return cropped_frames


def extract_features(video, model_name, device):
    """
    Loads model and processor, runs inference and extracts dense token features.
    """
    print(f"Loading processor for: {model_name}...")
    processor = AutoVideoProcessor.from_pretrained(model_name)
    
    print(f"Loading model: {model_name}...")
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # Prep video tensor (T x H x W x C) -> (T x C x H x W)
    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)
    inputs = processor(video_tensor, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print(f"Inputs to model: {inputs['pixel_values_videos'].shape} (B x T x C x H x W)")

    try:
        with torch.no_grad():
            outputs = model(**inputs)
        features = outputs.last_hidden_state
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "mps backend out of memory" in str(e).lower():
            print(f"\n[Warning] GPU/MPS inference failed due to memory limit: {e}")
            print("Falling back to CPU device for inference...")
            model = model.to("cpu")
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            features = outputs.last_hidden_state
        else:
            raise e

    # features shape: (1, num_tokens, embed_dim)
    return features.squeeze(0), model.config


def compute_pca(features, t_p, h_p, w_p, pca_per_frame=False):
    """
    Computes PCA (3 components) on dense feature embeddings using PyTorch SVD/lowrank.
    """
    # features shape: (T_p * H_p * W_p, embed_dim)
    embed_dim = features.shape[-1]
    
    if not pca_per_frame:
        print("Computing global PCA across the entire video...")
        mean = features.mean(dim=0, keepdim=True)
        centered = features - mean
        
        # SVD/lowrank approximation
        U, S, V = torch.pca_lowrank(centered, q=3, center=False)
        projected = torch.matmul(centered, V[:, :3])  # (T_p * H_p * W_p, 3)
        projected = projected.view(t_p, h_p, w_p, 3)
    else:
        print("Computing PCA per frame...")
        features_reshaped = features.view(t_p, h_p * w_p, embed_dim)
        projected_frames = []
        for t in range(t_p):
            frame_feats = features_reshaped[t]
            mean = frame_feats.mean(dim=0, keepdim=True)
            centered = frame_feats - mean
            U, S, V = torch.pca_lowrank(centered, q=3, center=False)
            proj = torch.matmul(centered, V[:, :3])  # (H_p * W_p, 3)
            projected_frames.append(proj.view(h_p, w_p, 3))
        projected = torch.stack(projected_frames, dim=0) # (T_p, H_p, W_p, 3)
        
    return projected


def normalize_components(projected):
    """
    Normalizes the projected PCA components to [0, 1] range using robust quantiles.
    """
    print("Normalizing PCA components using 1% and 99% quantiles...")
    normalized = torch.zeros_like(projected)
    for i in range(3):
        comp = projected[..., i]
        # Quantiles are computed over the entire flattened dimension of this component
        q_low = torch.quantile(comp, 0.01)
        q_high = torch.quantile(comp, 0.99)
        
        denom = q_high - q_low if q_high > q_low else 1e-8
        clamped = torch.clamp(comp, min=q_low, max=q_high)
        normalized[..., i] = (clamped - q_low) / denom
        
    return normalized


def interpolate_and_render(projected_pca, target_t, target_h, target_w):
    """
    Interpolates the PCA feature maps using trilinear mode to original/cropped resolution.
    """
    # projected_pca shape: (T_p, H_p, W_p, 3)
    # Permute to (1, 3, T_p, H_p, W_p) for grid interpolation
    pca_tensor = projected_pca.permute(3, 0, 1, 2).unsqueeze(0)
    
    print(f"Interpolating PCA features to target size: {target_t}x{target_h}x{target_w}...")
    interpolated = F.interpolate(
        pca_tensor,
        size=(target_t, target_h, target_w),
        mode='trilinear',
        align_corners=False
    )
    
    # Convert back to (T, H, W, 3) numpy array
    pca_numpy = interpolated.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
    
    # Scale to [0, 255] range
    pca_numpy = (pca_numpy * 255.0).astype(np.uint8)
    return pca_numpy


def save_video_clip(frames, output_path, fps=15.0):
    """
    Saves a numpy video array of shape (T, H, W, 3) as an MP4 video file using OpenCV.
    """
    T, H, W, C = frames.shape
    # OpenCV VideoWriter uses BGR, so convert RGB to BGR
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    for t in range(T):
        frame_bgr = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()
    print(f"Saved video to: {output_path}")


def save_keyframe_grid(original, pca, overlay, output_path, num_keyframes=8):
    """
    Saves a grid comparing original frames, PCA visualizations, and overlays.
    """
    T = original.shape[0]
    indices = np.linspace(0, T - 1, num_keyframes, dtype=int)
    
    fig, axes = plt.subplots(3, num_keyframes, figsize=(2 * num_keyframes, 6))
    
    for idx, frame_idx in enumerate(indices):
        # Row 0: Original
        axes[0, idx].imshow(original[frame_idx])
        axes[0, idx].axis('off')
        if idx == 0:
            axes[0, idx].set_ylabel("Original", fontsize=12, labelpad=10)
            
        # Row 1: PCA Color Map
        axes[1, idx].imshow(pca[frame_idx])
        axes[1, idx].axis('off')
        if idx == 0:
            axes[1, idx].set_ylabel("PCA Map", fontsize=12, labelpad=10)
            
        # Row 2: Overlay
        axes[2, idx].imshow(overlay[frame_idx])
        axes[2, idx].axis('off')
        if idx == 0:
            axes[2, idx].set_ylabel("Overlay", fontsize=12, labelpad=10)
            
        axes[0, idx].set_title(f"F {frame_idx}", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved keyframe grid image to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="V-JEPA 2.1 Dense Feature PCA Visualization")
    parser.add_argument("--video_path", type=str, default=SAMPLE_VIDEO_PATH, help="Path to input video")
    parser.add_argument("--model_name", type=str, default=HF_MODEL_NAME, help="Hugging Face model ID")
    parser.add_argument("--output_dir", type=str, default="option1/visualizations", help="Output directory")
    parser.add_argument("--num_frames", type=str, default="64", help="Number of frames to sample")
    parser.add_argument("--stride", type=str, default="2", help="Sampling stride")
    parser.add_argument("--pca_per_frame", action="store_true", help="Compute PCA per frame instead of globally")
    parser.add_argument("--alpha", type=str, default="0.5", help="Alpha blending weight for overlay")
    args = parser.parse_args()

    num_frames = int(args.num_frames)
    stride = int(args.stride)
    alpha = float(args.alpha)

    # 1. Download default video if needed
    if args.video_path == SAMPLE_VIDEO_PATH:
        download_sample_video(SAMPLE_VIDEO_URL, SAMPLE_VIDEO_PATH)

    # 2. Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 3. Load original video
    raw_video = load_video_frames(args.video_path, num_frames=num_frames, stride=stride)
    
    # 4. Geometry Preprocessing for visual comparison (resize shortest edge -> center crop to 256x256)
    cropped_video = resize_and_center_crop(raw_video, target_size=(256, 256), shortest_edge=292)

    # 5. Determine Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 6. Extract token features
    features, config = extract_features(cropped_video, args.model_name, device)
    print(f"Extracted features shape (flat): {features.shape} (Num Tokens x Embedding Dim)")

    # 7. Map tokens to spatial-temporal grid dimensions
    # Pre-trained models use specific tubelet and patch size tokenization
    # For ViT-L fpc64-256: tubelet_size=2, patch_size=16
    tubelet_size = getattr(config, "tubelet_size", 2)
    patch_size = getattr(config, "patch_size", 16)
    
    # Calculate grid dimensions based on target model crop size (256x256)
    t_p = num_frames // tubelet_size
    h_p = 256 // patch_size
    w_p = 256 // patch_size
    
    print(f"Configured grid dimensions: T_p={t_p}, H_p={h_p}, W_p={w_p}")
    expected_tokens = t_p * h_p * w_p
    if features.shape[0] != expected_tokens:
        print(f"Warning: Extracted tokens {features.shape[0]} doesn't match expected {expected_tokens}. "
              "Falling back to dynamic grid parsing.")
        # If tokens do not match, we'll try to deduce the closest sizes
        t_p = num_frames // tubelet_size
        h_p = int(np.sqrt(features.shape[0] / t_p))
        w_p = h_p
        print(f"Deduced grid dimensions: T_p={t_p}, H_p={h_p}, W_p={w_p}")

    # 8. Compute PCA Projection
    projected = compute_pca(features, t_p, h_p, w_p, pca_per_frame=args.pca_per_frame)

    # 9. Normalize components to [0, 1] range
    normalized_pca = normalize_components(projected)

    # 10. Interpolate features to original crop size (T x 256 x 256)
    pca_frames = interpolate_and_render(normalized_pca, num_frames, 256, 256)

    # 11. Create Overlays and Concatenation
    overlay_frames = []
    side_by_side_frames = []
    
    for t in range(num_frames):
        orig_f = cropped_video[t]
        pca_f = pca_frames[t]
        
        # Blend overlay
        blended = cv2.addWeighted(orig_f, 1.0 - alpha, pca_f, alpha, 0)
        overlay_frames.append(blended)
        
        # Side by side
        sbs = np.concatenate([orig_f, pca_f], axis=1)
        side_by_side_frames.append(sbs)
        
    overlay_frames = np.stack(overlay_frames)
    side_by_side_frames = np.stack(side_by_side_frames)

    # 12. Save videos
    save_video_clip(pca_frames, os.path.join(args.output_dir, "pca_raw.mp4"))
    save_video_clip(overlay_frames, os.path.join(args.output_dir, "pca_overlay.mp4"))
    save_video_clip(side_by_side_frames, os.path.join(args.output_dir, "pca_side_by_side.mp4"))

    # 13. Save static Keyframe Grid
    save_keyframe_grid(
        cropped_video,
        pca_frames,
        overlay_frames,
        os.path.join(args.output_dir, "keyframe_grid.png"),
        num_keyframes=8
    )
    print("\nSUCCESS! PCA feature visualization files generated in:", args.output_dir)


if __name__ == "__main__":
    main()
