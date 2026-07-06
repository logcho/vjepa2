import os
import sys
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

# Add workspace directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from option2.motion_guided_masking import MotionMaskGenerator
from decord import VideoReader

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Option 2 V-JEPA Spatial-Temporal Masking Strategies")
    parser.add_argument("--video_path", type=str, default="sample_video.mp4", help="Path to input video")
    parser.add_argument("--output_dir", type=str, default="option2/visualizations", help="Output directory")
    parser.add_argument("--num_frames", type=str, default="64", help="Number of frames to sample")
    parser.add_argument("--crop_size", type=int, default=256, help="Spatial crop size")
    parser.add_argument("--patch_size", type=int, default=16, help="Spatial patch size")
    parser.add_argument("--tubelet_size", type=int, default=2, help="Temporal tubelet size")
    parser.add_argument("--temp", type=float, default=0.05, help="Motion sampling temperature")
    return parser.parse_args()

def load_video(path, num_frames=64, crop_size=256):
    print(f"Loading video from: {path}")
    vr = VideoReader(path)
    total_frames = len(vr)
    print(f"Total video frames: {total_frames}, FPS: {vr.get_avg_fps():.2f}")
    
    # Calculate uniform sampling frame indices
    stride = max(1, total_frames // num_frames)
    indices = np.arange(0, num_frames) * stride
    indices = np.clip(indices, 0, total_frames - 1).astype(np.int64)
    print(f"Sampling {num_frames} frames with indices: {indices}")
    
    frames = vr.get_batch(indices).asnumpy()  # shape (T, H, W, C)
    
    # Crop to square crop_size x crop_size (center crop)
    T, H, W, C = frames.shape
    short_side = min(H, W)
    
    crop_h = (H - short_side) // 2
    crop_w = (W - short_side) // 2
    cropped = frames[:, crop_h:crop_h+short_side, crop_w:crop_w+short_side, :]
    
    # Resize to crop_size x crop_size
    resized = np.zeros((T, crop_size, crop_size, C), dtype=np.uint8)
    for i in range(T):
        resized[i] = cv2.resize(cropped[i], (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
        
    return resized

def generate_motion_heatmap(motion_map, target_shape):
    # motion_map: (duration, height, width)
    # target_shape: (T, H, W, C)
    T, H, W, C = target_shape
    duration, height, width = motion_map.shape
    tubelet_size = T // duration
    
    # Scale motion map to 0-255
    min_val = motion_map.min()
    max_val = motion_map.max()
    if max_val > min_val:
        norm_map = (motion_map - min_val) / (max_val - min_val)
    else:
        norm_map = motion_map
        
    # Convert to numpy
    norm_map_np = norm_map.cpu().numpy()
    
    heatmaps = np.zeros((T, H, W, C), dtype=np.uint8)
    for t in range(T):
        # Retrieve mapped motion map frame
        m_frame = norm_map_np[t // tubelet_size]
        # Resize to full H, W
        m_resized = cv2.resize(m_frame, (W, H), interpolation=cv2.INTER_NEAREST)
        m_uint8 = (m_resized * 255).astype(np.uint8)
        # Apply colormap (Jet)
        color_heatmap = cv2.applyColorMap(m_uint8, cv2.COLORMAP_JET)
        heatmaps[t] = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
        
    return heatmaps

def apply_mask_overlay(frames, masks_pred, patch_size, tubelet_size):
    # frames: (T, H, W, C)
    # masks_pred: 1D tensor of flat predictor (target) indices
    T, H, W, C = frames.shape
    duration = T // tubelet_size
    height = H // patch_size
    width = W // patch_size
    
    # Flatten grid
    mask_grid = torch.ones((duration, height, width), dtype=torch.float32)
    # Predictor targets are 0, context tokens are 1
    mask_grid.flatten()[masks_pred] = 0.0
    mask_grid_np = mask_grid.cpu().numpy()
    
    overlayed = frames.copy()
    overlay_color = np.array([200, 20, 20], dtype=np.uint8) # Translucent red for masked areas
    
    for t in range(T):
        m_frame = mask_grid_np[t // tubelet_size]
        # Upscale mask to full resolution
        m_resized = cv2.resize(m_frame, (W, H), interpolation=cv2.INTER_NEAREST)
        m_resized = np.expand_dims(m_resized, axis=-1) # (H, W, 1)
        
        # Blending context (original colors) and target (red overlay)
        # Context where mask is 1, target where mask is 0
        mask_3ch = np.repeat(m_resized, 3, axis=-1) # (H, W, 3)
        
        # Red translucent overlay on masked regions (where mask_3ch == 0)
        target_overlay = (frames[t] * 0.4 + overlay_color * 0.6).astype(np.uint8)
        overlayed[t] = (frames[t] * mask_3ch + target_overlay * (1 - mask_3ch)).astype(np.uint8)
        
    return overlayed

def compute_metrics(motion_map, masks_pred, height, width):
    # motion_map: (duration, height, width)
    # masks_pred: 1D tensor/array of flat predictor indices
    N = motion_map.numel()
    motion_flat = motion_map.flatten()
    
    # Find top 20% high-motion voxels
    k = int(N * 0.20)
    top_indices = torch.argsort(motion_flat, descending=True)[:k].cpu().numpy()
    
    # Convert masks_pred to numpy set for quick lookup
    pred_set = set(masks_pred.cpu().numpy())
    
    # 1. Motion Overlap Ratio: fraction of top 20% motion voxels that are targeted (masked)
    overlap_count = sum([1 for idx in top_indices if idx in pred_set])
    overlap_ratio = overlap_count / k if k > 0 else 0.0
    
    # 2. Spatial Saliency / Entropy: measures distribution of target tokens
    # Compute histogram of target tokens across the spatial grid (flattening temporal dimension)
    spatial_hits = np.zeros(height * width)
    for idx in pred_set:
        spatial_idx = idx % (height * width)
        spatial_hits[spatial_idx] += 1
        
    if len(pred_set) > 0:
        spatial_probs = spatial_hits / len(pred_set)
        # Filter out zeros for entropy calculation
        spatial_probs = spatial_probs[spatial_probs > 0]
        spatial_entropy = -np.sum(spatial_probs * np.log2(spatial_probs))
        # Max entropy for height * width locations
        max_entropy = np.log2(height * width)
        norm_entropy = spatial_entropy / max_entropy if max_entropy > 0 else 0.0
    else:
        norm_entropy = 0.0
        
    return overlap_ratio, norm_entropy

def write_video_file(output_path, frames, fps=15):
    # frames: (T, H, W, C)
    T, H, W, C = frames.shape
    # OpenCV expects BGR
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    for i in range(T):
        bgr_frame = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
    out.release()
    print(f"Saved video to: {output_path}")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert num_frames arg to int
    num_frames = int(args.num_frames)
    
    # 1. Load video and prepare input tensor
    video_frames = load_video(args.video_path, num_frames=num_frames, crop_size=args.crop_size)
    print(f"Cropped video shape: {video_frames.shape} (T x H x W x C)")
    
    # Convert to torch tensor of shape (B, C, T, H_crop, W_crop) matching MaskCollator expectation
    video_tensor = torch.from_numpy(video_frames).permute(3, 0, 1, 2).unsqueeze(0).float() # (1, C, T, H, W)
    
    # 2. Initialize our generators
    configs = {
        "Random Multi-block": {
            "motion_mode": "random",
            "spatial_scale": (0.15, 0.15),
            "temporal_scale": (1.0, 1.0),
            "aspect_ratio": (0.75, 1.5),
            "num_blocks": 8
        },
        "Token High-Motion as Context": {
            "motion_mode": "motion_context",
            "motion_keep_ratio": 0.25,
            "motion_pred_ratio": 0.75
        },
        "Token High-Motion as Target": {
            "motion_mode": "motion_target",
            "motion_keep_ratio": 0.25,
            "motion_pred_ratio": 0.75
        },
        "Block High-Motion as Context": {
            "motion_mode": "block_motion_context",
            "spatial_scale": (0.15, 0.15),
            "temporal_scale": (1.0, 1.0),
            "aspect_ratio": (0.75, 1.5),
            "num_blocks": 8,
            "motion_temperature": args.temp
        },
        "Block High-Motion as Target": {
            "motion_mode": "block_motion_target",
            "spatial_scale": (0.15, 0.15),
            "temporal_scale": (1.0, 1.0),
            "aspect_ratio": (0.75, 1.5),
            "num_blocks": 8,
            "motion_temperature": args.temp
        }
    }
    
    results = {}
    metrics_overlap = []
    metrics_entropy = []
    labels = []
    
    # First, run a default random mask generator to compute motion map
    # We create a dummy generator to access compute_motion_map function
    dummy_generator = MotionMaskGenerator(
        crop_size=(args.crop_size, args.crop_size),
        num_frames=num_frames,
        spatial_patch_size=(args.patch_size, args.patch_size),
        temporal_patch_size=args.tubelet_size
    )
    motion_map = dummy_generator.compute_motion_map(video_tensor)[0] # shape (duration, height, width)
    
    # Generate motion heatmap video frames
    motion_heatmap_frames = generate_motion_heatmap(motion_map, video_frames.shape)
    write_video_file(os.path.join(args.output_dir, "motion_heatmap.mp4"), motion_heatmap_frames)
    
    print("\n--- Generating Masks and Applying Overlays ---")
    for name, cfg in configs.items():
        print(f"\nRunning generator config: {name}")
        generator = MotionMaskGenerator(
            crop_size=(args.crop_size, args.crop_size),
            num_frames=num_frames,
            spatial_patch_size=(args.patch_size, args.patch_size),
            temporal_patch_size=args.tubelet_size,
            spatial_pred_mask_scale=cfg.get("spatial_scale", (0.2, 0.8)),
            temporal_pred_mask_scale=cfg.get("temporal_scale", (1.0, 1.0)),
            aspect_ratio=cfg.get("aspect_ratio", (0.3, 3.0)),
            npred=cfg.get("num_blocks", 1),
            motion_mode=cfg["motion_mode"],
            motion_keep_ratio=cfg.get("motion_keep_ratio", None),
            motion_pred_ratio=cfg.get("motion_pred_ratio", None),
            motion_temperature=cfg.get("motion_temperature", 0.05)
        )
        
        # Run generator to get masks
        masks_enc, masks_pred = generator(batch_size=1, video_tensor_batch=video_tensor)
        
        # Remove batch dim to process
        m_enc = masks_enc[0]
        m_pred = masks_pred[0]
        
        print(f"  Context (encoder) tokens: {len(m_enc)}")
        print(f"  Target (predictor) tokens: {len(m_pred)}")
        
        # Apply mask overlay
        overlayed_frames = apply_mask_overlay(video_frames, m_pred, args.patch_size, args.tubelet_size)
        
        # Write individual overlay video
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        write_video_file(os.path.join(args.output_dir, f"masking_{safe_name}.mp4"), overlayed_frames)
        results[name] = overlayed_frames
        
        # Compute quantitative metrics
        overlap_ratio, norm_entropy = compute_metrics(
            motion_map, m_pred, generator.height, generator.width
        )
        metrics_overlap.append(overlap_ratio)
        metrics_entropy.append(norm_entropy)
        labels.append(name.replace(" ", "\n"))
        print(f"  Motion Overlap Ratio (target hits high motion): {overlap_ratio:.4f}")
        print(f"  Normalized Spatial Entropy (target dispersion): {norm_entropy:.4f}")

    # 3. Create composite comparison grid video (2 rows, 3 columns)
    # Row 1: [Original, Motion Heatmap, Random Multi-block]
    # Row 2: [Token High-Motion as Target, Block High-Motion as Context, Block High-Motion as Target]
    print("\nCreating side-by-side comparison video...")
    H, W = args.crop_size, args.crop_size
    grid_frames = np.zeros((num_frames, H * 2, W * 3, 3), dtype=np.uint8)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (255, 255, 255)
    thickness = 2
    
    for t in range(num_frames):
        # Sub-frames
        orig = video_frames[t].copy()
        heatmap = motion_heatmap_frames[t].copy()
        rand_mask = results["Random Multi-block"][t].copy()
        token_target = results["Token High-Motion as Target"][t].copy()
        block_context = results["Block High-Motion as Context"][t].copy()
        block_target = results["Block High-Motion as Target"][t].copy()
        
        # Add text labels on top-left of each frame (OpenCV works on BGR, but we write RGB frames. Add label in RGB)
        # So we draw text in RGB. cv2.putText handles it if the color matches the color format of the image.
        cv2.putText(orig, "Original Video", (10, 25), font, font_scale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(heatmap, "Motion Heatmap", (10, 25), font, font_scale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(rand_mask, "Random Multi-block", (10, 25), font, font_scale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(token_target, "Token High-Motion Target", (10, 25), font, font_scale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(block_context, "Block Motion Context", (10, 25), font, font_scale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(block_target, "Block Motion Target", (10, 25), font, font_scale, font_color, thickness, cv2.LINE_AA)
        
        # Assemble grid
        grid_frames[t, 0:H, 0:W] = orig
        grid_frames[t, 0:H, W:W*2] = heatmap
        grid_frames[t, 0:H, W*2:W*3] = rand_mask
        grid_frames[t, H:H*2, 0:W] = token_target
        grid_frames[t, H:H*2, W:W*2] = block_context
        grid_frames[t, H:H*2, W*2:W*3] = block_target
        
    write_video_file(os.path.join(args.output_dir, "masking_comparison.mp4"), grid_frames)
    
    # 4. Plot quantitative metrics
    print("\nPlotting masking strategy metrics...")
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Bar plot for Motion Overlap Ratio
    color = "tab:red"
    ax1.set_ylabel("Motion Overlap Ratio (Target / High Motion)", color=color, fontweight="bold")
    rects1 = ax1.bar(x - width/2, metrics_overlap, width, label="Motion Overlap Ratio", color=color, alpha=0.85)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(0, 1.05)
    
    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Normalized Spatial Entropy (Dispersion)", color=color, fontweight="bold")
    rects2 = ax2.bar(x + width/2, metrics_entropy, width, label="Spatial Entropy", color=color, alpha=0.85)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(0, 1.05)
    
    # Add titles and legends
    plt.title("Quantitative Comparison of V-JEPA Masking Strategies", fontsize=14, fontweight="bold", pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9, fontweight="bold")
    
    fig.tight_layout()
    
    # Save quantitative metrics plot
    plot_path = os.path.join(args.output_dir, "masking_metrics.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved metrics comparison plot to: {plot_path}")
    
    # 5. Save static keyframe grid for documentation (Keyframe 20, 40)
    print("\nCreating static keyframe grid...")
    keyframe_indices = [20, 40]
    fig, axes = plt.subplots(2, len(configs) + 2, figsize=(16, 7))
    
    for row_idx, kf_t in enumerate(keyframe_indices):
        # Column 0: Original
        axes[row_idx, 0].imshow(video_frames[kf_t])
        axes[row_idx, 0].set_title("Original" if row_idx == 0 else f"Frame {kf_t}")
        axes[row_idx, 0].axis("off")
        
        # Column 1: Heatmap
        axes[row_idx, 1].imshow(motion_heatmap_frames[kf_t])
        axes[row_idx, 1].set_title("Motion Heatmap" if row_idx == 0 else f"Frame {kf_t}")
        axes[row_idx, 1].axis("off")
        
        # Columns 2+: Mask configs
        for col_idx, (name, overlayed) in enumerate(results.items()):
            axes[row_idx, col_idx + 2].imshow(overlayed[kf_t])
            axes[row_idx, col_idx + 2].set_title(name.replace(" ", "\n") if row_idx == 0 else f"Frame {kf_t}", fontsize=8)
            axes[row_idx, col_idx + 2].axis("off")
            
    plt.suptitle("Static Keyframe Comparison across Masking Strategies", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    grid_img_path = os.path.join(args.output_dir, "keyframe_grid.png")
    plt.savefig(grid_img_path, dpi=300)
    plt.close()
    print(f"Saved keyframe comparison grid to: {grid_img_path}")
    
    print("\nVisualization generation complete! Check the output directory:", args.output_dir)

if __name__ == "__main__":
    main()
