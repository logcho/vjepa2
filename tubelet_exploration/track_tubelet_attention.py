#!/usr/bin/env python3
import os
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from decord import VideoReader
from transformers import AutoModel, AutoVideoProcessor

SAMPLE_VIDEO_URL = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/bowling/-WH-lxmGJVY_000005_000015.mp4"
SAMPLE_VIDEO_PATH = "sample_video.mp4"
HF_MODEL_NAME = "facebook/vjepa2-vitl-fpc64-256"
OUTPUT_DIR = "tubelet_exploration"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

def download_sample_video(url, path):
    if not os.path.exists(path):
        print(f"Downloading sample video from: {url}")
        urllib.request.urlretrieve(url, path)
        print(f"Downloaded sample video to: {path}")

def load_video_frames(video_path, num_frames=16, stride=4):
    vr = VideoReader(video_path)
    total_frames = len(vr)
    max_index = min(total_frames, num_frames * stride)
    frame_idx = np.arange(0, max_index, stride)
    
    if len(frame_idx) < num_frames:
        frame_idx = np.pad(frame_idx, (0, num_frames - len(frame_idx)), mode="edge")
        
    frame_idx = frame_idx[:num_frames]
    video_data = vr.get_batch(frame_idx).asnumpy()
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

def track_attention():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    download_sample_video(SAMPLE_VIDEO_URL, SAMPLE_VIDEO_PATH)
    
    # 1. Load and process video
    raw_frames = load_video_frames(SAMPLE_VIDEO_PATH, num_frames=16, stride=4)
    cropped_frames = resize_and_center_crop(raw_frames, target_size=(256, 256))
    
    # 2. Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # 3. Load model and processor
    print("Loading model and processor...")
    processor = AutoVideoProcessor.from_pretrained(HF_MODEL_NAME)
    model = AutoModel.from_pretrained(HF_MODEL_NAME, attn_implementation="eager").to(device)
    model.eval()
    
    video_tensor = torch.from_numpy(cropped_frames).permute(0, 3, 1, 2)
    inputs = processor(video_tensor, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Hooks to capture attentions
    attention_maps = {}
    
    def get_attention_hook(layer_idx):
        def hook(module, input, output):
            attention_maps[layer_idx] = output[1].detach().cpu()
        return hook

    for idx, layer in enumerate(model.encoder.layer):
        layer.attention.register_forward_hook(get_attention_hook(idx))

    print("Running model inference to capture attention patterns...")
    with torch.no_grad():
        outputs = model(**inputs)
        
    print("Inference completed successfully!")
    
    # 4. Token parameters
    tubelet_size = model.config.tubelet_size
    patch_size = model.config.patch_size
    pv = inputs["pixel_values_videos"]
    B, T_raw, C, H_raw, W_raw = pv.shape
    
    T_p = T_raw // tubelet_size
    H_p = H_raw // patch_size
    W_p = W_raw // patch_size
    N_tokens = T_p * H_p * W_p
    
    # 5. Programmatically locate the high-motion query tubelet
    # Compute absolute frame difference
    video_tensor_float = video_tensor.float()
    diff = torch.abs(video_tensor_float[1:] - video_tensor_float[:-1]).mean(dim=1)
    diff = torch.cat([diff, diff[-1:]], dim=0)
    
    motion_tokens = F.avg_pool3d(
        diff.unsqueeze(0).unsqueeze(0),
        kernel_size=(tubelet_size, patch_size, patch_size),
        stride=(tubelet_size, patch_size, patch_size)
    ).squeeze(0).squeeze(0)  # [T_p, H_p, W_p]
    
    # Find indices of maximum motion
    max_idx_flat = torch.argmax(motion_tokens.flatten()).item()
    q_t = max_idx_flat // (H_p * W_p)
    q_y = (max_idx_flat % (H_p * W_p)) // W_p
    q_x = max_idx_flat % W_p
    
    print(f"\nQuery Token Selected (High-Motion Area):")
    print(f"  Flat Index: {max_idx_flat}")
    print(f"  Coordinates: t={q_t}, y={q_y}, x={q_x}")
    print(f"  Pixel bounding box: frame={q_t*2}, y=[{q_y*16}:{q_y*16+16}], x=[{q_x*16}:{q_x*16+16}]")
    
    # 6. Extract query attention heatmap at layers 2, 8, 14, 20
    target_layers = [2, 8, 14, 20]
    
    # Plotting grid: Columns = Layer depth, Rows = Time steps
    # We will display 4 keyframes (at time steps t=0, 2, 4, 6)
    time_steps = [0, 2, 4, 6]  # tubelet time steps (mapping to raw frames 0, 4, 8, 12)
    
    fig, axes = plt.subplots(len(time_steps), len(target_layers) + 1, figsize=(16, 12))
    
    for row_idx, t_step in enumerate(time_steps):
        raw_frame_idx = t_step * tubelet_size
        frame_rgb = cropped_frames[raw_frame_idx].copy()
        
        # Draw green bounding box around query tubelet on its active time step
        if t_step == q_t:
            cv2.rectangle(frame_rgb, (q_x*patch_size, q_y*patch_size), 
                          (q_x*patch_size+patch_size, q_y*patch_size+patch_size), 
                          (0, 255, 0), 2)
            cv2.putText(frame_rgb, "Query", (q_x*patch_size, q_y*patch_size - 4), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
            
        # Left-most column: Original keyframe
        axes[row_idx, 0].imshow(frame_rgb)
        axes[row_idx, 0].axis('off')
        if row_idx == 0:
            axes[row_idx, 0].set_title("Original\n(Query Green)", fontsize=11, fontweight='semibold')
        axes[row_idx, 0].set_ylabel(f"Frame {raw_frame_idx}", fontsize=11, fontweight='bold')
        
        for col_idx, layer_idx in enumerate(target_layers):
            attn_layer = attention_maps[layer_idx].squeeze(0)  # [num_heads, N_tokens, N_tokens]
            # Average attention over all heads
            attn_avg = attn_layer.mean(dim=0)  # [N_tokens, N_tokens]
            
            # Extract attention of query token over all key tokens
            # query token index is max_idx_flat
            q_attn = attn_avg[max_idx_flat]  # [N_tokens]
            
            # Reshape to 3D grid
            q_attn_3d = q_attn.view(T_p, H_p, W_p)
            
            # Extract the 2D spatial attention map for the current time step
            attn_map_2d = q_attn_3d[t_step].numpy()
            
            # Upsample attention map to match patch size (16x16) using nearest neighbor
            attn_upsampled = cv2.resize(attn_map_2d, (256, 256), interpolation=cv2.INTER_NEAREST)
            
            # Normalize to 0-255 for colormap overlay
            attn_norm = (attn_upsampled - attn_upsampled.min()) / (attn_upsampled.max() - attn_upsampled.min() + 1e-8)
            heatmap = cv2.applyColorMap((attn_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Overlay heatmap on original frame
            overlay = cv2.addWeighted(cropped_frames[raw_frame_idx], 0.6, heatmap, 0.4, 0)
            
            # Draw green bounding box around query on overlay if active time step
            if t_step == q_t:
                cv2.rectangle(overlay, (q_x*patch_size, q_y*patch_size), 
                              (q_x*patch_size+patch_size, q_y*patch_size+patch_size), 
                              (0, 255, 0), 2)
            
            ax = axes[row_idx, col_idx + 1]
            ax.imshow(overlay)
            ax.axis('off')
            if row_idx == 0:
                ax.set_title(f"Layer {layer_idx}\nAttention Heatmap", fontsize=11, fontweight='semibold')
                
    plt.suptitle("Spatio-Temporal Attention Tracking from Query Tubelet\n(V-JEPA 2.1 Layer-wise Self-Attention Field Visualized)", 
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "query_attention_tracking.png"), dpi=150)
    plt.close()
    
    print(f"Saved attention tracking visualization to: {os.path.join(PLOTS_DIR, 'query_attention_tracking.png')}")

if __name__ == "__main__":
    track_attention()
