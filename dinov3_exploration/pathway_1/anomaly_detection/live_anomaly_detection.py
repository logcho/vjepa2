#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from decord import VideoReader

# Add workspace search paths
sys.path.append("/Users/loganchoi/Desktop/vjepa2/vjepa2")
sys.path.append("/Users/loganchoi/Desktop/dinov3/dinov3")

from src.models.predictor import VisionTransformerPredictor
from src.masks.utils import apply_masks

# Configuration
VIDEO_PATH = "/Users/loganchoi/Desktop/vjepa2/sample_video.mp4"
MODEL_PATH = "/Users/loganchoi/Desktop/vjepa2/dinov3_exploration/pathway_1/predictor_pathway1.pth"
OUTPUT_VIDEO_PATH = "/Users/loganchoi/Desktop/vjepa2/dinov3_exploration/pathway_1/anomaly_detection/anomaly_output.mp4"
NUM_FRAMES = 8
STRIDE = 2
GRID_SIZE = 28
PATCH_SIZE = 16
EMBED_DIM = 384

# Anomaly Detection Config
WINDOW_SIZE = 30
THRESHOLD_K = 2.5

def preprocess_frame(frame, target_size=(448, 448)):
    # frame: H x W x C (numpy uint8)
    H, W, C = frame.shape
    shortest_edge = int(max(256, max(target_size) * 1.15))
    if H < W:
        new_h = shortest_edge
        new_w = int(W * (shortest_edge / H))
    else:
        new_w = shortest_edge
        new_h = int(H * (shortest_edge / W))
        
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    th, tw = target_size
    start_y = (new_h - th) // 2
    start_x = (new_w - tw) // 2
    cropped = resized[start_y:start_y+th, start_x:start_x+tw, :]
    
    # Preprocessed display frame (for OpenCV visualization)
    display_frame = cropped.copy()
    
    # Normalize for PyTorch DINOv3 model
    tensor = torch.from_numpy(cropped).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    normalized = (tensor - mean) / std
    return display_frame, normalized

def extract_dinov3_features(model, stacked_tensors, device):
    stacked_tensors = stacked_tensors.to(device)
    with torch.no_grad():
        outputs = model.forward_features(stacked_tensors)
        patch_tokens = outputs["x_norm_patchtokens"] # [8, 196, 384]
    return patch_tokens.cpu()

def generate_static_masks():
    """
    Generate masks for predicting the center 12x12 patches of the last 2 frames (frames 6 and 7).
    """
    total_tokens = NUM_FRAMES * GRID_SIZE * GRID_SIZE
    mask_pred = []
    
    # We only mask the last two frames (indices 6 and 7)
    masked_frames = [6, 7]
    # Center 12x12 patches out of 28x28 grid
    start_patch, end_patch = 8, 20
    
    for t in range(NUM_FRAMES):
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                token_idx = t * (GRID_SIZE * GRID_SIZE) + r * GRID_SIZE + c
                if t in masked_frames and (start_patch <= r < end_patch) and (start_patch <= c < end_patch):
                    mask_pred.append(token_idx)
                    
    mask_pred = torch.tensor(sorted(mask_pred), dtype=torch.long)
    mask_enc = torch.tensor(sorted(list(set(range(total_tokens)) - set(mask_pred.tolist()))), dtype=torch.long)
    
    # Reshape to [1, K] for batch dimension
    return mask_enc.unsqueeze(0), mask_pred.unsqueeze(0)

def draw_error_graph(errors_list, is_anomaly, threshold, width=448, height=448):
    graph = np.ones((height, width, 3), dtype=np.uint8) * 240 # Light gray background
    
    # Draw title
    cv2.putText(graph, "Live Prediction Error (MSE)", (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    if len(errors_list) < 2:
        return graph
        
    # Scale errors
    max_err = max(max(errors_list), threshold) if threshold is not None else max(errors_list)
    max_err = max(max_err, 0.001) # Avoid division by zero
    
    # Draw axes
    cv2.line(graph, (40, height - 40), (width - 20, height - 40), (0, 0, 0), 2)
    cv2.line(graph, (40, 60), (40, height - 40), (0, 0, 0), 2)
    
    # Draw points
    num_points = len(errors_list)
    step_x = (width - 60) / (num_points - 1)
    
    points = []
    for i, err in enumerate(errors_list):
        x = int(40 + i * step_x)
        y = int(height - 40 - (err / max_err) * (height - 100))
        points.append((x, y))
        
    for i in range(len(points) - 1):
        cv2.line(graph, points[i], points[i+1], (255, 0, 0), 2)
        
    # Draw threshold line
    if threshold is not None:
        thresh_y = int(height - 40 - (threshold / max_err) * (height - 100))
        cv2.line(graph, (40, thresh_y), (width - 20, thresh_y), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(graph, f"Threshold: {threshold:.4f}", (50, max(70, thresh_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
    # Current value text
    curr_err = errors_list[-1]
    cv2.putText(graph, f"Current MSE: {curr_err:.4f}", (40, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    if is_anomaly:
        cv2.putText(graph, "ANOMALY!", (width - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        
    return graph

def main():
    os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # 1. Load DINOv3 model
    print("Loading pretrained DINOv3 model...")
    dinov3_model = torch.hub.load(
        '/Users/loganchoi/Desktop/dinov3/dinov3', 
        'dinov3_vits16', 
        source='local', 
        pretrained=True
    ).to(device)
    dinov3_model.eval()
    
    # 2. Load trained V-JEPA predictor
    print("Loading V-JEPA predictor checkpoint...")
    predictor = VisionTransformerPredictor(
        img_size=(448, 448),
        patch_size=16,
        num_frames=NUM_FRAMES,
        tubelet_size=1,
        embed_dim=384,
        predictor_embed_dim=192,
        out_embed_dim=384,
        depth=4,
        num_heads=6,
        use_mask_tokens=True,
        num_mask_tokens=1
    ).to(device)
    predictor.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    predictor.eval()
    
    # 3. Read video
    print("Opening video file...")
    vr = VideoReader(VIDEO_PATH)
    total_video_frames = len(vr)
    
    # 4. Set up Video Writer
    # We will stitch original frame and error graph side-by-side: 448 * 2 = 896 width, 448 height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 10.0, (896, 448))
    
    # Generate constant masks
    masks_enc, masks_pred = generate_static_masks()
    masks_enc_dev = masks_enc.to(device)
    masks_pred_dev = masks_pred.to(device)
    
    # Process video frame-by-frame with a sliding window
    display_buffer = []
    tensor_buffer = []
    error_history = deque(maxlen=WINDOW_SIZE)
    full_error_history = []
    
    print("\nProcessing video stream through live anomaly detection pipeline...")
    for frame_idx in range(total_video_frames):
        raw_frame = vr[frame_idx].asnumpy()
        disp_f, norm_t = preprocess_frame(raw_frame)
        
        display_buffer.append(disp_f)
        tensor_buffer.append(norm_t)
        
        if len(display_buffer) < NUM_FRAMES:
            continue
            
        # We have a full buffer of 8 frames
        stacked_tensors = torch.stack(tensor_buffer) # [8, 3, 448, 448]
        features = extract_dinov3_features(dinov3_model, stacked_tensors, device) # [8, 784, 384]
        
        # Flatten spatiotemporal features: [1, 8 * 784, 384]
        features_flat = features.view(1, NUM_FRAMES * (GRID_SIZE * GRID_SIZE), EMBED_DIM).to(device)
        
        # Extract ground truth target features corresponding to the masked region
        indices = masks_pred_dev.squeeze(0).unsqueeze(-1).expand(-1, EMBED_DIM) # [N_pred, 384]
        y_target = torch.gather(features_flat.squeeze(0), 0, indices).unsqueeze(0) # [1, N_pred, 384]
        
        # Apply masks to get context
        x_context = apply_masks(features_flat, [masks_enc_dev]) # [1, N_enc, 384]
        
        # Run predictor to predict target tokens
        with torch.no_grad():
            y_pred = predictor(x_context, masks_enc_dev, masks_pred_dev) # [1, N_pred, 384]
            
        # Compute MSE loss between prediction and ground truth
        mse_error = F.mse_loss(y_pred, y_target).item()
        
        # Anomaly Detection Logic
        is_anomaly = False
        threshold = None
        
        if len(error_history) == error_history.maxlen:
            mean_err = np.mean(error_history)
            std_err = np.std(error_history)
            threshold = mean_err + THRESHOLD_K * std_err
            
            if mse_error > threshold:
                is_anomaly = True
                
        error_history.append(mse_error)
        full_error_history.append(mse_error)
        
        # Visualize
        frame_original = display_buffer[7].copy()
        frame_original = cv2.cvtColor(frame_original, cv2.COLOR_RGB2BGR)
        
        # Draw a yellow box around the masked region (rows 8-19, cols 8-19)
        cv2.rectangle(frame_original, (128, 128), (320, 320), (0, 255, 255), 4)
        cv2.putText(frame_original, "Input Video (Masked Box)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if is_anomaly:
            cv2.rectangle(frame_original, (10, 10), (438, 438), (0, 0, 255), 8)
            cv2.putText(frame_original, "ANOMALY!", (150, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
            
        # Draw Error Graph
        # We pass up to the last 100 errors for plotting history
        graph = draw_error_graph(full_error_history[-100:], is_anomaly, threshold)
        
        # Stitch them side-by-side
        stitched = np.hstack([frame_original, graph])
        out_writer.write(stitched)
        
        # Slide buffer
        display_buffer.pop(0)
        tensor_buffer.pop(0)
        
        if (frame_idx + 1) % 10 == 0:
            print(f"Processed {frame_idx + 1}/{total_video_frames} frames. Current MSE: {mse_error:.5f}")
            
    out_writer.release()
    print(f"\nLive anomaly detection complete! Output saved to: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()
