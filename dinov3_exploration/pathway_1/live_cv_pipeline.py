#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from decord import VideoReader

# Add workspace search paths
sys.path.append("/Users/loganchoi/Desktop/vjepa2/vjepa2")
sys.path.append("/Users/loganchoi/Desktop/dinov3/dinov3")

from src.models.predictor import VisionTransformerPredictor
from src.masks.utils import apply_masks

# Configuration
VIDEO_PATH = "/Users/loganchoi/Desktop/vjepa2/sample_video.mp4"
MODEL_PATH = "/Users/loganchoi/Desktop/vjepa2/dinov3_exploration/pathway_1/predictor_pathway1.pth"
OUTPUT_VIDEO_PATH = "/Users/loganchoi/Desktop/vjepa2/dinov3_exploration/pathway_1/pipeline_visualization.mp4"
NUM_FRAMES = 8
STRIDE = 2
GRID_SIZE = 28
PATCH_SIZE = 16
EMBED_DIM = 384

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
    Generate masks for predicting the center 6x6 patches of the last 2 frames (frames 6 and 7).
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

def main():
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
    
    # 3. Read video to fit PCA on the entire feature space for stable visualization color mapping
    print("Fitting PCA for semantic feature visualization...")
    vr = VideoReader(VIDEO_PATH)
    total_video_frames = len(vr)
    
    # Extract features for a subset of frames to fit PCA
    pca_frames = []
    for i in range(0, min(total_video_frames, 64), 2):
        _, norm_t = preprocess_frame(vr[i].asnumpy())
        pca_frames.append(norm_t)
    pca_frames = torch.stack(pca_frames).to(device)
    
    with torch.no_grad():
        pca_outputs = dinov3_model.forward_features(pca_frames)
        pca_features = pca_outputs["x_norm_patchtokens"].cpu().view(-1, EMBED_DIM).numpy()
        
    pca = PCA(n_components=3)
    pca.fit(pca_features)
    print("PCA fit successfully.")
    
    # 4. Set up Video Writer
    # We will stitch original, ground truth, and predicted frames side-by-side: 224 * 3 = 672 width, 224 height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 10.0, (672, 224))
    
    # Generate constant masks
    masks_enc, masks_pred = generate_static_masks()
    masks_enc_dev = masks_enc.to(device)
    masks_pred_dev = masks_pred.to(device)
    
    # Process video frame-by-frame with a sliding window
    display_buffer = []
    tensor_buffer = []
    
    print("\nProcessing video stream through live prediction pipeline...")
    for frame_idx in range(total_video_frames):
        raw_frame = vr[frame_idx].asnumpy()
        disp_f, norm_t = preprocess_frame(raw_frame)
        
        display_buffer.append(disp_f)
        tensor_buffer.append(norm_t)
        
        if len(display_buffer) < NUM_FRAMES:
            continue
            
        # We have a full buffer of 8 frames
        # Extract features for the buffer
        stacked_tensors = torch.stack(tensor_buffer) # [8, 3, 224, 224]
        features = extract_dinov3_features(dinov3_model, stacked_tensors, device) # [8, 196, 384]
        
        # Flatten spatiotemporal features: [1, 8 * 196, 384]
        features_flat = features.view(1, NUM_FRAMES * (GRID_SIZE * GRID_SIZE), EMBED_DIM).to(device)
        
        # Apply masks
        x_context = apply_masks(features_flat, [masks_enc_dev]) # [1, N_enc, 384]
        
        # Run predictor to predict target tokens
        with torch.no_grad():
            y_pred = predictor(x_context, masks_enc_dev, masks_pred_dev) # [1, N_pred, 384]
            
        # Reconstruct the representation sequence
        # We start with the ground-truth features, and replace the masked target indices with predictions
        reconstructed = features_flat.clone() # [1, 8 * 196, 384]
        
        # We need to assign predicted features to masks_pred indices
        # masks_pred has shape [1, N_pred]
        indices = masks_pred_dev.squeeze(0).unsqueeze(-1).expand(-1, EMBED_DIM) # [N_pred, 384]
        reconstructed.squeeze(0).scatter_(0, indices, y_pred.squeeze(0))
        
        # Reshape back to sequence
        features_gt = features_flat.view(NUM_FRAMES, GRID_SIZE, GRID_SIZE, EMBED_DIM).cpu()
        features_recon = reconstructed.view(NUM_FRAMES, GRID_SIZE, GRID_SIZE, EMBED_DIM).cpu()
        
        # Visualize the target frame (last frame of the sequence, i.e., index 7)
        # Original Frame 7 (BGR for OpenCV)
        frame_original = display_buffer[7].copy()
        frame_original = cv2.cvtColor(frame_original, cv2.COLOR_RGB2BGR)
        
        # Draw a yellow box around the masked region (rows 8-19, cols 8-19)
        # Coordinate calculation: 8*16 = 128 to 20*16 = 320
        cv2.rectangle(frame_original, (128, 128), (320, 320), (0, 255, 255), 4)
        cv2.putText(frame_original, "Input Video (Masked Box)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Resize original frame to 224x224 to match feature maps panel size
        frame_original_resized = cv2.resize(frame_original, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        # Visualize Ground Truth Features for frame 7
        gt_feat_f7 = features_gt[7] # [28, 28, 384]
        gt_visual = visualize_features(gt_feat_f7, pca)
        cv2.putText(gt_visual, "Ground Truth features", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Visualize Reconstructed Features (with predicted center region)
        recon_feat_f7 = features_recon[7] # [28, 28, 384]
        recon_visual = visualize_features(recon_feat_f7, pca)
        cv2.putText(recon_visual, "Predictor Reconstruction", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Stitch them side-by-side
        stitched = np.hstack([frame_original_resized, gt_visual, recon_visual])
        out_writer.write(stitched)
        
        # Slide buffer
        display_buffer.pop(0)
        tensor_buffer.pop(0)
        
    out_writer.release()
    print(f"\nLive pipeline processing complete! Output saved to: {OUTPUT_VIDEO_PATH}")

def visualize_features(feat, pca):
    h, w, d = feat.shape
    feat_flat = feat.reshape(-1, d).numpy()
    
    # Project to 3 channels using pre-fit PCA
    proj = pca.transform(feat_flat)
    
    # Normalize channels to [0, 255]
    proj_min = proj.min(axis=0, keepdims=True)
    proj_max = proj.max(axis=0, keepdims=True)
    proj_norm = (proj - proj_min) / (proj_max - proj_min + 1e-8)
    
    img = (proj_norm * 255.0).astype(np.uint8)
    img = img.reshape(h, w, 3)
    
    # Convert RGB projection to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Upsample using nearest neighbors for grid pixel block visibility
    img_upsampled = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_NEAREST)
    return img_upsampled

if __name__ == "__main__":
    main()
