#!/usr/bin/env python3
import os
import sys
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
from decord import VideoReader

# Add workspace search paths
sys.path.append("/Users/loganchoi/Desktop/vjepa2/vjepa2")
sys.path.append("/Users/loganchoi/Desktop/dinov3/dinov3")

from src.models.predictor import VisionTransformerPredictor
from src.masks.utils import apply_masks
from train_decoder import FeatureDecoder, load_and_preprocess_video, extract_dinov3_features

# Configuration
VIDEO_PATH = "/Users/loganchoi/Desktop/vjepa2/sample_video.mp4"
OUTPUT_DIR = "/Users/loganchoi/Desktop/vjepa2/dinov3_exploration/pathway_1/inpainting_decoding"
PREDICTOR_WEIGHTS = "/Users/loganchoi/Desktop/vjepa2/dinov3_exploration/pathway_1/predictor_pathway1.pth"
DECODER_WEIGHTS = os.path.join(OUTPUT_DIR, "decoder.pth")
NUM_FRAMES = 8

def get_right_half_mask(batch_size, num_frames=8, grid_size=28):
    """
    Creates a deterministic mask covering the right half of the frames.
    Returns masks_enc (left half tokens) and masks_pred (right half tokens).
    """
    N = grid_size * grid_size
    tokens_per_frame_enc = []
    tokens_per_frame_pred = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            token_idx = i * grid_size + j
            if j < grid_size // 2:
                tokens_per_frame_enc.append(token_idx)
            else:
                tokens_per_frame_pred.append(token_idx)
                
    # Expand to all frames
    masks_enc = []
    masks_pred = []
    for t in range(num_frames):
        offset = t * N
        masks_enc.extend([idx + offset for idx in tokens_per_frame_enc])
        masks_pred.extend([idx + offset for idx in tokens_per_frame_pred])
        
    masks_enc = torch.tensor(masks_enc, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    masks_pred = torch.tensor(masks_pred, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    return masks_enc, masks_pred

def denormalize(tensor):
    """ Convert ImageNet normalized tensor back to [0, 1] RGB """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)

def inpaint_video():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    print("Loading models...")
    # 1. DINOv3 Backbone
    dinov3_model = torch.hub.load(
        '/Users/loganchoi/Desktop/dinov3/dinov3', 
        'dinov3_vits16', 
        source='local', 
        pretrained=True
    ).to(device)
    dinov3_model.eval()
    
    # 2. V-JEPA Predictor
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
    predictor.load_state_dict(torch.load(PREDICTOR_WEIGHTS, map_location=device))
    predictor.eval()
    
    # 3. Decoder
    decoder = FeatureDecoder(embed_dim=384).to(device)
    decoder.load_state_dict(torch.load(DECODER_WEIGHTS, map_location=device))
    decoder.eval()
    
    print("Loading video...")
    normalized_frames, raw_frames = load_and_preprocess_video(VIDEO_PATH, num_frames=NUM_FRAMES, target_size=(448, 448))
    # We will process 1 sequence of 8 frames
    video_tensor = normalized_frames[:NUM_FRAMES].unsqueeze(0) # [1, 8, 3, 448, 448]
    
    print("Extracting features...")
    # [T, N, D] -> [1, T, N, D]
    features = extract_dinov3_features(dinov3_model, video_tensor.squeeze(0), device).unsqueeze(0)
    B, T, N, D = features.shape
    features_flat = features.view(B, T * N, D).to(device)
    
    print("Predicting masked regions...")
    masks_enc, masks_pred = get_right_half_mask(B, num_frames=T, grid_size=28)
    masks_enc = masks_enc.to(device)
    masks_pred = masks_pred.to(device)
    
    with torch.no_grad():
        x_context = apply_masks(features_flat, [masks_enc])
        # Use predictor to guess the masked right half
        y_pred = predictor(x_context, masks_enc, masks_pred)
        
        # Reconstruct full feature map
        reconstructed_features = torch.zeros_like(features_flat)
        # scatter expects src to match index shape, so expand masks_enc
        idx_enc = masks_enc.unsqueeze(-1).expand(-1, -1, D)
        reconstructed_features.scatter_(1, idx_enc, x_context)
        
        idx_pred = masks_pred.unsqueeze(-1).expand(-1, -1, D)
        reconstructed_features.scatter_(1, idx_pred, y_pred)
        
        # Reshape back to [B, T, N, D]
        reconstructed_features = reconstructed_features.view(B, T, N, D)
        
        # Also create a masked features map (zero out the right half) for visualization
        masked_features = torch.zeros_like(features_flat)
        masked_features.scatter_(1, idx_enc, x_context)
        masked_features = masked_features.view(B, T, N, D)
        
        print("Decoding to RGB...")
        # Decode Ground Truth
        decoded_gt = decoder(features.to(device))
        # Decode Masked (no prediction, just context)
        decoded_masked = decoder(masked_features)
        # Decode Inpainted (context + prediction)
        decoded_inpainted = decoder(reconstructed_features)
        
    print("Generating video...")
    out_path = os.path.join(OUTPUT_DIR, "inpainted_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # 3 frames side-by-side: Masked | Inpainted | Ground Truth
    w, h = 448, 448
    out_vid = cv2.VideoWriter(out_path, fourcc, 4, (w * 3, h))
    
    decoded_gt = denormalize(decoded_gt.squeeze(0)).cpu().numpy()
    decoded_masked = denormalize(decoded_masked.squeeze(0)).cpu().numpy()
    decoded_inpainted = denormalize(decoded_inpainted.squeeze(0)).cpu().numpy()
    
    for t in range(T):
        frame_gt = (decoded_gt[t].transpose(1, 2, 0) * 255).astype(np.uint8)
        frame_masked = (decoded_masked[t].transpose(1, 2, 0) * 255).astype(np.uint8)
        frame_inpainted = (decoded_inpainted[t].transpose(1, 2, 0) * 255).astype(np.uint8)
        
        # BGR for OpenCV
        frame_gt = cv2.cvtColor(frame_gt, cv2.COLOR_RGB2BGR)
        frame_masked = cv2.cvtColor(frame_masked, cv2.COLOR_RGB2BGR)
        frame_inpainted = cv2.cvtColor(frame_inpainted, cv2.COLOR_RGB2BGR)
        
        # Add labels
        cv2.putText(frame_masked, "Context Only", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_inpainted, "V-JEPA Inpainted", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_gt, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        concat_frame = np.concatenate([frame_masked, frame_inpainted, frame_gt], axis=1)
        out_vid.write(concat_frame)
        
    out_vid.release()
    print(f"Inpainting video saved to {out_path}")

if __name__ == "__main__":
    inpaint_video()
