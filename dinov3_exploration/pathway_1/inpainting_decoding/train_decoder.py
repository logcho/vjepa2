#!/usr/bin/env python3
import os
import sys
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from decord import VideoReader

# Add workspace search paths
sys.path.append("/Users/loganchoi/Desktop/vjepa2/vjepa2")
sys.path.append("/Users/loganchoi/Desktop/dinov3/dinov3")

# Configuration
VIDEO_PATH = "/Users/loganchoi/Desktop/vjepa2/sample_video.mp4"
OUTPUT_DIR = "/Users/loganchoi/Desktop/vjepa2/dinov3_exploration/pathway_1/inpainting_decoding"
NUM_FRAMES = 32
STRIDE = 2
BATCH_SIZE = 4
EPOCHS = 100
LR = 1e-3

class FeatureDecoder(nn.Module):
    def __init__(self, embed_dim=384):
        super().__init__()
        # Input: (B*T, 384, 28, 28)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 192, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(192, affine=False),
            nn.GELU(),
            # Output: 56x56
            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(96, affine=False),
            nn.GELU(),
            # Output: 112x112
            nn.ConvTranspose2d(96, 48, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(48, affine=False),
            nn.GELU(),
            # Output: 224x224
            nn.ConvTranspose2d(48, 3, kernel_size=4, stride=2, padding=1, bias=False),
            # Output: 448x448
        )

    def forward(self, x):
        """
        x: [B, T, N, D] where N = 28*28 = 784, D = 384
        returns: [B, T, 3, 448, 448]
        """
        B, T, N, D = x.shape
        H_p = W_p = int(math.sqrt(N))
        
        # Reshape to [B*T, D, H_p, W_p]
        x_reshaped = x.view(B * T, H_p, W_p, D).permute(0, 3, 1, 2)
        
        # Decode
        decoded = self.decoder(x_reshaped) # [B*T, 3, 448, 448]
        
        # Reshape back to [B, T, 3, 448, 448]
        return decoded.view(B, T, 3, 448, 448)

def load_and_preprocess_video(video_path, num_frames=32, target_size=(448, 448)):
    print(f"Loading video from {video_path}...")
    vr = VideoReader(video_path)
    total_frames = len(vr)
    
    indices = np.arange(0, min(total_frames, num_frames * STRIDE), STRIDE)
    if len(indices) < num_frames:
        indices = np.pad(indices, (0, num_frames - len(indices)), mode="edge")
    indices = indices[:num_frames]
    
    frames = vr.get_batch(indices).asnumpy()  # T x H x W x C
    T, H, W, C = frames.shape
    
    shortest_edge = int(max(256, max(target_size) * 1.15))
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
    
    # Normalize to [0, 1] and ImageNet statistics
    tensor = torch.from_numpy(cropped_frames).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    normalized = (tensor - mean) / std
    return normalized, tensor

def extract_dinov3_features(model, video_tensor, device):
    """
    Pass T frames through DINOv3 to extract frame-level spatial patch tokens.
    """
    video_tensor = video_tensor.to(device)
    with torch.no_grad():
        outputs = model.forward_features(video_tensor)
        patch_tokens = outputs["x_norm_patchtokens"]
    return patch_tokens.cpu()

def train_decoder():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
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
        
    normalized_frames, raw_frames = load_and_preprocess_video(VIDEO_PATH, num_frames=NUM_FRAMES, target_size=(448, 448))
    print(f"Video tensor shape: {normalized_frames.shape}")
    
    print("Extracting DINOv3 patch tokens...")
    # [T, 784, 384]
    features = extract_dinov3_features(dinov3_model, normalized_frames, device)
    print(f"Extracted patch tokens shape: {features.shape}")
    
    # We will train the decoder using individual frames as batch elements to maximize variety
    # Our decoder takes [B, T, N, D], so we'll wrap frames in a fake T=1 dimension
    T_total = features.shape[0]
    
    decoder = FeatureDecoder(embed_dim=384).to(device)
    optimizer = optim.AdamW(decoder.parameters(), lr=LR)
    # We will compute MSE loss against the NORMALIZED frames
    criterion = nn.MSELoss()
    
    print("Starting decoder training...")
    for epoch in range(1, EPOCHS + 1):
        decoder.train()
        epoch_loss = 0.0
        
        # Shuffle frames
        indices = torch.randperm(T_total)
        
        num_batches = math.ceil(T_total / BATCH_SIZE)
        
        for b in range(num_batches):
            batch_indices = indices[b*BATCH_SIZE : (b+1)*BATCH_SIZE]
            
            # [B_curr, 1, N, D]
            batch_features = features[batch_indices].unsqueeze(1).to(device)
            # [B_curr, 1, 3, H, W]
            batch_targets = normalized_frames[batch_indices].unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            batch_preds = decoder(batch_features)
            
            loss = criterion(batch_preds, batch_targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch_indices)
            
        avg_loss = epoch_loss / T_total
        
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d}/{EPOCHS} | Loss: {avg_loss:.5f}")
            
    print("Training completed.")
    
    decoder_path = os.path.join(OUTPUT_DIR, "decoder.pth")
    torch.save(decoder.state_dict(), decoder_path)
    print(f"Decoder weights saved to: {decoder_path}")

if __name__ == "__main__":
    train_decoder()
