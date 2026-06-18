#!/usr/bin/env python3
"""
Simple script to test the V-JEPA 2 model.
This script:
1. Downloads a sample video from Kinetics-mini if not already present.
2. Uses decord (installed via eva-decord) to load and sample video frames.
3. Automatically detects the best available hardware accelerator (CUDA, MPS, or CPU).
4. Loads the V-JEPA 2 ViT-L model and processor from Hugging Face.
5. Runs a forward pass to extract video features.
6. Prints the output shapes and feature statistics.
"""

import os
import urllib.request
import numpy as np
import torch
from decord import VideoReader
from transformers import AutoModel, AutoVideoProcessor

SAMPLE_VIDEO_URL = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/bowling/-WH-lxmGJVY_000005_000015.mp4"
SAMPLE_VIDEO_PATH = "sample_video.mp4"
HF_MODEL_NAME = "facebook/vjepa2-vitl-fpc64-256"


def download_sample_video(url, path):
    if not os.path.exists(path):
        print(f"Downloading sample video from: {url}")
        # Standard urllib is used to ensure cross-platform compatibility without wget/curl
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
        last_frame = frame_idx[-1] if len(frame_idx) > 0 else 0
        frame_idx = np.pad(frame_idx, (0, num_frames - len(frame_idx)), mode="edge")

    # Keep only the requested number of frames
    frame_idx = frame_idx[:num_frames]

    print(f"Sampling frame indices: {frame_idx}")
    video_data = vr.get_batch(frame_idx).asnumpy()  # T x H x W x C
    return video_data


def run_inference():
    # 1. Download video
    download_sample_video(SAMPLE_VIDEO_URL, SAMPLE_VIDEO_PATH)

    # 2. Load and sample video frames
    video = load_video_frames(SAMPLE_VIDEO_PATH, num_frames=64, stride=2)
    print(f"Loaded video numpy shape: {video.shape} (T x H x W x C)")

    # 3. Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS device.")
    else:
        device = torch.device("cpu")
        print("Using CPU device.")

    # 4. Load processor and model
    print(f"Loading processor for: {HF_MODEL_NAME}...")
    processor = AutoVideoProcessor.from_pretrained(HF_MODEL_NAME)
    
    print(f"Loading model: {HF_MODEL_NAME}...")
    model = AutoModel.from_pretrained(HF_MODEL_NAME).to(device)
    model.eval()

    # 5. Process video to tensor
    # The processor expects lists of frames or a batch of frames, typically T x C x H x W
    # Convert T x H x W x C to T x C x H x W
    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)
    
    print("Preprocessing video tensor...")
    inputs = processor(video_tensor, return_tensors="pt")
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print(f"Inputs processed shape: {inputs['pixel_values_videos'].shape} (B x T x C x H x W)")

    # 6. Forward pass
    print("Running model inference to extract features...")
    try:
        with torch.no_grad():
            outputs = model(**inputs)
        features = outputs.last_hidden_state
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "mps backend out of memory" in str(e).lower():
            print(f"\n[Warning] GPU/MPS inference failed due to memory limit: {e}")
            print("Falling back to CPU device for inference. This might take a little longer...")
            # Move model and inputs to CPU
            model = model.to("cpu")
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            features = outputs.last_hidden_state
        else:
            raise e
    
    # Retrieve last hidden states (features)
    # The V-JEPA 2 model returns features of shape [batch_size, num_tokens, embedding_dim]
    features = outputs.last_hidden_state
    print("\n--- Inference Results ---")
    print(f"Features shape: {features.shape} (Batch Size x Num Tokens x Embedding Dim)")
    
    # Calculate some basic statistics
    features_cpu = features.cpu()
    print(f"Features mean:  {features_cpu.mean().item():.4f}")
    print(f"Features std:   {features_cpu.std().item():.4f}")
    print(f"Features min:   {features_cpu.min().item():.4f}")
    print(f"Features max:   {features_cpu.max().item():.4f}")
    print("-------------------------")
    print("Success! V-JEPA 2 model is running and extracting dense features.")


if __name__ == "__main__":
    run_inference()
