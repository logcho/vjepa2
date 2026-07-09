#!/usr/bin/env python3
"""
Live Webcam Gesture Recognition → Motion Classification Pipeline

This script:
1. Opens the webcam and runs MediaPipe hand detection each frame.
2. When a closed fist is detected, starts tracking the fist centroid position.
3. Renders synthetic "circle" frames at the tracked positions (matching the training data format).
4. After collecting 8 tracked positions, extracts DINOv3 features and runs the
   V-JEPA Causal Future MLP probe to classify the motion trajectory.
5. Displays the live webcam feed with overlays showing tracking state, trajectory,
   and classification results.

Controls:
  - Press 'q' to quit
  - Press 'r' to reset tracking and start a new gesture
"""

import os
import sys
import cv2
import math
import time
import numpy as np
import torch
import torch.nn as nn

# Add workspace search paths
sys.path.append("/Users/loganchoi/Desktop/vjepa2/vjepa2")
sys.path.append("/Users/loganchoi/Desktop/dinov3/dinov3")

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

from src.models.predictor import VisionTransformerPredictor
from src.masks.utils import apply_masks

# ─── Configuration ───────────────────────────────────────────────────────────
FRAME_SIZE = 448          # Synthetic frame size for DINOv3
CIRCLE_RADIUS = 30        # Circle radius in the synthetic frames
NUM_TRACK_FRAMES = 8      # Number of positions to collect before classifying
CAPTURE_INTERVAL = 0.15   # Seconds between position captures (controls speed sensitivity)

CLASSES_DESC = [
    "Horizontal Left-to-Right",
    "Horizontal Right-to-Left",
    "Vertical Bottom-to-Top",
    "Vertical Top-to-Bottom",
    "Clockwise Circle",
    "Counter-Clockwise Circle",
    "Diagonal TL to BR",
    "Diagonal BL to TR"
]

CLASS_COLORS = [
    (46, 204, 113),   # Green
    (231, 76, 60),    # Red
    (52, 152, 219),   # Blue
    (241, 196, 15),   # Yellow
    (155, 89, 182),   # Purple
    (26, 188, 156),   # Teal
    (230, 126, 34),   # Orange
    (149, 165, 166),  # Gray
]

# ─── MLP Probe (must match training architecture) ────────────────────────────
class MLPProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)


# ─── Hand / Fist Detection (MediaPipe Tasks API) ────────────────────────────
def is_fist(landmarks):
    """
    Detect a closed fist by checking if all four finger tips
    are below their respective PIP (proximal interphalangeal) joints.
    `landmarks` is a list of mediapipe NormalizedLandmark objects.
    Landmark indices:
      - Finger tips: 8, 12, 16, 20
      - Finger PIPs: 6, 10, 14, 18
    """
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    closed_count = 0
    for tip_idx, pip_idx in zip(tips, pips):
        tip = landmarks[tip_idx]
        pip_joint = landmarks[pip_idx]
        # In MediaPipe's coordinate system, y increases downward.
        # A curled finger has its tip below (higher y) than the PIP joint.
        if tip.y > pip_joint.y:
            closed_count += 1
    # Also check thumb: tip (4) vs IP joint (3)
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    # For thumb, check x distance to wrist instead (works for both hands)
    wrist = landmarks[0]
    if abs(thumb_tip.x - wrist.x) < abs(thumb_ip.x - wrist.x):
        closed_count += 1
    return closed_count >= 4  # At least 4 of 5 fingers curled


def get_fist_center(landmarks, frame_w, frame_h):
    """
    Get the pixel center of the fist using the wrist (0) and middle finger MCP (9) midpoint.
    `landmarks` is a list of mediapipe NormalizedLandmark objects.
    """
    wrist = landmarks[0]
    mcp = landmarks[9]
    cx = int((wrist.x + mcp.x) / 2 * frame_w)
    cy = int((wrist.y + mcp.y) / 2 * frame_h)
    return cx, cy


# ─── Synthetic Frame Renderer ───────────────────────────────────────────────
def render_synthetic_frame(cx, cy, frame_size=FRAME_SIZE, radius=CIRCLE_RADIUS):
    """
    Render a synthetic black frame with a white circle at (cx, cy),
    matching the training data format used in generate_data.py.
    """
    frame = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)
    cv2.circle(frame, (int(cx), int(cy)), radius, (255, 255, 255), -1)
    return frame


# ─── Main Pipeline ──────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Live Webcam Gesture → Motion Classification Pipeline")
    print("=" * 60)

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # ── Load DINOv3 backbone ──
    print("Loading DINOv3 backbone...")
    dinov3_model = torch.hub.load(
        '/Users/loganchoi/Desktop/dinov3/dinov3',
        'dinov3_vits16',
        source='local',
        pretrained=True
    ).to(device)
    dinov3_model.eval()
    for p in dinov3_model.parameters():
        p.requires_grad = False

    # ── Load V-JEPA Predictor ──
    print("Loading V-JEPA Predictor...")
    predictor = VisionTransformerPredictor(
        img_size=(FRAME_SIZE, FRAME_SIZE),
        patch_size=16,
        num_frames=NUM_TRACK_FRAMES,
        tubelet_size=1,
        embed_dim=384,
        predictor_embed_dim=192,
        out_embed_dim=384,
        depth=4,
        num_heads=6,
        use_mask_tokens=True,
        num_mask_tokens=1
    ).to(device)
    predictor.load_state_dict(torch.load(
        "/Users/loganchoi/Desktop/vjepa2/dinov3_exploration/pathway_1/predictor_pathway1.pth",
        map_location=device
    ))
    predictor.eval()

    # ── Load Pre-Trained MLP Probe ──
    print("Loading pre-trained MLP classifier...")
    data_dir = "/Users/loganchoi/Desktop/vjepa2/dinov3_exploration/pathway_1/video_classification"
    mlp_save_path = os.path.join(data_dir, "mlp_probe.pth")
    if not os.path.exists(mlp_save_path):
        print(f"ERROR: Pre-trained probe not found at {mlp_save_path}. Run train_probe.py first to train and save the probe.")
        sys.exit(1)

    T = 8
    N_patches = 784
    D = 384

    mlp_model = MLPProbe(input_dim=D).to(device)
    mlp_model.load_state_dict(torch.load(mlp_save_path, map_location=device))
    mlp_model.eval()
    print("MLP Classifier loaded successfully!")

    # ── ImageNet normalization constants ──
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # ── MediaPipe Hands (Tasks API) ──
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task"
    )
    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    hand_landmarker = HandLandmarker.create_from_options(hand_options)
    frame_timestamp_ms = 0

    # ── Open Webcam ──
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n" + "=" * 60)
    print("  READY! Show your fist to the camera to start tracking.")
    print("  Move your fist in a pattern, then open your hand to classify.")
    print("  Press 'q' to quit, 'r' to reset.")
    print("=" * 60 + "\n")

    # ── State Variables ──
    tracking = False
    tracked_positions = []       # List of (cx, cy) in webcam coords
    last_capture_time = 0
    classification_result = None  # (class_idx, confidence, probs)
    fist_detected = False
    frame_w, frame_h = 640, 480

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror for natural interaction
        frame_h, frame_w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Hand Detection (Tasks API) ──
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        frame_timestamp_ms += 33  # ~30fps
        results = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        fist_detected = False
        hand_center = None

        if results.hand_landmarks:
            for landmarks in results.hand_landmarks:
                # Draw hand skeleton on the frame
                for lm in landmarks:
                    px = int(lm.x * frame_w)
                    py = int(lm.y * frame_h)
                    cv2.circle(frame, (px, py), 3, (200, 200, 200), -1)
                # Draw connections
                connections = [
                    (0,1),(1,2),(2,3),(3,4),  # Thumb
                    (0,5),(5,6),(6,7),(7,8),  # Index
                    (5,9),(9,10),(10,11),(11,12),  # Middle
                    (9,13),(13,14),(14,15),(15,16),  # Ring
                    (13,17),(17,18),(18,19),(19,20),(0,17)  # Pinky + palm
                ]
                for i, j in connections:
                    p1 = (int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h))
                    p2 = (int(landmarks[j].x * frame_w), int(landmarks[j].y * frame_h))
                    cv2.line(frame, p1, p2, (100, 100, 100), 1)

                fist_detected = is_fist(landmarks)
                hand_center = get_fist_center(landmarks, frame_w, frame_h)

                if fist_detected:
                    # Draw a filled circle at fist position
                    cv2.circle(frame, hand_center, 20, (0, 255, 0), -1)
                    cv2.circle(frame, hand_center, 22, (255, 255, 255), 2)

        # ── State Machine ──
        current_time = time.time()

        if not tracking and fist_detected and hand_center:
            # Start tracking
            tracking = True
            tracked_positions = [hand_center]
            last_capture_time = current_time
            classification_result = None
            print(f"[TRACKING] Started! Position 1/{NUM_TRACK_FRAMES}: {hand_center}")

        elif tracking and fist_detected and hand_center:
            # Continue tracking - capture position at intervals
            if current_time - last_capture_time >= CAPTURE_INTERVAL:
                if len(tracked_positions) < NUM_TRACK_FRAMES:
                    tracked_positions.append(hand_center)
                    last_capture_time = current_time
                    print(f"[TRACKING] Position {len(tracked_positions)}/{NUM_TRACK_FRAMES}: {hand_center}")

                    if len(tracked_positions) == NUM_TRACK_FRAMES:
                        # ── CLASSIFY! ──
                        print("\n[CLASSIFYING] Running DINOv3 + V-JEPA pipeline...")

                        # Map webcam coords to synthetic frame coords
                        synthetic_coords = []
                        for (wx, wy) in tracked_positions:
                            sx = int(np.clip(wx / frame_w * FRAME_SIZE, CIRCLE_RADIUS, FRAME_SIZE - CIRCLE_RADIUS))
                            sy = int(np.clip(wy / frame_h * FRAME_SIZE, CIRCLE_RADIUS, FRAME_SIZE - CIRCLE_RADIUS))
                            synthetic_coords.append((sx, sy))

                        # Render synthetic frames
                        synthetic_frames = []
                        for (sx, sy) in synthetic_coords:
                            synthetic_frames.append(render_synthetic_frame(sx, sy))
                        synthetic_frames = np.stack(synthetic_frames)  # [8, 448, 448, 3]

                        # Extract DINOv3 features
                        tensor = torch.from_numpy(synthetic_frames).permute(0, 3, 1, 2).float().to(device) / 255.0
                        normalized = (tensor - mean) / std

                        with torch.no_grad():
                            outputs = dinov3_model.forward_features(normalized)
                            patch_tokens = outputs["x_norm_patchtokens"]  # [8, 784, 384]

                            # Causal future masking: context=frames 0-3, predict=frames 4-7
                            feats_flat = patch_tokens.unsqueeze(0).reshape(1, T * N_patches, D)
                            masks_enc = torch.arange(0, 4 * N_patches, device=device).unsqueeze(0)
                            masks_pred = torch.arange(4 * N_patches, T * N_patches, device=device).unsqueeze(0)

                            x_context = apply_masks(feats_flat, [masks_enc])
                            y_pred = predictor(x_context, masks_enc, masks_pred)
                            pooled = y_pred.mean(dim=1)  # [1, 384]

                            logits = mlp_model(pooled)
                            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                            pred_class = int(probs.argmax())
                            pred_conf = float(probs[pred_class])

                        classification_result = (pred_class, pred_conf, probs)
                        print(f"[RESULT] Predicted: {CLASSES_DESC[pred_class]} ({pred_conf*100:.1f}%)")
                        for i, p in enumerate(probs):
                            marker = " <<<" if i == pred_class else ""
                            print(f"   C{i}: {CLASSES_DESC[i]:<30} {p*100:5.1f}%{marker}")
                        print()

        elif tracking and not fist_detected:
            # Hand opened or lost - if we have enough positions, keep result displayed
            if len(tracked_positions) < NUM_TRACK_FRAMES:
                print(f"[RESET] Hand lost with only {len(tracked_positions)}/{NUM_TRACK_FRAMES} positions. Resetting.")
                tracking = False
                tracked_positions = []
                classification_result = None

        # ── Draw Overlay ──
        overlay = frame.copy()

        # Draw trajectory trail
        if len(tracked_positions) >= 2:
            for i in range(1, len(tracked_positions)):
                progress = i / (NUM_TRACK_FRAMES - 1)
                color = (int(50 + 200 * progress), int(255 * (1 - progress)), int(100))
                thickness = int(2 + 3 * progress)
                cv2.line(overlay, tracked_positions[i-1], tracked_positions[i], color, thickness)
            # Draw dots at each position
            for i, pos in enumerate(tracked_positions):
                progress = i / (NUM_TRACK_FRAMES - 1)
                color = (int(50 + 200 * progress), int(255 * (1 - progress)), int(100))
                cv2.circle(overlay, pos, 6, color, -1)
                cv2.circle(overlay, pos, 7, (255, 255, 255), 1)

        # Status bar at the top
        status_bar_h = 50
        cv2.rectangle(overlay, (0, 0), (frame_w, status_bar_h), (30, 30, 30), -1)

        if tracking and len(tracked_positions) < NUM_TRACK_FRAMES:
            status_text = f"TRACKING: {len(tracked_positions)}/{NUM_TRACK_FRAMES} positions"
            cv2.putText(overlay, status_text, (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Progress bar
            bar_x = 380
            bar_w = 240
            bar_h = 16
            progress = len(tracked_positions) / NUM_TRACK_FRAMES
            cv2.rectangle(overlay, (bar_x, 17), (bar_x + bar_w, 17 + bar_h), (80, 80, 80), -1)
            cv2.rectangle(overlay, (bar_x, 17), (bar_x + int(bar_w * progress), 17 + bar_h), (0, 255, 0), -1)

        elif classification_result:
            pred_class, pred_conf, probs = classification_result
            color = CLASS_COLORS[pred_class]
            status_text = f"CLASSIFIED: {CLASSES_DESC[pred_class]} ({pred_conf*100:.0f}%)"
            cv2.putText(overlay, status_text, (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        else:
            status_text = "Show FIST to start tracking"
            cv2.putText(overlay, status_text, (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Fist indicator
        if fist_detected:
            cv2.putText(overlay, "FIST", (frame_w - 80, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(overlay, "OPEN", (frame_w - 80, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

        # Classification probability sidebar (when result available)
        if classification_result:
            pred_class, pred_conf, probs = classification_result
            sidebar_x = frame_w - 220
            sidebar_y = 60
            bar_height = 18
            bar_spacing = 22

            cv2.rectangle(overlay, (sidebar_x - 10, sidebar_y - 5),
                          (frame_w - 5, sidebar_y + 8 * bar_spacing + 10), (20, 20, 20), -1)
            cv2.rectangle(overlay, (sidebar_x - 10, sidebar_y - 5),
                          (frame_w - 5, sidebar_y + 8 * bar_spacing + 10), (100, 100, 100), 1)

            for i in range(8):
                y = sidebar_y + i * bar_spacing
                bar_w = int(probs[i] * 180)
                color = CLASS_COLORS[i] if i == pred_class else (80, 80, 80)
                cv2.rectangle(overlay, (sidebar_x, y), (sidebar_x + bar_w, y + bar_height), color, -1)
                label = f"C{i}: {probs[i]*100:.0f}%"
                cv2.putText(overlay, label, (sidebar_x + 2, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Controls hint
        cv2.putText(overlay, "q: quit | r: reset", (10, frame_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

        cv2.imshow("Gesture Motion Classification", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracking = False
            tracked_positions = []
            classification_result = None
            print("[RESET] Tracking cleared.")

    cap.release()
    cv2.destroyAllWindows()
    hand_landmarker.close()
    print("Pipeline closed.")


if __name__ == "__main__":
    main()
