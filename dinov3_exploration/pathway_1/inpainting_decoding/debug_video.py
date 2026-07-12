import cv2
import numpy as np

def debug_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w, c = frame.shape
        w_single = w // 3
        
        frame_masked = frame[:, :w_single]
        frame_inpainted = frame[:, w_single:2*w_single]
        frame_gt = frame[:, 2*w_single:]
        
        diff_masked_gt = np.mean((frame_masked - frame_gt)**2)
        diff_inpaint_gt = np.mean((frame_inpainted - frame_gt)**2)
        diff_masked_inpaint = np.mean((frame_masked - frame_inpainted)**2)
        
        print(f"Frame {frame_idx}:")
        print(f"  MSE(Masked, GT): {diff_masked_gt:.2f}")
        print(f"  MSE(Inpainted, GT): {diff_inpaint_gt:.2f}")
        print(f"  MSE(Masked, Inpainted): {diff_masked_inpaint:.2f}")
        frame_idx += 1
    cap.release()

debug_video('/Users/loganchoi/Desktop/vjepa2/dinov3_exploration/pathway_1/inpainting_decoding/inpainted_video.mp4')
