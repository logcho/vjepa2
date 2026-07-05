# Option 1: Dense Feature Visualization and PCA-based Temporal Tracking

This directory contains the code, visualizations, and scientific findings for **Option 1: Dense Feature Visualization and PCA-based Temporal Tracking** for investigating V-JEPA 2.1 self-supervised representation models.

---

## 📂 Directory Structure

* 📄 [visualize_pca.py](file:///Users/loganchoi/Desktop/vjepa2/option1/visualize_pca.py): Main execution script for token extraction, SVD-based PCA projection, interpolation, overlays, and stability analyses.
* 📂 [visualizations/](file:///Users/loganchoi/Desktop/vjepa2/option1/visualizations): Directory containing output videos and static graphs:
  * 🎥 [pca_raw.mp4](file:///Users/loganchoi/Desktop/vjepa2/option1/visualizations/pca_raw.mp4): Raw projected 3-component PCA colors over the video.
  * 🎥 [pca_overlay.mp4](file:///Users/loganchoi/Desktop/vjepa2/option1/visualizations/pca_overlay.mp4): Alpha-blended overlay of PCA colors onto original frames.
  * 🎥 [pca_side_by_side.mp4](file:///Users/loganchoi/Desktop/vjepa2/option1/visualizations/pca_side_by_side.mp4): Original frames side-by-side with global PCA.
  * 🎥 [pca_global_vs_per_frame.mp4](file:///Users/loganchoi/Desktop/vjepa2/option1/visualizations/pca_global_vs_per_frame.mp4): Side-by-side comparison of Global SVD vs. Per-Frame SVD projection.
  * 🎥 [pca_flicker_comparison.mp4](file:///Users/loganchoi/Desktop/vjepa2/option1/visualizations/pca_flicker_comparison.mp4): Side-by-side comparison of Global vs. Per-Frame temporal variance (flicker).
  * 🖼️ [keyframe_grid.png](file:///Users/loganchoi/Desktop/vjepa2/option1/visualizations/keyframe_grid.png): 4-row grid displaying Keyframes of original, global, per-frame, and temporal variance.
  * 🖼️ [pca_trajectory_plot.png](file:///Users/loganchoi/Desktop/vjepa2/option1/visualizations/pca_trajectory_plot.png): Quantitative line plots of PCA component trajectories over all 64 frames.

---

## 🚀 How to Run

To run the pipeline and regenerate all visualizations:
```bash
.venv/bin/python option1/visualize_pca.py --video_path sample_video.mp4 --output_dir option1/visualizations --num_frames 64
```

### Script Arguments:
* `--video_path`: Path to input video (default: `sample_video.mp4`).
* `--model_name`: Hugging Face model checkpoint (default: `facebook/vjepa2-vitl-fpc64-256`).
* `--output_dir`: Directory to save outputs (default: `option1/visualizations`).
* `--num_frames`: Number of frames to sample from video (default: 64).
* `--stride`: Sampling stride (default: 2).
* `--alpha`: Alpha blend weight for the overlay video (default: 0.5).

---

## 🔬 Scientific & Architectural Insights

Our exploration of V-JEPA 2.1 features led to several key findings regarding self-supervised representation stability:

### 1. What is a Latent Representation?
Instead of representing videos using raw RGB pixels (which are high-dimensional, redundant, and context-free), V-JEPA maps spatial-temporal 3D blocks (tubelets of size $2\text{ frames} \times 16 \times 16\text{ pixels}$) to a **1024-dimensional latent space** using a Vision Transformer (ViT). 

In this latent space, spatial-temporal tokens are placed close to each other based on **semantic and dynamic similarity** rather than raw pixel colors. By running PCA SVD, we project this 1024-D space onto its top 3 principal components, mapping them directly to RGB values for human visualization.

### 2. Contextual "Leakage" via Self-Attention
Even when a region in the video is completely static (e.g. a background wall), the plotted PCA curves exhibit minor, smooth fluctuations. This is because **Self-Attention** in Transformers causes every token to exchange information with all other tokens. As the bowler moves and the ball rolls, the changing global context "leaks" into the static background representations, causing their values to shift slightly over time.

### 3. Quantitative Stability Verification
To prove representation stability, the project implements three analytical methods:
1. **Global vs. Per-Frame PCA**: Recalculating PCA independently per-frame rotates the projection coordinate axes randomly, resulting in chaotic color flickers (`pca_global_vs_per_frame.mp4`). Global PCA preserves the coordinate space across all frames, establishing a coherent tracking manifold.
2. **Flicker Map (Temporal Variance)**: By calculating the standard deviation of colors over a sliding window of 5 frames, we visualize instability. Static areas remain entirely **black** (zero variance) under Global PCA, whereas Per-Frame PCA results in a flashing, white-noise variance storm (`pca_flicker_comparison.mp4`).
3. **Feature Trajectory Plotting**: By graphing the projected component values of individual pixels over all 64 frames, we show that static background regions display **flat, straight lines** in V-JEPA, whereas unstable per-frame baselines show jagged, erratic spikes (`pca_trajectory_plot.png`).
