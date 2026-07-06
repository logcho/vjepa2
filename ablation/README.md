# Ablation and Optimization of Spatial-Temporal Masking Strategies

This directory contains the implementations, configurations, visualizations, and scientific findings for **2: Ablation and Optimization of Spatial-Temporal Masking Strategies** for the V-JEPA 2.1 architecture.

---

## Directory Structure

* [motion_guided_masking.py](./motion_guided_masking.py): Core mask collator and generator classes supporting token-level and block-level motion guidance.
* [visualize_masking.py](./visualize_masking.py): Evaluation and visualization script. Loads videos, runs mask generators, overlays mask boundaries, computes metrics, and produces comparisons.
* [configs/pretrain_motion.yaml](./configs/pretrain_motion.yaml): Sample pretraining configuration demonstrating how to use the motion mask collator.
* [visualizations/](./visualizations):
  - [motion_heatmap.mp4](./visualizations/motion_heatmap.mp4): Heatmap of average motion maps across spatial-temporal tubelets.
  - [masking_random_multi_block.mp4](./visualizations/masking_random_multi_block.mp4): Default random multi-block masking.
  - [masking_token_high_motion_as_context.mp4](./visualizations/masking_token_high_motion_as_context.mp4): Voxel-level high motion kept as encoder context.
  - [masking_token_high_motion_as_target.mp4](./visualizations/masking_token_high_motion_as_target.mp4): Voxel-level high motion targeted as predictor masks.
  - [masking_block_high_motion_as_context.mp4](./visualizations/masking_block_high_motion_as_context.mp4): Block-level context biased towards high-motion areas.
  - [masking_block_high_motion_as_target.mp4](./visualizations/masking_block_high_motion_as_target.mp4): Block-level predictor targets biased towards high-motion areas.
  - [masking_comparison.mp4](./visualizations/masking_comparison.mp4): 2x3 grid comparison video showing all overlays side-by-side.
  - [keyframe_grid.png](./visualizations/keyframe_grid.png): Static keyframe visual comparison grid (Frames 20 and 40).
  - [masking_metrics.png](./visualizations/masking_metrics.png): Bar chart comparing Motion Overlap Ratio and Normalized Spatial Entropy across strategies.

---

## Mathematical and Technical Formulations

### 1. Motion Map Computation
We compute a dense motion map directly in PyTorch to enable fast, hardware-accelerated processing:
1. Let the video tensor batch be $X \in \mathbb{R}^{B \times C \times T \times H \times W}$.
2. Compute absolute temporal frame differences:
   $$\text{diff}_{b, c, t, h, w} = |X_{b, c, t+1, h, w} - X_{b, c, t, h, w}|$$
3. Average over channels to obtain a single channel motion map:
   $$\text{diff\_gray}_{b, t, h, w} = \frac{1}{C}\sum_{c=1}^C \text{diff}_{b, c, t, h, w}$$
4. Pad along the time dimension to restore length $T$:
   $$\text{diff\_gray}_{b, T-1, h, w} = \text{diff\_gray}_{b, T-2, h, w}$$
5. Apply 3D average pooling with kernel size and stride matching the token dimensions $(t_{\text{size}}, h_{\text{size}}, w_{\text{size}})$ (e.g. tubelet size $2$ and patch size $16 \times 16$):
   $$M = \text{AvgPool3d}(\text{diff\_gray})$$
   Where $M \in \mathbb{R}^{B \times d \times h_p \times w_p}$ represents the motion map matching the flat token grid dimensions $(d, h_p, w_p)$.

### 2. Block-level Conv3D Softmax Sampling
To perform block-level motion-guided sampling, we evaluate the motion sum of every potential block of size $(t, h, w)$:
1. Define a 3D convolutional kernel $K$ of shape $(1, 1, t, h, w)$ filled with ones.
2. Convolve the motion map $M$ (squeezed to batch dimension 1) with $K$ using stride=1 and padding=0:
   $$S = M * K$$
   The output $S$ has shape $(d - t + 1, h_p - h + 1, w_p - w + 1)$, where each element $S_{start, top, left}$ represents the sum of motion inside a block placed at those coordinates.
3. Compute the average block score:
   $$\bar{S} = \frac{S}{t \times h \times w}$$
4. Flatten $\bar{S}$ to a vector of length $K_{placements}$. We sample the flat placement index using a softmax distribution with temperature $\tau$:
   - For **`block_motion_target`** (biasing target blocks towards high motion):
     $$P_i = \frac{e^{\bar{S}_i / \tau}}{\sum_j e^{\bar{S}_j / \tau}}$$
   - For **`block_motion_context`** (biasing context blocks towards high motion, i.e., target blocks towards low motion):
     $$P_i = \frac{e^{-\bar{S}_i / \tau}}{\sum_j e^{-\bar{S}_j / \tau}}$$
5. We sample a placement index using $P_i$ and decode it back to the 3D block coordinates $(start, top, left)$.

---

## Quantitative Metrics Summary

Below are the quantitative results computed on `sample_video.mp4`:

| Masking Strategy | Motion Overlap Ratio (Target / High Motion) | Normalized Spatial Entropy (Dispersion) | Description |
| :--- | :---: | :---: | :--- |
| **Random Multi-block** | 0.8138 | 0.9314 | Balanced coverage with high spatial dispersion. |
| **Token High-Motion Context** | 0.0000 | 0.9966 | Keeps high-motion areas as context; predictor targets only static regions. |
| **Token High-Motion Target** | 1.0000 | 0.9889 | Predictor targets exclusively high-motion areas; context contains background. |
| **Block High-Motion Context** | 0.0391 | 0.6740 | Target blocks are sampled in static regions; context spans motion parts. |
| **Block High-Motion Target** | 0.3236 | 0.7259 | Target blocks are sampled in motion regions; context is biased to background. |

### Insights and Takeaways:
1. **Precision vs. Block Structure**:
   - Voxel/Token sorting (`Token High-Motion Target` / `Token High-Motion Context`) yields extreme overlap values ($100\%$ and $0\%$ respectively) while preserving near-perfect spatial dispersion (entropy $> 0.98$). However, it results in scattered token masks that break spatial-temporal block structures.
   - Block-level Conv3D sampling (`Block High-Motion Target` / `Block High-Motion Context`) preserves the contiguous 3D block structure required by the Vision Transformer's patch processing but displays lower spatial entropy (entropy $\sim 0.70$) due to local grouping.
2. **Motion Overlap Tradeoff**:
   - `Block High-Motion Target` achieves a significant increase in motion targeting ($32.36\%$) compared to `Block High-Motion Context` ($3.91\%$), demonstrating the effectiveness of the Conv3D softmax biasing.
   - Standard random multi-block masking shows relatively high overlap ($81.38\%$) on this specific short clip because the 8 sampled target blocks of scale 0.15 cover a large total area ($70\%$ of the video). This underscores the importance of scaling and density in mask generator configurations.
