# Pathway 1: Frozen DINOv3 Feature Prediction (Lightweight World Model)

This directory implements the core integration and validation of Pathway 1, connecting a frozen pretrained **DINOv3** vision model with a trainable **V-JEPA Predictor**.

## Directory Structure

*   [explore_pathway_1.py](file:///Users/loganchoi/Desktop/vjepa2/dinov3_exploration/pathway_1/explore_pathway_1.py): Script to train and evaluate the V-JEPA predictor on dense DINOv3 spatio-temporal representations.
*   [live_cv_pipeline.py](file:///Users/loganchoi/Desktop/vjepa2/dinov3_exploration/pathway_1/live_cv_pipeline.py): Live CV pipeline demonstrating online inference on a sliding queue and visualizing predictions side-by-side.
*   [predictor_pathway1.pth](file:///Users/loganchoi/Desktop/vjepa2/dinov3_exploration/pathway_1/predictor_pathway1.pth): Exported PyTorch state dict weights of the trained predictor model.
*   [pipeline_visualization.mp4](file:///Users/loganchoi/Desktop/vjepa2/dinov3_exploration/pathway_1/pipeline_visualization.mp4): Visualization video demonstrating live predictions vs ground-truth DINOv3 features.
*   **Downstream Applications**:
    *   [planning_rl/](file:///Users/loganchoi/Desktop/vjepa2/dinov3_exploration/pathway_1/planning_rl/): Code and configs for utilizing the predictor as a latent world model for trajectory planning and Model Predictive Control (MPC).
    *   [video_classification/](file:///Users/loganchoi/Desktop/vjepa2/dinov3_exploration/pathway_1/video_classification/): Implementation of spatiotemporal action recognition via linear probing and MLP classification on predicted future features. Includes [generate_data.py](file:///Users/loganchoi/Desktop/vjepa2/dinov3_exploration/pathway_1/video_classification/generate_data.py) (synthetic action videos with pre-extracted DINOv3 features) and [train_probe.py](file:///Users/loganchoi/Desktop/vjepa2/dinov3_exploration/pathway_1/video_classification/train_probe.py) (training/evaluating spatiotemporal classification probes).
    *   [anomaly_detection/](file:///Users/loganchoi/Desktop/vjepa2/dinov3_exploration/pathway_1/anomaly_detection/): Scripts for live video anomaly/prediction error monitoring based on prediction loss spikes.
    *   [inpainting_decoding/](file:///Users/loganchoi/Desktop/vjepa2/dinov3_exploration/pathway_1/inpainting_decoding/): Code for training decoder networks to project predictor latent states back to pixel space for video inpainting and synthesis.

---

## Performance Summary (448px Resolution)

The model is evaluated using a challenging high-resolution configuration ($448 \times 448$ pixels, yielding $28 \times 28 = 784$ patches per frame).

*   **Training Loss (MSE)**: Reduced from `0.09855` (Epoch 1) to `0.02974` (Epoch 40).
*   **Prediction Cosine Similarity**: Reached **0.8885** at Epoch 40.
*   **Static Context Baseline**: **0.6906**.
*   **Relative Improvement vs Baseline**: **+28.7%**.

The significant outperformance against the baseline verifies that the V-JEPA predictor has successfully learned to model fine-grained spatiotemporal video dynamics rather than simply copying context representations.

---

## Downstream Application Roadmaps

### 1. Latent State Planning (`planning_rl/`)
Use the V-JEPA predictor as a simulator of visual semantics:
*   Given current observation features $z_t$, rollout the predictor $K$ steps ahead using action embeddings $a_{t:t+K}$.
*   Evaluate predicted latents $\hat{z}_{t+K}$ against target task rewards to plan optimal action sequences without rendering intermediate pixels.

### 2. Action Recognition (`video_classification/`)
Train linear and MLP classifiers on frozen predictor spatiotemporal features:
*   **Methodology**: Extract spatiotemporal representations by masking frames 4–7 (the future) and predicting them using context frames 0–3. Pooled target predictions `y_pred` represent future dynamics.
*   **Probing Accuracy**:
    *   **Baseline A (DINOv3-Mean)**: **`57.5%`** (Temporal order lost).
    *   **Baseline B (DINOv3-Concat)**: **`85.0%`** (Temporally flattened).
    *   **V-JEPA Predictor (Random Mask MLP)**: **`62.5%`** (Noisy due to dynamic masks).
    *   **V-JEPA Predictor (Causal Future MLP)**: **`95.0%`** (Strong spatiotemporal modeling).
*   The high accuracy of the Causal Future MLP probe proves that V-JEPA's predictions of the future are highly discriminative of video dynamics. See the full [video_classification README](file:///Users/loganchoi/Desktop/vjepa2/dinov3_exploration/pathway_1/video_classification/README.md) for details.

### 3. Live Anomaly Detection (`anomaly_detection/`)
Monitor feature prediction errors:
*   Compute online MSE loss or cosine distance between predicted features $\hat{y}$ and future ground-truth features $y$.
*   Flag intervals where the prediction error exceeds a rolling standard deviation threshold as physical anomalies or unpredictable events.

### 4. Semantic Video Inpainting (`inpainting_decoding/`)
Project latent codes back to RGB pixels:
*   Feed the predicted representations of the masked block to a lightweight decoder (such as a transposed-convolutional U-Net).
*   Reconstruct clean pixel outputs to dynamically fill in or remove objects in video streams.
