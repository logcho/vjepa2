# Option 3: Hierarchical Layer Representation Analysis

This directory contains implementation, code, visualizations, and scientific findings for **Option 3: Hierarchical Layer Representation Analysis** of the V-JEPA 2.1 architecture.

---

## Directory Structure

* [visualize_hierarchical.py](./visualize_hierarchical.py): Execution pipeline script. Loads a video, extracts representations from all 25 layer states, trains linear probes, computes Representational Similarity Analysis (RSA), runs PCA on multiple layers, and produces visualization charts and grid videos.
* [visualizations/](./visualizations):
  - [layer_probing_metrics.png](./visualizations/layer_probing_metrics.png): Validation accuracy curves for Structure (Spatial Location) and Motion magnitude classification across layers.
  - [context_leakage_rsa.png](./visualizations/context_leakage_rsa.png): Average cosine similarity curves (leakage/cross-similarity, foreground coherence, and background coherence) across layers.
  - [layer_pca_comparison.mp4](./visualizations/layer_pca_comparison.mp4): A 2x3 grid comparison video showing the original video alongside the trilinearly interpolated PCA overlays at Layers 2, 8, 14, 20, and 24.
  - [keyframe_layers_grid.png](./visualizations/keyframe_layers_grid.png): Static keyframe grid comparing original frames vs PCA overlays across different depths.
  - Individual layer PCA overlay videos: `layer_2_pca.mp4`, `layer_8_pca.mp4`, `layer_14_pca.mp4`, `layer_20_pca.mp4`, `layer_24_pca.mp4`.

---

## Mathematical and Technical Formulations

### 1. Linear Probing for Feature Abstraction
To quantify the information content at varying transformer depths, we train linear classifiers on the frozen representations $z_{t, y, x}^{(l)} \in \mathbb{R}^D$ (where $D = 1024$ and $l \in [0, 24]$) on two proxy tasks:

* **Structure Task (Spatial Coordinate Classification)**:
  Predict the row $Y \in \{0, \dots, 15\}$ and column $X \in \{0, \dots, 15\}$ of the patch token.
  $$\hat{P}(\text{Row} = y) = \text{Softmax}(W_{\text{row}}^{(l)} z_{t, y, x}^{(l)} + b_{\text{row}}^{(l)})$$
  $$\hat{P}(\text{Col} = x) = \text{Softmax}(W_{\text{col}}^{(l)} z_{t, y, x}^{(l)} + b_{\text{col}}^{(l)})$$
  $$\mathcal{L}_{\text{structure}} = \mathcal{L}_{\text{CE}}(\hat{P}(\text{Row}), Y) + \mathcal{L}_{\text{CE}}(\hat{P}(\text{Col}), X)$$
  Structure Probing Accuracy is computed as the average of the Row and Column classification accuracies on a 20% validation split.

* **Motion Task (Motion Magnitude Classification)**:
  We compute the dense frame-difference motion map for the token's spatial-temporal tubelet. The continuous motion magnitude is binned into 5 categories using quantiles ($20\%, 40\%, 60\%, 80\%$) to ensure a balanced target distribution:
  $$\hat{P}(\text{Motion} = m) = \text{Softmax}(W_{\text{motion}}^{(l)} z_{t, y, x}^{(l)} + b_{\text{motion}}^{(l)})$$
  $$\mathcal{L}_{\text{motion}} = \mathcal{L}_{\text{CE}}(\hat{P}(\text{Motion}), M)$$
  Motion Probing Accuracy is evaluated on the validation split.

### 2. Representational Similarity Analysis (RSA): Context Leakage
To trace how self-attention integrates global context across layer depths, we compute the representational cosine similarity between high-motion foreground tokens and static background tokens:
1. Let $Z_{\text{fg}}^{(l)}$ be the L2-normalized representations of the top 15% high-motion tokens, and $Z_{\text{bg}}^{(l)}$ be the L2-normalized representations of the bottom 30% low-motion tokens at layer $l$.
2. The **Context Leakage (Cross-Similarity)** score is defined as:
   $$\text{Leakage}(l) = \frac{1}{|Z_{\text{fg}}^{(l)}| |Z_{\text{bg}}^{(l)}|} \sum_{u \in Z_{\text{fg}}^{(l)}} \sum_{v \in Z_{\text{bg}}^{(l)}} \cos(u, v)$$
3. **Foreground and Background Coherences** represent intra-class similarities:
   $$\text{Coherence}_{\text{fg}}(l) = \frac{1}{|Z_{\text{fg}}^{(l)}|^2} \sum_{u \in Z_{\text{fg}}^{(l)}} \sum_{v \in Z_{\text{fg}}^{(l)}} \cos(u, v)$$
   $$\text{Coherence}_{\text{bg}}(l) = \frac{1}{|Z_{\text{bg}}^{(l)}|^2} \sum_{u \in Z_{\text{bg}}^{(l)}} \sum_{v \in Z_{\text{bg}}^{(l)}} \cos(u, v)$$

---

## Quantitative Metrics Summary

Probing accuracies and RSA scores computed across the transformer layers on `sample_video.mp4`:

| Layer Index | Layer Type / Description | Structure Acc (Chance: 0.0625) | Motion Acc (Chance: 0.2000) | Context Leakage (Cross-Similarity) | FG Coherence | BG Coherence |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Layer 0** | Patch Embeddings | 0.2081 | 0.4015 | 0.1096 | 0.2618 | 0.1802 |
| **Layer 2** | Early Transformer Block | 0.2962 | 0.5302 | 0.8125 | 0.8404 | 0.8521 |
| **Layer 4** | Early Transformer Block | 0.5488 | 0.5772 | 0.8592 | 0.8872 | 0.8920 |
| **Layer 6** | Mid-Early Transformer Block | 0.7889 | 0.5595 | 0.8955 | 0.9169 | 0.9208 |
| **Layer 8** | Mid-Early Transformer Block | **0.9094** | 0.5863 | **0.8998** | 0.9228 | 0.9234 |
| **Layer 10**| Middle Transformer Block | 0.8719 | **0.6174** | 0.8882 | 0.9150 | 0.9110 |
| **Layer 12**| Middle Transformer Block | 0.7974 | 0.6138 | 0.8731 | 0.9079 | 0.8967 |
| **Layer 14**| Mid-Late Transformer Block | 0.7074 | 0.6089 | 0.8646 | 0.9048 | 0.8880 |
| **Layer 16**| Mid-Late Transformer Block | 0.6290 | 0.5833 | 0.8417 | 0.8931 | 0.8687 |
| **Layer 18**| Late Transformer Block | 0.5775 | 0.5668 | 0.7834 | 0.8601 | 0.8228 |
| **Layer 20**| Late Transformer Block | 0.5482 | 0.5259 | 0.6849 | 0.8152 | 0.7388 |
| **Layer 22**| Late Transformer Block | 0.5131 | 0.5137 | 0.5750 | 0.7818 | 0.6433 |
| **Layer 24**| Final Transformer Output | 0.5040 | 0.5101 | 0.2416 | 0.6974 | 0.3860 |

---

## Scientific and Architectural Insights

### 1. The Structure vs. Motion Trade-Off
* **Spatial/Structure Localization**: Patch embeddings start with a low spatial decodability (0.2081) as raw tokens haven't interacted. As they pass through self-attention, coordinates are resolved, peaking at **Layer 8 (0.9094)**. Deeper layers (10-24) see a steady decline in structure accuracy down to **0.5040** at the final output. This occurs as representation abstraction increases, transforming grid-based positional details into translation-invariant semantic concepts.
* **Motion / Temporal Dynamics**: Motion magnitude starts moderately high at the patch embeddings (0.4015) since adjacent frame differences correlate with raw pixel differences. It peaks in the middle layers (**Layer 10: 0.6174**), showing that intermediate representation spaces are optimized to capture temporal trajectories.

### 2. Context Leakage and Foreground-Background Disentanglement
* **Early Attention Mixing**: Between Layer 0 and Layer 8, foreground-background cross-similarity climbs rapidly from **0.1096 to 0.8998**. This illustrates how self-attention distributes global context: static background tokens "leak" representations to incorporate moving objects' features, forming a unified scene graph.
* **Late Disentanglement / Semantic Separation**: A striking finding is the sharp drop in cross-similarity in the deep layers (Layers 18 to 24), falling all the way to **0.2416**. Despite full global receptive fields, the model explicitly disentangles static background context from moving foreground actors in the final layers. This serves the self-supervised objective of predicting masked regions (since separate representations for actor and scene allow clean composition and target-prediction).
* This is further verified by looking at **Background Coherence**, which collapses from **0.9234** at Layer 8 to **0.3860** at Layer 24. While foreground features maintain high coherence (**0.6974**) to track the bowler's action, background tokens adapt to local structures rather than sharing a generic scene representation.

### 3. Visual Layer-wise PCA Evolution
* **Layer 2 (Local detail)**: Displays high-frequency, noisy color distributions that mirror pixel-level edges and textures rather than semantic regions.
* **Layer 8 (Coherent tracking grid)**: Highly resolved object boundaries with a clear layout. The moving bowler and bowling lane are sharply outlined in distinct global PCA coordinates.
* **Layer 24 (Semantic Abstraction & Disentanglement)**: Smooth, broad semantic colors. High-frequency edges vanish, replaced by unified regional masks. The background lane/pins share a highly uniform representation (confirming the background coherence collapse), and the bowler is represented as a separate, distinct color region.
