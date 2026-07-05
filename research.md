# V-JEPA 2 and 2.1 Research and Experimentation Options

This document maps out four potential research directions for investigating the V-JEPA 2 and 2.1 self-supervised representation models.

---

## Option 1: Dense Feature Visualization and PCA-based Temporal Tracking

V-JEPA 2.1 is optimized to produce temporally consistent, high-quality dense features. This research direction explores what the frozen model representations capture and track across frame transformations.

* **Research Focus:** Analyze and visualize the continuity of spatial-temporal embeddings in videos.
* **Core Components:**
  * Extraction Script: Extract token embeddings using pre-trained V-JEPA models (loaded via Hugging Face or PyTorch Hub).
  * Projection: Apply Principal Component Analysis (PCA) across spatial and temporal dimensions of video feature maps.
  * Visualization: Project the top PCA components back into RGB channels, creating visualization overlays that track object parts and shapes across motions.
* **Code Reference:**
  * Existing inference test base: [statejepa/test_vjepa.py](file:///Users/loganchoi/Desktop/vjepa2/statejepa/test_vjepa.py)
  * Hugging Face collection: facebook/vjepa2-vitl-fpc64-256

---

## Option 2: Ablation and Optimization of Spatial-Temporal Masking Strategies

Masking parameters define the complexity of the self-supervised target prediction. This research direction examines how different partitioning configurations impact representations and convergence.

* **Research Focus:** Alter spatial scale, temporal scale, keep-ratio, and aspect ratio of context/target tokens to study pre-training behavior.
* **Core Components:**
  * Mask Generator: Inspect and modify how masks are sampled and generated.
  * Motion-guided Masking: Implement custom masking methods that preserve high-motion areas (using frame differences or optical flow) as either context or targets to study performance changes.
* **Code Reference:**
  * 3D Mask Generator: [vjepa2/src/masks/multiseq_multiblock3d.py](file:///Users/loganchoi/Desktop/vjepa2/vjepa2/src/masks/multiseq_multiblock3d.py)
  * Pre-training config yaml layouts: [vjepa2/configs/train_2_1/](file:///Users/loganchoi/Desktop/vjepa2/vjepa2/configs/train_2_1)

---

## Option 3: Hierarchical Layer Representation Analysis

V-JEPA 2.1 leverages Deep Self-Supervision by distilling representations across intermediate layer depths. This research direction investigates what features are encoded at varying depths of the Vision Transformer.

* **Research Focus:** Measure feature abstraction (e.g., structure vs. motion) at different transformer layers.
* **Core Components:**
  * Layer Probe: Retrieve embeddings from distinct layer depths (e.g., layers 2, 5, 8, 11 in a 12-layer ViT-L model).
  * Downstream Probing: Train classification probes on separate intermediate outputs using downstream tasks (e.g., action classification) to identify where specific semantic and visual traits emerge.
* **Code Reference:**
  * Transformer implementation: [vjepa2/app/vjepa_2_1/models/vision_transformer.py](file:///Users/loganchoi/Desktop/vjepa2/vjepa2/app/vjepa_2_1/models/vision_transformer.py) (see hierarchical layers configurations)
  * Evaluation script: [vjepa2/evals/video_classification_frozen/eval.py](file:///Users/loganchoi/Desktop/vjepa2/vjepa2/evals/video_classification_frozen/eval.py)

---

## Option 4: Action-Conditioned World Modeling

For robotic manipulation and prediction tasks, V-JEPA can be conditioned on action trajectories to predict future states in representation space.

* **Research Focus:** Assess the accuracy of predicting latent representations conditioned on physical/control actions.
* **Core Components:**
  * Action Predictor: Investigate how control signals guide state transition predictions.
  * Post-training: Configure a small model checkpoint to predict future frame representations using sequence actions.
* **Code Reference:**
  * Action-Conditioned Predictor: [vjepa2/src/models/ac_predictor.py](file:///Users/loganchoi/Desktop/vjepa2/vjepa2/src/models/ac_predictor.py)
