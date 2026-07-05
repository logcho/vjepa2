# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from logging import getLogger
from multiprocessing import Value

import torch

_GLOBAL_SEED = 0
logger = getLogger()


class MotionMaskCollator(object):

    def __init__(
        self,
        cfgs_mask,
        dataset_fpcs,
        crop_size=(224, 224),
        patch_size=(16, 16),
        tubelet_size=2,
    ):
        super(MotionMaskCollator, self).__init__()

        self.mask_generators = dict()
        for fpc in dataset_fpcs:
            self.mask_generators[fpc] = []
            for m in cfgs_mask:
                mask_generator = MotionMaskGenerator(
                    crop_size=crop_size,
                    num_frames=fpc,
                    spatial_patch_size=patch_size,
                    temporal_patch_size=tubelet_size,
                    spatial_pred_mask_scale=m.get("spatial_scale"),
                    temporal_pred_mask_scale=m.get("temporal_scale"),
                    aspect_ratio=m.get("aspect_ratio"),
                    npred=m.get("num_blocks"),
                    max_context_frames_ratio=m.get("max_temporal_keep", 1.0),
                    max_keep=m.get("max_keep", None),
                    full_complement=m.get("full_complement", False),
                    pred_full_complement=m.get("pred_full_complement", False),
                    inv_block=m.get("inv_block", False),
                    motion_mode=m.get("motion_mode", "random"),
                    motion_keep_ratio=m.get("motion_keep_ratio", None),
                    motion_pred_ratio=m.get("motion_pred_ratio", None),
                    motion_temperature=m.get("motion_temperature", 0.05),
                )
                self.mask_generators[fpc].append(mask_generator)

    def step(self):
        for fpc in self.mask_generators:
            for mask_generator in self.mask_generators[fpc]:
                mask_generator.step()

    def __call__(self, batch):
        # Batch: [buffer, label, clip_indices] for video
        # or [buffer, label] for images
        filtered_batches = {fpc: [] for fpc in self.mask_generators}
        for sample in batch:
            # Check if sample is from video dataset (has clip_indices) or image dataset
            if len(sample) >= 3 and isinstance(sample[-1], (list, tuple)):
                try:
                    fpc = len(sample[-1][-1])
                except (TypeError, IndexError):
                    fpc = 1
            else:
                fpc = 1
            if fpc in filtered_batches:
                filtered_batches[fpc] += [sample]

        fpc_collations = []
        for fpc in filtered_batches:
            fpc_batch = filtered_batches[fpc]
            batch_size = len(fpc_batch)
            if batch_size == 0:
                continue
            collated_batch = torch.utils.data.default_collate(fpc_batch)
            
            # Extract video tensor for motion computation if available
            video_tensor_batch = None
            if len(collated_batch) > 0:
                first_elem = collated_batch[0]
                if isinstance(first_elem, list) and len(first_elem) > 0:
                    video_tensor_batch = first_elem[0]
                elif torch.is_tensor(first_elem):
                    video_tensor_batch = first_elem

            collated_masks_pred, collated_masks_enc = [], []
            for i, mask_generator in enumerate(self.mask_generators[fpc]):
                masks_enc, masks_pred = mask_generator(batch_size, video_tensor_batch=video_tensor_batch)
                collated_masks_enc.append(masks_enc)
                collated_masks_pred.append(masks_pred)
            fpc_collations += [
                (collated_batch, collated_masks_enc, collated_masks_pred)
            ]

        return fpc_collations


class MotionMaskGenerator(object):

    def __init__(
        self,
        crop_size=(224, 224),
        num_frames=16,
        spatial_patch_size=(16, 16),
        temporal_patch_size=2,
        spatial_pred_mask_scale=(0.2, 0.8),
        temporal_pred_mask_scale=(1.0, 1.0),
        aspect_ratio=(0.3, 3.0),
        npred=1,
        max_context_frames_ratio=1.0,
        max_keep=None,
        inv_block=False,
        full_complement=False,
        pred_full_complement=False,
        motion_mode="random",  # "random", "motion_context", "motion_target", "block_motion_context", "block_motion_target"
        motion_keep_ratio=None,
        motion_pred_ratio=None,
        motion_temperature=0.05,
    ):
        super(MotionMaskGenerator, self).__init__()
        if not isinstance(crop_size, tuple):
            crop_size = (crop_size,) * 2
        if not isinstance(spatial_patch_size, tuple):
            spatial_patch_size = (spatial_patch_size,) * 2
        self.crop_size = crop_size
        self.height, self.width = [
            crop_size[i] // spatial_patch_size[i] for i in (0, 1)
        ]
        self.duration = num_frames // temporal_patch_size
        self.full_complement = full_complement
        self.pred_full_complement = pred_full_complement

        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.aspect_ratio = aspect_ratio
        self.spatial_pred_mask_scale = spatial_pred_mask_scale
        self.temporal_pred_mask_scale = temporal_pred_mask_scale
        self.npred = npred
        self.max_context_duration = max(
            1, int(self.duration * max_context_frames_ratio)
        )
        self.max_keep = max_keep
        self._itr_counter = Value("i", -1)
        self.inv_block = inv_block

        # Motion-guided configuration
        self.motion_mode = motion_mode
        self.motion_temperature = motion_temperature
        
        # Set default ratios for sorting-based motion masking
        if motion_keep_ratio is None:
            self.motion_keep_ratio = 0.25
        else:
            self.motion_keep_ratio = motion_keep_ratio
            
        if motion_pred_ratio is None:
            self.motion_pred_ratio = 0.75
        else:
            self.motion_pred_ratio = motion_pred_ratio

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def compute_motion_map(self, video_tensor_batch):
        # video_tensor_batch shape: (B, C, T, H, W) or (B, T, C, H, W)
        shape = video_tensor_batch.shape
        if len(shape) != 5:
            # Fallback if tensor shape is unexpected
            return torch.zeros((shape[0], self.duration, self.height, self.width), device=video_tensor_batch.device)
            
        # Detect layout and permute to (B, C, T, H, W) if needed
        # Check if C is dimension 2
        if shape[2] == 3 and shape[1] != 3:
            video_tensor_batch = video_tensor_batch.permute(0, 2, 1, 3, 4)
            
        B, C, T, H, W = video_tensor_batch.shape
        device = video_tensor_batch.device
        
        # Absolute frame differences
        diff = torch.abs(video_tensor_batch[:, :, 1:] - video_tensor_batch[:, :, :-1])  # (B, C, T-1, H, W)
        diff_gray = torch.mean(diff, dim=1)  # (B, T-1, H, W)
        
        # Pad to time dimension T
        diff_gray = torch.cat([diff_gray, diff_gray[:, -1:]], dim=1)  # (B, T, H, W)
        diff_gray = diff_gray.unsqueeze(1)  # (B, 1, T, H, W)
        
        kernel_t = self.temporal_patch_size
        kernel_h = self.spatial_patch_size[0]
        kernel_w = self.spatial_patch_size[1]
        
        # Pad spatial/temporal dimensions to match patch sizes
        pad_t = (kernel_t - T % kernel_t) % kernel_t
        pad_h = (kernel_h - H % kernel_h) % kernel_h
        pad_w = (kernel_w - W % kernel_w) % kernel_w
        
        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            diff_gray = torch.nn.functional.pad(diff_gray, (0, pad_w, 0, pad_h, 0, pad_t))
            
        # 3D pooling to match token resolution
        pooled = torch.nn.functional.avg_pool3d(
            diff_gray,
            kernel_size=(kernel_t, kernel_h, kernel_w),
            stride=(kernel_t, kernel_h, kernel_w)
        )  # (B, 1, duration, height, width)
        
        motion_map = pooled.squeeze(1)
        # Crop back to matching dims
        motion_map = motion_map[:, :self.duration, :self.height, :self.width]
        return motion_map

    def _sample_block_size(
        self, generator, temporal_scale, spatial_scale, aspect_ratio_scale
    ):
        _rand = torch.rand(1, generator=generator).item()
        min_t, max_t = temporal_scale
        temporal_mask_scale = min_t + _rand * (max_t - min_t)
        t = max(1, int(self.duration * temporal_mask_scale))

        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = spatial_scale
        spatial_mask_scale = min_s + _rand * (max_s - min_s)
        spatial_num_keep = int(self.height * self.width * spatial_mask_scale)

        _rand = torch.rand(1, generator=generator).item()
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)

        h = int(round(math.sqrt(spatial_num_keep * aspect_ratio)))
        w = int(round(math.sqrt(spatial_num_keep / aspect_ratio)))
        h = min(h, self.height)
        w = min(w, self.width)

        return (t, h, w)

    def _sample_block_mask(self, b_size, motion_map_b=None):
        t, h, w = b_size
        
        # Block-level motion-guided sampling
        if motion_map_b is not None and self.motion_mode in ("block_motion_target", "block_motion_context"):
            D_out = self.duration - t + 1
            H_out = self.height - h + 1
            W_out = self.width - w + 1
            
            if D_out > 0 and H_out > 0 and W_out > 0:
                device = motion_map_b.device
                kernel = torch.ones((1, 1, t, h, w), dtype=motion_map_b.dtype, device=device)
                motion_padded = motion_map_b.unsqueeze(0).unsqueeze(0)  # (1, 1, duration, height, width)
                
                # Compute block sums using Conv3D
                block_sums = torch.nn.functional.conv3d(motion_padded, kernel, stride=1, padding=0)
                block_sums = block_sums.squeeze(0).squeeze(0)  # (D_out, H_out, W_out)
                
                # Average motion score per block
                block_scores = block_sums / (t * h * w)
                block_scores_flat = block_scores.flatten()
                
                # Determine probabilities
                if self.motion_mode == "block_motion_target":
                    scores = block_scores_flat / self.motion_temperature
                else:  # block_motion_context
                    scores = -block_scores_flat / self.motion_temperature
                    
                # Numerical stability adjustment
                scores = scores - torch.max(scores)
                probs = torch.softmax(scores, dim=0)
                
                # Sample flat index
                flat_idx = torch.multinomial(probs, 1).item()
                
                # Decode 3D index
                start = flat_idx // (H_out * W_out)
                rem = flat_idx % (H_out * W_out)
                top = rem // W_out
                left = rem % W_out
            else:
                top = torch.randint(0, self.height - h + 1, (1,)).item()
                left = torch.randint(0, self.width - w + 1, (1,)).item()
                start = torch.randint(0, self.duration - t + 1, (1,)).item()
        else:
            top = torch.randint(0, self.height - h + 1, (1,)).item()
            left = torch.randint(0, self.width - w + 1, (1,)).item()
            start = torch.randint(0, self.duration - t + 1, (1,)).item()

        device = motion_map_b.device if motion_map_b is not None else "cpu"
        mask = torch.ones((self.duration, self.height, self.width), dtype=torch.int32, device=device)
        mask[start : start + t, top : top + h, left : left + w] = 0

        if self.max_context_duration < self.duration:
            mask[self.max_context_duration :, :, :] = 0

        return mask

    def __call__(self, batch_size, video_tensor_batch=None):
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            temporal_scale=self.temporal_pred_mask_scale,
            spatial_scale=self.spatial_pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio,
        )

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_enc = min_keep_pred = self.duration * self.height * self.width
        
        # 1. Compute motion maps if applicable
        motion_map = None
        if video_tensor_batch is not None and self.motion_mode != "random":
            with torch.no_grad():
                motion_map = self.compute_motion_map(video_tensor_batch)

        for b in range(batch_size):
            motion_map_b = motion_map[b] if motion_map is not None else None
            
            # Token-level sorting masking
            if motion_map_b is not None and self.motion_mode in ("motion_context", "motion_target"):
                N = self.duration * self.height * self.width
                motion_flat = motion_map_b.flatten()
                
                # Sort indices of tokens by motion descending
                sorted_indices = torch.argsort(motion_flat, descending=True)
                
                # Calculate number of context (encoder) and target (predictor) tokens
                K_enc = int(N * self.motion_keep_ratio)
                K_pred = int(N * self.motion_pred_ratio)
                
                K_enc = max(1, min(K_enc, N - 1))
                K_pred = max(1, min(K_pred, N - K_enc))
                
                if self.motion_mode == "motion_context":
                    # High motion as context: first K_enc are context, next K_pred are targets
                    mask_e = sorted_indices[:K_enc]
                    mask_p = sorted_indices[K_enc : K_enc + K_pred]
                else:  # motion_target
                    # High motion as target: first K_pred are targets, next K_enc are context
                    mask_p = sorted_indices[:K_pred]
                    mask_e = sorted_indices[K_pred : K_pred + K_enc]
                
                # Convert to CPU for standard collator collation compatibility
                mask_p = mask_p.to("cpu")
                mask_e = mask_e.to("cpu")
                
                min_keep_pred = min(min_keep_pred, len(mask_p))
                min_keep_enc = min(min_keep_enc, len(mask_e))
                collated_masks_pred.append(mask_p)
                collated_masks_enc.append(mask_e)
                
            else:
                # Block-level sampling (supports both block_motion_target/block_motion_context and random)
                empty_context = True
                while empty_context:
                    mask_e = torch.ones(
                        (self.duration, self.height, self.width), dtype=torch.int32, device="cpu"
                    )
                    for _ in range(self.npred):
                        mask_e *= self._sample_block_mask(p_size, motion_map_b=motion_map_b).to("cpu")
                    mask_e = mask_e.flatten()

                    mask_p = torch.argwhere(mask_e == 0).squeeze()
                    mask_e = torch.nonzero(mask_e).squeeze()

                    # Handle single token squeeze edge-cases
                    if mask_p.ndim == 0:
                        mask_p = mask_p.unsqueeze(0)
                    if mask_e.ndim == 0:
                        mask_e = mask_e.unsqueeze(0)

                    empty_context = len(mask_e) == 0
                    if not empty_context:
                        min_keep_pred = min(min_keep_pred, len(mask_p))
                        min_keep_enc = min(min_keep_enc, len(mask_e))
                        collated_masks_pred.append(mask_p)
                        collated_masks_enc.append(mask_e)

        if self.max_keep is not None:
            min_keep_enc = min(min_keep_enc, self.max_keep)

        collated_masks_enc = [cm[:min_keep_enc] for cm in collated_masks_enc]
        collated_masks_pred = [cm[:min_keep_pred] for cm in collated_masks_pred]
        
        # Post-process full complements
        if self.full_complement:
            collated_masks_pred = [
                torch.tensor(
                    sorted(
                        list(
                            set(range(int(self.duration * self.height * self.width)))
                            - set(cm.tolist())
                        )
                    ),
                    dtype=cm.dtype,
                    device=cm.device
                )
                for cm in collated_masks_enc
            ]
        elif self.pred_full_complement:
            collated_masks_enc = [
                torch.tensor(
                    sorted(
                        list(
                            set(range(int(self.duration * self.height * self.width)))
                            - set(cm.tolist())
                        )
                    ),
                    dtype=cm.dtype,
                    device=cm.device
                )
                for cm in collated_masks_pred
            ]

        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)

        if self.inv_block:
            return collated_masks_pred, collated_masks_enc
        else:
            return collated_masks_enc, collated_masks_pred
