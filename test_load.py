import torch
import sys
sys.path.append("/Users/loganchoi/Desktop/vjepa2/vjepa2")
from src.models.predictor import VisionTransformerPredictor

device = "cpu"
predictor = VisionTransformerPredictor(
    img_size=(448, 448),
    patch_size=16,
    num_frames=16,
    tubelet_size=1,
    embed_dim=384,
    predictor_embed_dim=192,
    out_embed_dim=384,
    depth=4,
    num_heads=6,
    use_mask_tokens=True,
    num_mask_tokens=1
).to(device)

state_dict = torch.load("/Users/loganchoi/Desktop/vjepa2/dinov3_exploration/pathway_1/predictor_pathway1.pth", map_location=device)
missing, unexpected = predictor.load_state_dict(state_dict, strict=False)
print("Missing:", missing)
print("Unexpected:", unexpected)
